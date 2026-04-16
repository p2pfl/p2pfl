#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""WorkerPool singleton for managing FrameworkWorkerActor lifecycle."""

import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from p2pfl.learning.frameworks.ray.framework_worker import FrameworkWorkerActor
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

_HEADROOM = 10  # Extra concurrency slots for non-training RPCs


@dataclass
class _Registration:
    """Record of a node registration for crash recovery."""

    node_id: str
    _learner_ref: Any = field(repr=False)  # weakref to Learner — avoids pinning model weights in memory

    @property
    def learner(self) -> Any | None:
        """Dereference the weak reference, returning None if collected."""
        ref = self._learner_ref
        if ref is None:
            return None
        return ref() if isinstance(ref, weakref.ref) else ref

    @staticmethod
    def create(node_id: str, learner: Any) -> "_Registration":
        """Create a registration, using a weakref if the learner supports it."""
        try:
            ref = weakref.ref(learner)
        except TypeError:
            ref = learner  # Some objects can't be weakly referenced
        return _Registration(node_id=node_id, _learner_ref=ref)


class WorkerPool:
    """
    Singleton that manages FrameworkWorkerActor lifecycle and node-to-worker assignment.

    Detects cluster topology and spawns one worker per physical node (multi-node)
    or a single worker with all resources (single machine). Workers are assigned
    to callers via round-robin.
    """

    _instance: "WorkerPool | None" = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, *_args: object, **_kwargs: object) -> "WorkerPool":
        """Create or return the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the worker pool based on cluster topology."""
        with self.__class__._lock:
            if self._initialized:
                return
            self._do_init()

    def _do_init(self) -> None:
        """Actual initialization (called under lock)."""
        # Detect cluster topology
        alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
        resources = ray.cluster_resources()
        total_cpus = resources.get("CPU", 0.0)
        total_gpus = resources.get("GPU", 0.0)

        # Determine max_concurrent from settings
        pool_size = Settings.training.RAY_ACTOR_POOL_SIZE
        if pool_size <= 0:
            # Cap concurrent training to half the CPUs (rounded up), keeping
            # the other half free for system overhead, gossip, heartbeat, and
            # PyTorch intra-op threads.  Previous default (total - 1) caused
            # 100% CPU saturation on machines with many cores.
            usable_cpus = max(int(total_cpus) - 1, 1) if total_cpus > 1 else max(int(total_cpus), 1)
            max_concurrent = max(1, -(-usable_cpus // 2))  # ceil division
        else:
            max_concurrent = pool_size

        max_concurrency = max_concurrent + _HEADROOM

        self._workers: list = []
        self._assign_index: int = 0
        self._registrations: dict[int, list[_Registration]] = {}

        if len(alive_nodes) <= 1:
            # Single machine: one worker with minimal CPU reservation
            actor_options: dict = {
                "num_cpus": 1,
                "max_concurrency": max_concurrency,
            }
            if total_gpus > 0:
                actor_options["num_gpus"] = total_gpus

            worker = FrameworkWorkerActor.options(**actor_options).remote(max_concurrent)
            self._workers.append(worker)
            logger.info(
                "WorkerPool",
                f"Single-node mode: 1 worker, 1 CPU reserved, {total_gpus} GPUs, " f"max_concurrent={max_concurrent}",
            )
        else:
            # Multi-node: one worker per node with minimal CPU reservation
            num_nodes = len(alive_nodes)
            gpus_per_node = total_gpus / num_nodes if total_gpus > 0 else 0.0

            for node in alive_nodes:
                node_id = node["NodeID"]
                actor_options = {
                    "num_cpus": 1,
                    "max_concurrency": max_concurrency,
                    "scheduling_strategy": NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                }
                if gpus_per_node > 0:
                    actor_options["num_gpus"] = gpus_per_node

                worker = FrameworkWorkerActor.options(**actor_options).remote(max_concurrent)
                self._workers.append(worker)

            logger.info(
                "WorkerPool",
                f"Multi-node mode: {num_nodes} workers, 1 CPU/node reserved, "
                f"{gpus_per_node:.1f} GPUs/node, max_concurrent={max_concurrent}",
            )

        self._registrations = {i: [] for i in range(len(self._workers))}
        self._initialized = True

    def assign_worker(self):
        """
        Return the next worker via round-robin assignment.

        Returns:
            A FrameworkWorkerActor handle.

        """
        if not self._workers:
            raise RuntimeError("WorkerPool has no workers. Was it shut down?")
        worker = self._workers[self._assign_index % len(self._workers)]
        self._assign_index += 1
        return worker

    @property
    def worker_count(self) -> int:
        """Number of active workers."""
        return len(self._workers)

    def _worker_index(self, worker) -> int | None:
        """Find the index of a worker in the pool."""
        for i, w in enumerate(self._workers):
            if w is worker:
                return i
        return None

    def register_node(self, worker, node_id: str, learner) -> None:
        """Track a node registration for crash recovery."""
        idx = self._worker_index(worker)
        if idx is not None:
            self._registrations[idx].append(_Registration.create(node_id=node_id, learner=learner))

    def unregister_node(self, worker, node_id: str) -> None:
        """Remove a node registration record."""
        idx = self._worker_index(worker)
        if idx is not None:
            self._registrations[idx] = [r for r in self._registrations[idx] if r.node_id != node_id]

    def rekey_node(self, worker, old_id: str, new_id: str) -> None:
        """Update node_id in registration records."""
        idx = self._worker_index(worker)
        if idx is not None:
            for reg in self._registrations[idx]:
                if reg.node_id == old_id:
                    reg.node_id = new_id
                    break

    def recover_worker(self, worker_index: int) -> None:
        """Re-spawn a crashed worker and replay its registrations."""
        if worker_index >= len(self._workers):
            return

        old_registrations = self._registrations.get(worker_index, [])

        # Re-spawn with same options as original
        resources = ray.cluster_resources()
        total_cpus = resources.get("CPU", 1)
        total_gpus = resources.get("GPU", 0)
        num_workers = len(self._workers)

        pool_size = Settings.training.RAY_ACTOR_POOL_SIZE
        if pool_size <= 0:
            usable_cpus = max(int(total_cpus) - 1, 1) if total_cpus > 1 else max(int(total_cpus), 1)
            max_concurrent = max(1, -(-usable_cpus // 2))
        else:
            max_concurrent = pool_size

        max_concurrency = max_concurrent + _HEADROOM

        opts: dict[str, object] = {
            "num_cpus": 1,
            "max_concurrency": max_concurrency,
        }
        if total_gpus > 0:
            opts["num_gpus"] = total_gpus / num_workers

        new_worker = FrameworkWorkerActor.options(**opts).remote(max_concurrent)
        self._workers[worker_index] = new_worker

        # Replay registrations (skip any whose learner was garbage-collected)
        self._registrations[worker_index] = []
        for reg in old_registrations:
            learner = reg.learner
            if learner is None:
                logger.warning("WorkerPool", f"Skipping re-registration of '{reg.node_id}': learner was garbage-collected")
                continue
            try:
                ray.get(new_worker.register_node.remote(reg.node_id, learner))
                self._registrations[worker_index].append(reg)
                logger.info("WorkerPool", f"Re-registered node '{reg.node_id}' after worker recovery")
            except Exception as e:
                logger.error("WorkerPool", f"Failed to re-register node '{reg.node_id}': {e}")

    def shutdown(self) -> None:
        """Kill all workers and reset the singleton."""
        with self.__class__._lock:
            for worker in self._workers:
                ray.kill(worker)
            self._workers.clear()
            self._registrations.clear()
            self.__class__._instance = None
            self._initialized = False
