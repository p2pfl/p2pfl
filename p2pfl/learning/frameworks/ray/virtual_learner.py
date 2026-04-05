#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Virtual Node Learner - thin proxy that delegates to a shared FrameworkWorkerActor."""

from typing import Any

import numpy as np
import ray

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.ray.worker_pool import WorkerPool
from p2pfl.management.logger import logger
from p2pfl.utils.node_component import NodeComponent

# Timeout for blocking ray.get() calls on config/metadata RPCs (seconds).
# Training methods use await and don't need this.
_RPC_TIMEOUT = 60


class VirtualNodeLearner(Learner):
    """Lightweight proxy that delegates all operations to a shared FrameworkWorkerActor via WorkerPool.

    Does NOT create its own Ray actor. Instead, it registers the learner with a
    worker obtained from the WorkerPool singleton and forwards all calls using
    the node key for dispatch.
    """

    def __init__(self, learner: Learner) -> None:
        """Initialize the virtual learner proxy.

        Args:
            learner: The concrete learner instance to register with the worker.

        """
        NodeComponent.__init__(self)
        self.callbacks: list[Any] = []
        self.epochs: int = 1
        self.steps_per_epoch: int | None = None

        # Derive node key from address or id
        self._node_key: str = learner.address or str(id(learner))
        self.address: str = learner.address

        # Get a worker from the pool and register
        self._pool = WorkerPool()
        self._worker = self._pool.assign_worker()
        ray.get(self._worker.register_node.remote(self._node_key, learner))
        self._pool.register_node(self._worker, self._node_key, learner)

        logger.debug(
            self._node_key,
            "VirtualNodeLearner registered with shared worker",
        )

    def __del__(self) -> None:
        """Unregister from worker actor and pool on garbage collection."""
        try:
            self._worker.unregister_node.remote(self._node_key)
            self._pool.unregister_node(self._worker, self._node_key)
        except Exception as e:
            # Best-effort cleanup; Ray may already be shut down.
            # Log instead of silencing so leaked registrations are visible.
            try:
                logger.debug(self._node_key, f"Cleanup during __del__ failed: {e}")
            except Exception:
                pass  # Logger itself may be torn down

    # --- Sync proxy methods ---

    def set_address(self, address: str) -> str:
        """Set address: atomically rekey the registration and update the worker."""
        old_key = self._node_key
        new_key = address
        ray.get(
            self._worker.rekey_and_set_address.remote(old_key, new_key),
            timeout=_RPC_TIMEOUT,
        )
        self._pool.rekey_node(self._worker, old_key, new_key)
        self._node_key = new_key
        self.address = new_key
        return address

    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """Set model via object store for zero-copy transfer."""
        ref = ray.put(model)
        try:
            ray.get(self._worker.set_model.remote(self._node_key, ref), timeout=_RPC_TIMEOUT)
        finally:
            del ref

    def get_model(self) -> P2PFLModel:
        """Get model from the worker."""
        return ray.get(self._worker.get_model.remote(self._node_key), timeout=_RPC_TIMEOUT)

    def set_data(self, data: P2PFLDataset) -> None:
        """Set data via object store for zero-copy transfer."""
        ref = ray.put(data)
        try:
            ray.get(self._worker.set_data.remote(self._node_key, ref), timeout=_RPC_TIMEOUT)
        finally:
            del ref

    def get_data(self) -> P2PFLDataset:
        """Get data from the worker."""
        return ray.get(self._worker.get_data.remote(self._node_key), timeout=_RPC_TIMEOUT)

    def set_epochs(self, epochs: int) -> None:
        """Set epochs on the worker."""
        ray.get(self._worker.set_epochs.remote(self._node_key, epochs), timeout=_RPC_TIMEOUT)

    def get_epochs(self) -> int:
        """Get epochs from the worker."""
        return ray.get(self._worker.get_epochs.remote(self._node_key), timeout=_RPC_TIMEOUT)

    def set_steps_per_epoch(self, steps: int) -> None:
        """Set steps per epoch on the worker."""
        ray.get(self._worker.set_steps_per_epoch.remote(self._node_key, steps), timeout=_RPC_TIMEOUT)

    def get_steps_per_epoch(self) -> int | None:
        """Get steps per epoch from the worker."""
        return ray.get(self._worker.get_steps_per_epoch.remote(self._node_key), timeout=_RPC_TIMEOUT)

    def indicate_aggregator(self, aggregator: Aggregator) -> None:
        """Indicate aggregator on the worker."""
        ray.get(self._worker.indicate_aggregator.remote(self._node_key, aggregator), timeout=_RPC_TIMEOUT)

    def update_callbacks_with_model_info(self) -> None:
        """Update callbacks with model info on the worker."""
        ray.get(self._worker.update_callbacks_with_model_info.remote(self._node_key), timeout=_RPC_TIMEOUT)

    def add_callback_info_to_model(self) -> None:
        """Add callback info to model on the worker."""
        ray.get(self._worker.add_callback_info_to_model.remote(self._node_key), timeout=_RPC_TIMEOUT)

    def configure(self, **kwargs: Any) -> None:
        """Apply multiple configuration settings in one remote call."""
        ray.get(self._worker.configure.remote(self._node_key, **kwargs), timeout=_RPC_TIMEOUT)

    def get_framework(self) -> str:
        """Get framework name from the worker."""
        return ray.get(self._worker.get_framework.remote(self._node_key), timeout=_RPC_TIMEOUT)

    # --- Async training methods ---

    async def fit(self) -> P2PFLModel:
        """Fit the model. Worker handles semaphore and returns model directly."""
        return await self._worker.fit.remote(self._node_key)

    async def train_on_batch(self) -> P2PFLModel:
        """Train on batch. Worker handles semaphore and returns model directly."""
        return await self._worker.train_on_batch.remote(self._node_key)

    async def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the worker."""
        return await self._worker.evaluate.remote(self._node_key)

    async def interrupt_fit(self) -> None:
        """Interrupt fit - no-op since fit runs inside the shared worker."""
        logger.info(self._node_key, "interrupt_fit called (no-op: fit runs inside worker)")

    # --- Async interface ---

    async def aset_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """Async set_model via object store."""
        ref = ray.put(model)
        try:
            await self._worker.set_model.remote(self._node_key, ref)
        finally:
            del ref

    async def aget_model(self) -> P2PFLModel:
        """Async get_model."""
        return await self._worker.get_model.remote(self._node_key)

    async def aset_data(self, data: P2PFLDataset) -> None:
        """Async set_data via object store."""
        ref = ray.put(data)
        try:
            await self._worker.set_data.remote(self._node_key, ref)
        finally:
            del ref

    async def aset_address(self, address: str) -> str:
        """Async set_address: atomically rekey and update on worker."""
        old_key = self._node_key
        new_key = address
        await self._worker.rekey_and_set_address.remote(old_key, new_key)
        self._pool.rekey_node(self._worker, old_key, new_key)
        self._node_key = new_key
        self.address = new_key
        return address

    async def aconfigure(self, **kwargs: Any) -> None:
        """Async batch configuration in one remote call."""
        await self._worker.configure.remote(self._node_key, **kwargs)
