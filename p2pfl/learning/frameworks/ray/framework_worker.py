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
"""
Framework Worker Actor.

A shared Ray actor that loads the ML framework once and serves multiple nodes.
Replaces the 1-actor-per-node pattern to avoid redundant framework loading.
"""

import asyncio
import gc
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import ray

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.settings import Settings
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger

# Warn when process RSS exceeds this fraction of total system memory
_MEMORY_WARN_THRESHOLD = 0.8
# Pause new training when RSS exceeds this fraction (backpressure)
_MEMORY_BACKPRESSURE_THRESHOLD = 0.9
_BACKPRESSURE_POLL_INTERVAL = 2.0  # seconds
_BACKPRESSURE_MAX_WAIT = 30.0  # seconds


@dataclass
class NodeSlot:
    """Registry entry for a node managed by the worker actor."""

    learner: Learner
    last_used: float = field(default_factory=time.time)


@ray.remote
class FrameworkWorkerActor:
    """Ray actor that loads the ML framework once and multiplexes across nodes.

    Each node registers its Learner instance. The actor delegates operations
    to the appropriate learner, guarding compute-heavy ops (fit, train_on_batch,
    evaluate) with an asyncio.Semaphore to limit concurrency.

    Args:
        max_concurrent: Maximum number of concurrent training operations.

    """

    def __init__(self, max_concurrent: int = 2) -> None:
        """Initialize the worker actor."""
        self._registry: dict[str, NodeSlot] = {}
        self._max_concurrent = max_concurrent
        self._training_semaphore = asyncio.Semaphore(max_concurrent)

        # Limit PyTorch intra-op threads to prevent CPU saturation
        # when multiple nodes train concurrently on the same machine.
        torch_threads = Settings.training.TORCH_NUM_THREADS
        if torch_threads <= 0:
            total_cpus = os.cpu_count() or 1
            torch_threads = max(1, total_cpus // max(max_concurrent, 1))
        try:
            import torch
            torch.set_num_threads(torch_threads)
            logger.info("FrameworkWorkerActor", f"PyTorch intra-op threads set to {torch_threads}")
        except ImportError:
            pass
        # Also set environment variables for other BLAS backends
        os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    # --- Helpers ---

    def _get_learner(self, node_id: str) -> Learner:
        """Look up the learner for a node, raising KeyError if not found."""
        try:
            return self._registry[node_id].learner
        except KeyError:
            raise KeyError(f"Node '{node_id}' is not registered in this worker actor")

    def _touch(self, node_id: str) -> None:
        """Update last_used timestamp for a node."""
        if node_id in self._registry:
            self._registry[node_id].last_used = time.time()

    async def _check_memory(self, node_id: str) -> None:
        """Apply backpressure if memory exceeds threshold, otherwise warn."""
        try:
            import psutil
        except ImportError:
            return

        waited = 0.0
        while True:
            process = psutil.Process()
            rss = process.memory_info().rss
            total = psutil.virtual_memory().total
            usage_pct = rss / total

            if usage_pct <= _MEMORY_WARN_THRESHOLD:
                return

            if usage_pct > _MEMORY_BACKPRESSURE_THRESHOLD:
                if waited >= _BACKPRESSURE_MAX_WAIT:
                    logger.warning(
                        node_id,
                        f"Memory backpressure timeout after {_BACKPRESSURE_MAX_WAIT:.0f}s "
                        f"({usage_pct:.0%} used), proceeding anyway",
                    )
                    return
                logger.warning(
                    node_id,
                    f"Memory backpressure: {rss / 1024**2:.0f}MB "
                    f"({usage_pct:.0%}), pausing training...",
                )
                await asyncio.sleep(_BACKPRESSURE_POLL_INTERVAL)
                waited += _BACKPRESSURE_POLL_INTERVAL
                continue

            # Between warn and backpressure threshold — warn and proceed
            lru_node = min(
                ((nid, slot) for nid, slot in self._registry.items() if nid != node_id),
                key=lambda x: x[1].last_used,
                default=None,
            )
            lru_info = f" LRU node: {lru_node[0]}" if lru_node else ""
            logger.warning(
                node_id,
                f"Memory pressure: {rss / 1024**2:.0f}MB "
                f"({usage_pct:.0%} of {total / 1024**2:.0f}MB).{lru_info}",
            )
            return

    def get_memory_stats(self) -> dict:
        """Return memory usage stats for monitoring."""
        stats: dict[str, Any] = {"node_count": len(self._registry)}
        try:
            import psutil

            process = psutil.Process()
            rss = process.memory_info().rss
            available = psutil.virtual_memory().total
            stats["rss_mb"] = round(rss / 1024**2, 1)
            stats["total_mb"] = round(available / 1024**2, 1)
            stats["usage_pct"] = round(rss / available * 100, 1)
        except ImportError:
            stats["rss_mb"] = -1
        return stats

    # --- Node registration ---

    def register_node(self, node_id: str, learner: Learner) -> None:
        """Register a node with its learner instance.

        Args:
            node_id: Unique identifier for the node.
            learner: The Learner instance for this node.

        """
        self._registry[node_id] = NodeSlot(learner=learner)
        self._touch(node_id)
        logger.debug(node_id, "Registered in framework worker actor")

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from the registry.

        Args:
            node_id: The node to unregister.

        """
        del self._registry[node_id]
        logger.debug(node_id, "Unregistered from framework worker actor")

    def rekey_node(self, old_id: str, new_id: str) -> None:
        """Change a node's key in the registry (e.g. after address change).

        Args:
            old_id: The current node identifier.
            new_id: The new node identifier.

        """
        slot = self._registry.pop(old_id)
        self._registry[new_id] = slot
        self._touch(new_id)
        logger.debug(new_id, f"Rekeyed from '{old_id}' in framework worker actor")

    def rekey_and_set_address(self, old_id: str, new_id: str) -> str:
        """Atomically rekey a node and set its address in one RPC.

        Combines rekey_node + set_address to avoid inconsistent state if
        one of two separate RPCs were to fail.

        Args:
            old_id: The current node identifier.
            new_id: The new node identifier / address.

        Returns:
            The new address.

        """
        slot = self._registry.pop(old_id)
        self._registry[new_id] = slot
        self._touch(new_id)
        result = self._get_learner(new_id).set_address(new_id)
        logger.debug(new_id, f"Rekeyed from '{old_id}' and set address in framework worker actor")
        return result

    # --- Model / data operations ---

    def set_model(self, node_id: str, model: P2PFLModel | Any) -> None:
        """Set the model on a node's learner.

        Args:
            node_id: The target node.
            model: The model (or parameters) to set.

        """
        self._touch(node_id)
        self._get_learner(node_id).set_model(model)

    def get_model(self, node_id: str) -> P2PFLModel:
        """Get the model from a node's learner.

        Ray automatically places the return value in the object store,
        so explicit ray.put() is unnecessary.

        Args:
            node_id: The target node.

        Returns:
            The model instance.

        """
        self._touch(node_id)
        return self._get_learner(node_id).get_model()

    def set_data(self, node_id: str, data: P2PFLDataset) -> None:
        """Set the dataset on a node's learner.

        Args:
            node_id: The target node.
            data: The dataset to set.

        """
        self._touch(node_id)
        self._get_learner(node_id).set_data(data)

    def get_data(self, node_id: str) -> P2PFLDataset:
        """Get the dataset from a node's learner.

        Args:
            node_id: The target node.

        Returns:
            The node's dataset.

        """
        self._touch(node_id)
        return self._get_learner(node_id).get_data()

    # --- Configuration ---

    def set_address(self, node_id: str, address: str) -> str:
        """Set the address on a node's learner.

        Args:
            node_id: The target node.
            address: The address to set.

        Returns:
            The address that was set.

        """
        self._touch(node_id)
        return self._get_learner(node_id).set_address(address)

    def set_epochs(self, node_id: str, epochs: int) -> None:
        """Set the number of training epochs for a node.

        Args:
            node_id: The target node.
            epochs: Number of epochs.

        """
        self._touch(node_id)
        self._get_learner(node_id).set_epochs(epochs)

    def get_epochs(self, node_id: str) -> int:
        """Get the number of training epochs for a node.

        Args:
            node_id: The target node.

        Returns:
            The number of epochs.

        """
        self._touch(node_id)
        return self._get_learner(node_id).get_epochs()

    def set_steps_per_epoch(self, node_id: str, steps: int) -> None:
        """Set the steps per epoch for a node.

        Args:
            node_id: The target node.
            steps: Number of steps per epoch.

        """
        self._touch(node_id)
        self._get_learner(node_id).set_steps_per_epoch(steps)

    def get_steps_per_epoch(self, node_id: str) -> int | None:
        """Get the steps per epoch for a node.

        Args:
            node_id: The target node.

        Returns:
            The number of steps per epoch, or None.

        """
        self._touch(node_id)
        return self._get_learner(node_id).get_steps_per_epoch()

    def indicate_aggregator(self, node_id: str, aggregator: Aggregator) -> None:
        """Indicate the aggregator to a node's learner.

        Args:
            node_id: The target node.
            aggregator: The aggregator to indicate.

        """
        self._touch(node_id)
        self._get_learner(node_id).indicate_aggregator(aggregator)

    def update_callbacks_with_model_info(self, node_id: str) -> None:
        """Update callbacks with model info for a node.

        Args:
            node_id: The target node.

        """
        self._touch(node_id)
        self._get_learner(node_id).update_callbacks_with_model_info()

    def add_callback_info_to_model(self, node_id: str) -> None:
        """Add callback info to the model for a node.

        Args:
            node_id: The target node.

        """
        self._touch(node_id)
        self._get_learner(node_id).add_callback_info_to_model()

    def configure(self, node_id: str, **kwargs: Any) -> None:
        """Apply multiple configuration settings in one call.

        Supported kwargs: epochs, steps_per_epoch, aggregator,
        update_callbacks, add_callback_info.

        Args:
            node_id: The target node.
            **kwargs: Configuration key-value pairs.

        """
        self._touch(node_id)
        learner = self._get_learner(node_id)
        if "epochs" in kwargs:
            learner.set_epochs(kwargs["epochs"])
        if "steps_per_epoch" in kwargs:
            learner.set_steps_per_epoch(kwargs["steps_per_epoch"])
        if "aggregator" in kwargs:
            learner.indicate_aggregator(kwargs["aggregator"])
        if kwargs.get("update_callbacks"):
            learner.update_callbacks_with_model_info()
        if kwargs.get("add_callback_info"):
            learner.add_callback_info_to_model()

    def get_framework(self, node_id: str) -> str:
        """Get the framework name from a node's learner.

        Args:
            node_id: The target node.

        Returns:
            The framework name string.

        """
        self._touch(node_id)
        return self._get_learner(node_id).get_framework()

    # --- Async training operations (guarded by semaphore) ---

    async def fit(self, node_id: str) -> P2PFLModel:
        """Fit the model with batch-level interleaving.

        Uses train_on_batch() in a loop, acquiring/releasing the semaphore
        per batch so concurrent fit() calls can truly interleave. Falls back
        to learner.fit() if train_on_batch is not supported.

        Args:
            node_id: The target node.

        Returns:
            The fitted model (Ray places the return value in the object store automatically).

        """
        try:
            self._touch(node_id)
            await self._check_memory(node_id)
            learner = self._get_learner(node_id)

            # Try interleaved batch training first
            try:
                epochs = learner.get_epochs()
                steps_per_epoch = learner.get_steps_per_epoch()
                if steps_per_epoch:
                    batches_per_epoch = steps_per_epoch
                else:
                    data = learner.get_data()
                    num_samples = data.get_num_samples(train=True)
                    batch_size = Settings.training.DEFAULT_BATCH_SIZE
                    batches_per_epoch = max(1, -(-num_samples // batch_size))
                total_batches = epochs * batches_per_epoch

                for _ in range(total_batches):
                    async with self._training_semaphore:
                        await learner.train_on_batch()
                    await asyncio.sleep(0)  # yield for interleaving

            except NotImplementedError:
                # Framework doesn't support batch training — fall back to full fit
                async with self._training_semaphore:
                    await learner.fit()

            # Release batch training state (DataLoader, iterator, optimizer)
            # to free memory before serializing the model.
            for attr in ("_batch_dataloader", "_batch_iter", "_batch_optimizer"):
                if hasattr(learner, attr):
                    setattr(learner, attr, None)

            model = learner.get_model()
            gc.collect()
            return model
        except Exception as ex:
            logger.error(node_id, traceback.format_exc())
            logger.error(node_id, f"An error occurred during remote fit: {ex}")
            raise

    async def train_on_batch(self, node_id: str) -> P2PFLModel:
        """Train on one batch for a node.

        Guarded by the training semaphore to limit concurrency.

        Args:
            node_id: The target node.

        Returns:
            The model after batch training.

        """
        async with self._training_semaphore:
            try:
                self._touch(node_id)
                await self._check_memory(node_id)
                model = await self._get_learner(node_id).train_on_batch()
                gc.collect()
                return model
            except Exception as ex:
                logger.error(node_id, traceback.format_exc())
                logger.error(node_id, f"An error occurred during remote train_on_batch: {ex}")
                raise

    async def evaluate(self, node_id: str) -> dict[str, float]:
        """Evaluate the model for a node.

        Guarded by the training semaphore to limit concurrency.

        Args:
            node_id: The target node.

        Returns:
            Dictionary of evaluation metrics.

        """
        async with self._training_semaphore:
            try:
                self._touch(node_id)
                await self._check_memory(node_id)
                result = await self._get_learner(node_id).evaluate()
                gc.collect()
                return result
            except Exception as ex:
                logger.error(node_id, traceback.format_exc())
                logger.error(node_id, f"An error occurred during remote evaluation: {ex}")
                raise
