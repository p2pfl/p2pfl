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
"""Virtual Node Learner."""

import asyncio
import traceback
from typing import Any, TypeVar

import ray

from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.ray.placement_group_manager import PlacementGroupManager
from p2pfl.management.logger import logger
from p2pfl.utils.node_component import NodeComponent

_T = TypeVar("_T")


def _with_learner_delegates(cls: type[_T]) -> type[_T]:
    """Add async delegate methods for all public Learner methods."""
    import inspect

    for name, _ in inspect.getmembers(Learner, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue

        async def method(self: Any, *args: Any, _n: str = name, **kwargs: Any) -> Any:
            result = getattr(self._learner, _n)(*args, **kwargs)
            return await result if asyncio.iscoroutine(result) else result

        method.__name__ = name
        if name not in cls.__dict__:
            setattr(cls, name, method)
    return cls


@ray.remote
@_with_learner_delegates
class VirtualLearnerActor:
    """Ray actor wrapper for learners. Plain class to avoid ABC serialization issues with Ray."""

    def __init__(self, learner: Learner) -> None:
        """Initialize the actor with a learner instance."""
        self._learner = learner

    def set_model_ref(self, model_ref: ray.ObjectRef) -> None:
        """Set model from an object store reference (zero-copy on same node)."""
        model = ray.get(model_ref)
        self._learner.set_model(model)

    def get_model_ref(self) -> ray.ObjectRef:
        """Put model in object store and return ref (zero-copy on same node)."""
        return ray.put(self._learner.get_model())

    def set_data_ref(self, data_ref: ray.ObjectRef) -> None:
        """Set data from an object store reference (zero-copy on same node)."""
        data = ray.get(data_ref)
        self._learner.set_data(data)

    def configure(
        self,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        aggregator=None,
        update_callbacks: bool = False,
        add_callback_info: bool = False,
    ) -> None:
        """Apply multiple configuration settings in one call."""
        if epochs is not None:
            self._learner.set_epochs(epochs)
        if steps_per_epoch is not None:
            self._learner.set_steps_per_epoch(steps_per_epoch)
        if aggregator is not None:
            self._learner.indicate_aggregator(aggregator)
        if update_callbacks:
            self._learner.update_callbacks_with_model_info()
        if add_callback_info:
            self._learner.add_callback_info_to_model()

    async def fit(self) -> ray.ObjectRef:
        """Fit the model and return result via object store."""
        result = self._learner.fit()
        model = await result if asyncio.iscoroutine(result) else result
        return ray.put(model)

    async def train_on_batch(self) -> ray.ObjectRef:
        """Train on batch and return result via object store."""
        result = self._learner.train_on_batch()
        model = await result if asyncio.iscoroutine(result) else result
        return ray.put(model)


class VirtualNodeLearner(Learner):
    """Wrapper that runs a Learner as a Ray actor for distributed execution."""

    def __init__(self, learner: Learner) -> None:
        """Initialize the learner."""
        # Initialize base class attributes (callbacks, epochs, etc.)
        # without model/data since those live on the actor
        NodeComponent.__init__(self)
        self.callbacks: list = []
        self.epochs: int = 1
        self.steps_per_epoch: int | None = None

        pg_manager = PlacementGroupManager()
        pg = pg_manager.get_placement_group()

        self._pending_fit_ref: ray.ObjectRef | None = None

        self.actor = VirtualLearnerActor.options(  # type: ignore[attr-defined]
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        ).remote(learner)
        self.address = learner.address

    async def fit(self) -> P2PFLModel:
        """Fit the model. Actor returns model via object store — no extra get_model call."""
        try:
            self._pending_fit_ref = self.actor.fit.remote()
            model_ref = await self._pending_fit_ref
            self._pending_fit_ref = None
            return ray.get(model_ref)
        except ray.exceptions.TaskCancelledError:
            self._pending_fit_ref = None
            logger.info(self.address, "Fit was cancelled via interrupt_fit")
            return await self.aget_model()
        except Exception as ex:
            self._pending_fit_ref = None
            logger.error(self.address, traceback.format_exc())
            logger.error(self.address, f"An error occurred during remote fit: {ex}")
            raise ex

    async def train_on_batch(self) -> P2PFLModel:
        """Train on batch. Actor returns model via object store — no extra get_model call."""
        try:
            model_ref = await self.actor.train_on_batch.remote()
            return ray.get(model_ref)
        except Exception as ex:
            logger.error(self.address, traceback.format_exc())
            logger.error(self.address, f"An error occurred during remote train_on_batch: {ex}")
            raise ex

    async def interrupt_fit(self) -> None:
        """Interrupt the fit process by cancelling the pending Ray task."""
        if self._pending_fit_ref is not None:
            ray.cancel(self._pending_fit_ref, force=False)
            self._pending_fit_ref = None

    async def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        try:
            return await self.actor.evaluate.remote()
        except Exception as ex:
            logger.error(self.address, traceback.format_exc())
            logger.error(self.address, f"An error occurred during remote evaluation: {ex}")
            raise ex

    # Proxy configuration & lifecycle methods
    def set_address(self, address: str) -> str:
        """Set the address on both local and remote actor."""
        ray.get(self.actor.set_address.remote(address))
        # Cache because is expensive and highly used on logs
        self.address = address
        return super().set_address(address)

    def set_model(self, model) -> None:
        """Set the P2PFL model via object store for zero-copy transfer."""
        ref = ray.put(model)
        ray.get(self.actor.set_model_ref.remote(ref))

    def get_model(self) -> P2PFLModel:
        """Get the P2PFL model via object store for zero-copy transfer."""
        model_ref = ray.get(self.actor.get_model_ref.remote())
        return ray.get(model_ref)

    def set_data(self, data) -> None:
        """Set the data via object store for zero-copy transfer."""
        ref = ray.put(data)
        ray.get(self.actor.set_data_ref.remote(ref))

    def get_data(self):
        """Get the data from the remote actor."""
        return ray.get(self.actor.get_data.remote())

    def indicate_aggregator(self, aggregator) -> None:
        """Indicate the aggregator on the remote actor."""
        ray.get(self.actor.indicate_aggregator.remote(aggregator))

    def get_epochs(self) -> int:
        """Get the number of epochs from the remote actor."""
        return ray.get(self.actor.get_epochs.remote())

    def set_epochs(self, epochs: int) -> None:
        """Set the number of epochs on the remote actor."""
        ray.get(self.actor.set_epochs.remote(epochs))

    def get_steps_per_epoch(self) -> int | None:
        """Get the steps per epoch from the remote actor."""
        return ray.get(self.actor.get_steps_per_epoch.remote())

    def set_steps_per_epoch(self, steps: int) -> None:
        """Set the steps per epoch on the remote actor."""
        ray.get(self.actor.set_steps_per_epoch.remote(steps))

    def update_callbacks_with_model_info(self) -> None:
        """Update callbacks with model info on the remote actor."""
        ray.get(self.actor.update_callbacks_with_model_info.remote())

    def add_callback_info_to_model(self) -> None:
        """Add callback info to model on the remote actor."""
        ray.get(self.actor.add_callback_info_to_model.remote())

    def configure(self, **kwargs) -> None:
        """Apply multiple configuration settings in one remote call."""
        ray.get(self.actor.configure.remote(**kwargs))

    def get_framework(self) -> str:
        """Get the framework from the remote actor."""
        return ray.get(self.actor.get_framework.remote())

    # Async interface — true async using await on Ray ObjectRefs

    async def aset_model(self, model) -> None:
        """Async set_model via object store."""
        ref = ray.put(model)
        await self.actor.set_model_ref.remote(ref)

    async def aget_model(self) -> P2PFLModel:
        """Async get_model via object store."""
        model_ref = await self.actor.get_model_ref.remote()
        return ray.get(model_ref)

    async def aset_data(self, data) -> None:
        """Async set_data via object store."""
        ref = ray.put(data)
        await self.actor.set_data_ref.remote(ref)

    async def aset_address(self, address: str) -> str:
        """Async set_address on remote actor."""
        await self.actor.set_address.remote(address)
        self.address = address
        return address

    async def aconfigure(self, **kwargs) -> None:
        """Async batch configuration in one remote call."""
        await self.actor.configure.remote(**kwargs)
