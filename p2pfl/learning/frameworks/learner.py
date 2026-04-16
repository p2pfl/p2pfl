#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

"""NodeLearning Interface - Template Pattern."""

from abc import ABC, abstractmethod

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.callback import P2PFLCallback
from p2pfl.learning.frameworks.callback_factory import CallbackFactory
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check


class Learner(ABC, NodeComponent):
    """
    Template to implement learning processes, including metric monitoring during training.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        aggregator: The aggregator used in the learning process.
        steps_per_epoch: The number of steps per epoch for the model.

    """

    def __init__(
        self,
        model: P2PFLModel | None = None,
        data: P2PFLDataset | None = None,
        aggregator: Aggregator | None = None,
        steps_per_epoch: int | None = None,
    ) -> None:
        """Initialize the learner."""
        # Super
        NodeComponent.__init__(self)
        # Indicate aggregator (init callbacks)
        self.callbacks: list[P2PFLCallback] = []
        if aggregator:
            self.indicate_aggregator(aggregator)
        self.epochs: int = 1  # Default epochs
        self.steps_per_epoch: int | None = steps_per_epoch
        # Model and data init (dummy if not)
        self.__model: P2PFLModel | None = None
        if model:
            self.set_model(model)
        self.__data: P2PFLDataset | None = None
        if data:
            self.set_data(data)

    @allow_no_addr_check
    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        if isinstance(model, P2PFLModel):
            self.__model = model
        elif isinstance(model, list | bytes):
            self.get_model().set_parameters(model)

        # Update callbacks with model info
        self.update_callbacks_with_model_info()

    @allow_no_addr_check
    def get_model(self) -> P2PFLModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        if self.__model is None:
            raise ValueError("Model not initialized, please ensure to set the model before accessing it. Use .set_model() method.")
        return self.__model

    @allow_no_addr_check
    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner. It is used to fit the model.

        Args:
            data: The data of the learner.

        """
        self.__data = data

    @allow_no_addr_check
    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The data of the learner.

        """
        if self.__data is None:
            raise ValueError("Data not initialized, please ensure to set the data before accessing it. Use .set_data() method.")
        return self.__data

    @allow_no_addr_check
    def indicate_aggregator(self, aggregator: Aggregator) -> None:
        """
        Indicate to the learner the aggregators that are being used in order to instantiate the callbacks.

        Args:
            aggregator: The aggregator used in the learning process.

        """
        if aggregator:
            self.callbacks = self.callbacks + CallbackFactory.create_callbacks(framework=self.get_framework(), aggregator=aggregator)

    @allow_no_addr_check
    def get_epochs(self) -> int:
        """
        Get the number of epochs of the model.

        Returns:
            The number of epochs of the model.

        """
        return self.epochs

    @allow_no_addr_check
    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.

        """
        self.epochs = epochs

    @allow_no_addr_check
    def get_steps_per_epoch(self) -> int | None:
        """
        Get the number of steps per epoch of the model.

        Returns:
            The number of steps per epoch of the model.

        """
        return self.steps_per_epoch

    @allow_no_addr_check
    def set_steps_per_epoch(self, steps_per_epoch: int) -> None:
        """
        Set the number of steps per epoch of the model.

        Args:
            steps_per_epoch: The number of steps per epoch of the model.

        """
        self.steps_per_epoch = steps_per_epoch

    @allow_no_addr_check
    def update_callbacks_with_model_info(self) -> None:
        """Update the callbacks with the model additional information."""
        new_info = self.get_model().get_info()
        for callback in self.callbacks:
            try:
                callback_name = callback.get_name()
                callback.set_info(new_info[callback_name])
            except KeyError:
                pass

    @allow_no_addr_check
    def add_callback_info_to_model(self) -> None:
        """Add the additional information from the callbacks to the model."""
        for c in self.callbacks:
            self.get_model().add_info(c.get_name(), c.get_info())

    @abstractmethod
    async def fit(self) -> P2PFLModel:
        """
        Fit the model.

        Returns:
            The model after fitting.

        """
        pass

    @abstractmethod
    async def train_on_batch(self) -> P2PFLModel:
        """
        Train the model on the next batch manually.

        Returns:
            The model after training on the batch.

        """
        pass

    @abstractmethod
    async def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        pass

    @abstractmethod
    async def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        pass

    @abstractmethod
    def get_framework(self) -> str:
        """
        Retrieve the learner name.

        Returns:
            The name of the learner class.

        """
        pass

    # Async interface — defaults delegate to sync methods.
    # VirtualNodeLearner overrides with true async implementations.

    @allow_no_addr_check
    async def aset_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """Async version of set_model. Default delegates to sync."""
        self.set_model(model)

    @allow_no_addr_check
    async def aget_model(self) -> P2PFLModel:
        """Async version of get_model. Default delegates to sync."""
        return self.get_model()

    @allow_no_addr_check
    async def aset_data(self, data: P2PFLDataset) -> None:
        """Async version of set_data. Default delegates to sync."""
        self.set_data(data)

    async def aset_address(self, address: str) -> str:
        """Async version of set_address. Default delegates to sync."""
        return self.set_address(address)

    @allow_no_addr_check
    async def aconfigure(self, **kwargs) -> None:
        """Async batch configuration. Default applies settings individually."""
        if "epochs" in kwargs:
            self.set_epochs(kwargs["epochs"])
        if "steps_per_epoch" in kwargs:
            self.set_steps_per_epoch(kwargs["steps_per_epoch"])
        if "aggregator" in kwargs:
            self.indicate_aggregator(kwargs["aggregator"])
        if kwargs.get("update_callbacks"):
            self.update_callbacks_with_model_info()
        if kwargs.get("add_callback_info"):
            self.add_callback_info_to_model()


class LearnerDecorator(Learner):
    """
    Decorator class for Learner. Delegates calls to a wrapped Learner instance.

    Allows for extension of Learner behavior without modifying the base class.
    """

    def __init__(self, learner: Learner):
        """
        Initialize the decorator with a Learner instance.

        Args:
            learner: The Learner instance to wrap.

        """
        self._learner = learner

    def set_address(self, address: str) -> str:
        """
        Set the address of the learner.

        Args:
            address: The address to set.

        Returns:
            The address of the learner.

        """
        return self._learner.set_address(address)

    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model to set.

        """
        self._learner.set_model(model)

    def get_model(self) -> P2PFLModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        return self._learner.get_model()

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner.

        Args:
            data: The P2PFLDataset to set.

        """
        self._learner.set_data(data)

    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The P2PFLDataset of the learner.

        """
        return self._learner.get_data()

    def indicate_aggregator(self, aggregator: Aggregator) -> None:
        """
        Indicate to the learner the aggregator being used.

        Args:
            aggregator: The Aggregator to indicate.

        """
        self._learner.indicate_aggregator(aggregator)

    def get_epochs(self) -> int:
        """
        Get the number of epochs of the model.

        Returns:
            The number of epochs of the model.

        """
        return self._learner.get_epochs()

    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs to set.

        """
        self._learner.set_epochs(epochs)

    def get_steps_per_epoch(self) -> int | None:
        """
        Get the number of steps per epoch of the model.

        Returns:
            The number of steps per epoch of the model.

        """
        return self._learner.get_steps_per_epoch()

    def set_steps_per_epoch(self, steps_per_epoch: int) -> None:
        """
        Set the number of steps per epoch of the model.

        Args:
            steps_per_epoch: The number of steps per epoch to set.

        """
        self._learner.set_steps_per_epoch(steps_per_epoch)

    def update_callbacks_with_model_info(self) -> None:
        """
        Update the callbacks with the model additional information.

        This method retrieves the model's information and updates each callback with it.
        """
        self._learner.update_callbacks_with_model_info()

    def add_callback_info_to_model(self) -> None:
        """
        Add the additional information from the callbacks to the model.

        This method iterates through all callbacks and adds their information to the model.
        """
        self._learner.add_callback_info_to_model()

    async def fit(self) -> P2PFLModel:
        """
        Fit the model using the learner's fit method.

        Returns:
            The P2PFLModel after fitting.

        """
        return await self._learner.fit()

    async def train_on_batch(self) -> P2PFLModel:
        """
        Train the model on the next batch using the learner's train_on_batch method.

        Returns:
            The P2PFLModel after training on the batch.

        """
        return await self._learner.train_on_batch()

    async def interrupt_fit(self) -> None:
        """
        Interrupt the fit process using the learner's interrupt_fit method.

        This method allows for graceful interruption of the training process.
        """
        await self._learner.interrupt_fit()

    async def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model using the learner's evaluate method.

        Returns:
            A dictionary containing the evaluation results, such as loss and accuracy.

        """
        return await self._learner.evaluate()

    def get_framework(self) -> str:
        """
        Retrieve the framework name of the learner.

        Returns:
            The name of the framework used by the learner.

        """
        return self._learner.get_framework()
