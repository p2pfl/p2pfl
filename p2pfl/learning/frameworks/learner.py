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
        self_addr: The address of the learner.

    """

    def __init__(self, model: P2PFLModel | None = None, data: P2PFLDataset | None = None, aggregator: Aggregator | None = None) -> None:
        """Initialize the learner."""
        # (addr) Super
        NodeComponent.__init__(self)
        # Indicate aggregator (init callbacks)
        self.callbacks: list[P2PFLCallback] = []
        if aggregator:
            self.indicate_aggregator(aggregator)
        self.epochs: int = 1  # Default epochs
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
    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.

        """
        self.epochs = epochs

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
    def fit(self) -> P2PFLModel:
        """Fit the model."""
        pass

    @abstractmethod
    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
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
