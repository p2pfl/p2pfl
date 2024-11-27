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
from typing import Dict, List, Optional, Union

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.callback import P2PFLCallback
from p2pfl.learning.frameworks.callback_factory import CallbackFactory
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class Learner(ABC):
    """
    Template to implement learning processes, including metric monitoring during training.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        self_addr: The address of the learner.

    """

    def __init__(
        self, model: P2PFLModel, data: P2PFLDataset, self_addr: str = "unknown-node", aggregator: Optional[Aggregator] = None
    ) -> None:
        """Initialize the learner."""
        self.model: P2PFLModel = model
        self.data: P2PFLDataset = data
        self._self_addr = self_addr
        self.callbacks: List[P2PFLCallback] = []
        if aggregator:
            self.callbacks = CallbackFactory.create_callbacks(framework=self.get_framework(), aggregator=aggregator)
        self.epochs: int = 1  # Default epochs

    def set_addr(self, addr: str) -> None:
        """
        Set the address of the learner.

        Args:
            addr: The address of the learner.

        """
        self._self_addr = addr

    def set_model(self, model: Union[P2PFLModel, List[np.ndarray], bytes]) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        if isinstance(model, P2PFLModel):
            self.model = model
        elif isinstance(model, (list, bytes)):
            self.model.set_parameters(model)

        # Update callbacks with model info
        self.update_callbacks_with_model_info()

    def get_model(self) -> P2PFLModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        return self.model

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner. It is used to fit the model.

        Args:
            data: The data of the learner.

        """
        self.data = data

    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The data of the learner.

        """
        return self.data

    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.

        """
        self.epochs = epochs

    def update_callbacks_with_model_info(self) -> None:
        """Update the callbacks with the model additional information."""
        new_info = self.model.get_info()
        for callback in self.callbacks:
            try:
                callback_name = callback.get_name()
                callback.set_info(new_info[callback_name])
            except KeyError:
                pass

    def add_callback_info_to_model(self) -> None:
        """Add the additional information from the callbacks to the model."""
        for c in self.callbacks:
            self.model.add_info(c.get_name(), c.get_info())

    @abstractmethod
    def fit(self) -> P2PFLModel:
        """Fit the model."""
        pass

    @abstractmethod
    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
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
