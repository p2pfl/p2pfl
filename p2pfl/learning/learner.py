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
from typing import Any, Dict, List, Union

import numpy as np

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.p2pfl_model import P2PFLModel


class NodeLearner(ABC):
    """
    Template to implement learning processes, including metric monitoring during training.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        self_addr: The address of the learner.

    """

    def __init__(self, model: P2PFLModel, data: P2PFLDataset, self_addr: str, callbacks: List[Any]) -> None:
        """Initialize the learner."""
        self.model: P2PFLModel = model
        self.data: P2PFLDataset = data
        self._self_addr = self_addr
        self.callbacks: List[Any] = callbacks if callbacks is not None else []
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

    def set_callbacks_additional_info(self, callbacks: List[Any]) -> None:
        """
        Update the callbacks with the model additional information.

        Args:
            callbacks: The callbacks.

        """
        for callback in callbacks:
            callback.additional_info = self.model.additional_info

    def get_callbacks_additional_info(self, callbacks: List[Any]):
        """
        Get the additional information from the callbacks to update the learner's model.

        Args:
            callbacks: The callbacks.

        """
        for callback in callbacks:
            if hasattr(callback, "additional_info"):
                self.model.additional_info.update(callback.additional_info)

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

    @staticmethod
    @abstractmethod
    def get_framework() -> str:
        """
        Get the framework of the learner.

        Returns:
            The framework of the learner.

        """
        pass
