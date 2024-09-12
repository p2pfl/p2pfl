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

from typing import Dict, List, Union

import numpy as np

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.p2pfl_model import P2PFLModel


class NodeLearner:
    """
    Template to implement learning processes, including metric monitoring during training.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        self_addr: The address of the learner.

    """

    def __init__(self, model: P2PFLModel, data: P2PFLDataset, self_addr: str) -> None:
        """Initialize the learner."""
        raise NotImplementedError

    def set_addr(self, addr: str) -> None:
        """
        Set the address of the learner.

        Args:
            addr: The address of the learner.

        """
        raise NotImplementedError

    def set_model(self, model: Union[P2PFLModel, List[np.ndarray], bytes]) -> None:
        """
        Set the model of the learner (not wheights).

        Args:
            model: The model of the learner.

        Raises:
            ModelNotMatchingError: If the model is not matching the learner.

        """
        raise NotImplementedError

    def get_model(self) -> P2PFLModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        raise NotImplementedError

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner. It is used to fit the model.

        Args:
            data: The data of the learner.

        """
        raise NotImplementedError

    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The data of the learner.

        """
        raise NotImplementedError

    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.

        """
        raise NotImplementedError

    def fit(self) -> None:
        """Fit the model."""
        raise NotImplementedError

    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        raise NotImplementedError

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        raise NotImplementedError
