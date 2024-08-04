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

"""NodeLearning Interface - Template Pattern."""

from typing import Any, Dict, Optional, Tuple


class NodeLearner:
    """
    Template to implement learning processes, including metric monitoring during training.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        self_addr: The address of the learner.
        epochs: The number of epochs of the model.

    """

    def __init__(self, model: Any, data: Any, self_addr: str, epochs: int) -> None:
        """Initialize the learner."""
        raise NotImplementedError

    def set_model(self, model: Any) -> None:
        """
        Set the model of the learner (not wheights).

        Args:
            model: The model of the learner.

        Raises:
            ModelNotMatchingError: If the model is not matching the learner.

        """
        raise NotImplementedError

    def set_data(self, data: Any) -> None:
        """
        Set the data of the learner. It is used to fit the model.

        Args:
            data: The data of the learner.

        """
        raise NotImplementedError

    def encode_parameters(self, params: Optional[Any] = None) -> bytes:
        """
        Encode the parameters of the model. (binary) If params are not provided, self parameters are encoded.

        Args:
            params: The parameters of the model. (non-binary)
            contributors: The contributors of the model.
            weight: The weight of the model.

        Returns:
            The encoded parameters of the model.

        """
        raise NotImplementedError

    def decode_parameters(self, data: bytes) -> Any:
        """
        Decode the parameters of the model (binary).

        Args:
            data: The encoded parameters of the model.

        Returns:
            The decoded parameters of the model.

        """
        raise NotImplementedError

    def set_parameters(self, params: Any) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model. (non-binary)

        Raises:
            ModelNotMatchingError: If the model is not matching the learner.

        """
        raise NotImplementedError

    def get_parameters(self) -> Any:
        """
        Get the parameters of the model.

        Returns
            The parameters of the model. (non-binary)

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

    def get_num_samples(self) -> Tuple[int, int]:
        """
        Get the number of samples of the model.

        Returns
            The number of samples of the model.

        """
        raise NotImplementedError
