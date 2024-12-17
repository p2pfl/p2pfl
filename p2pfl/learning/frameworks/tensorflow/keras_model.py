#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Keras model abstraction for P2PFL."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

#####################
#    KerasModel     #
#####################


class ModelNotBuiltError(Exception):
    """Raised when a model is not built."""

    pass


class KerasModel(P2PFLModel):
    """
    P2PFL model abstraction for TensorFlow/Keras.

    Args:
        model: The Keras model to encapsulate.
        params: Optional initial parameters (list of NumPy arrays or bytes).
        num_samples: Optional number of samples used for training.
        contributors: Optional list of contributor nodes.
        additional_info: Optional dictionary for extra information.

    """

    def __init__(
        self,
        model: tf.keras.Model,
        params: Optional[Union[List[np.ndarray], bytes]] = None,
        num_samples: Optional[int] = None,
        contributors: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the KerasModel."""
        super().__init__(model, params, num_samples, contributors, additional_info)

        # Ensure the model is built
        if len(model.get_weights()) == 0:  # type: ignore
            raise ModelNotBuiltError(
                "Model must be built before creating a P2PFLMODEL! Please be sure that model.get_weights() return a non empty list."
            )

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        return self.model.get_weights()

    def set_parameters(self, params: Union[List[np.ndarray], bytes]) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        if isinstance(params, bytes):
            params, additional_info = self.decode_parameters(params)
            self.additional_info.update(additional_info)

        # Set weights layer by layer
        try:
            self.model.set_weights(params)
        except ValueError as e:
            raise ModelNotMatchingError("Parameters don't match the model. Please check the model architecture and the parameters.") from e

    def get_framework(self) -> str:
        """
        Retrieve the model framework name.

        Returns:
            The name of the model framework.

        """
        return Framework.TENSORFLOW.value


####
# Example MLP
####


class MLP(tf.keras.Model):
    """Multilayer Perceptron (MLP) for MNIST classification using Keras."""

    def __init__(self, hidden_sizes=None, out_channels=10, lr_rate=0.001, seed=None, **kwargs):
        """
        Initialize the MLP.

        Args:
            hidden_sizes (list): List of integers representing the number of neurons in each hidden layer.
            out_channels (int): Number of output classes (10 for MNIST).
            lr_rate (float): Learning rate for the Adam optimizer.
            seed (int, optional): Random seed for reproducibility.
            kwargs: Additional arguments.

        """
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        super().__init__()

        if seed is not None:
            tf.random.set_seed(seed)

        # Define layers
        self.flatten = Flatten()
        self.hidden_layers = [Dense(size, activation="relu") for size in hidden_sizes]
        self.output_layer = Dense(out_channels)

        # Define loss, optimizer, and metrics
        self.loss = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=lr_rate)

    def call(self, inputs):
        """Forward pass of the MLP."""
        x = self.flatten(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def get_framework(self) -> str:
        """
        Retrieve the model name.

        Returns:
            The name of the model class.

        """
        return Framework.TENSORFLOW.value
