#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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

"""Simple MLP on Tensorflow Keras for MNIST."""

import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed

####
# Example MLP
####


@tf.keras.utils.register_keras_serializable("p2pfl")
class MLP(tf.keras.Model):
    """Multilayer Perceptron (MLP) for MNIST classification using Keras."""

    def __init__(self, hidden_sizes=None, out_channels=10, lr_rate=0.001, **kwargs):
        """
        Initialize the MLP.

        Args:
            hidden_sizes (list): List of integers representing the number of neurons in each hidden layer.
            out_channels (int): Number of output classes (10 for MNIST).
            lr_rate (float): Learning rate for the Adam optimizer.
            kwargs: Additional arguments.

        """
        set_seed(Settings.general.SEED, "tensorflow")
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        super().__init__()
        # Define layers
        self.flatten = Flatten()
        self.hidden_layers = [Dense(size, activation="relu") for size in hidden_sizes]
        self.output_layer = Dense(out_channels)

        # Define loss, optimizer, and metrics
        self.loss = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=lr_rate)

        # Force the model to be built
        self(tf.zeros((1, 28, 28, 1)))

    def call(self, inputs):
        """Forward pass of the MLP."""
        x = self.flatten(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


# Export P2PFL model
def model_build_fn(*args, **kwargs) -> KerasModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return KerasModel(MLP(*args, **kwargs), compression=compression)  # type: ignore
