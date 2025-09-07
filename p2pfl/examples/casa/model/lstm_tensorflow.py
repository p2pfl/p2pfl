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

"""LSTM model on TensorFlow Keras for CASA."""

from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed

####
# LSTM Model for CASA
####


def create_lstm_model(
    lstm_units: int = 100,
    hidden_sizes: list[int] | None = None,
    out_channels: int = 10,
    input_shape: tuple[int, int] = (1, 36),
    lr_rate: float = 0.001,
    trainable_layers: list[int] | None = None,
    **kwargs,
):
    """
    Create a Sequential LSTM model for CASA time-series classification.

    Args:
        lstm_units (int): Number of LSTM units.
        hidden_sizes (list): List of integers representing the number of neurons in each dense layer.
        out_channels (int): Number of output classes (10 for CASA).
        input_shape (tuple): Input shape for the LSTM layer (time_steps, features).
        lr_rate (float): Learning rate for the Adam optimizer.
        trainable_layers (list): Optional list of layer indices to make trainable (for partial training).
        kwargs: Additional arguments.

    Returns:
        Sequential: Compiled Keras Sequential model.

    """
    set_seed(Settings.general.SEED, "tensorflow")

    if hidden_sizes is None:
        hidden_sizes = [72, 50, 36, 28]

    # Create Sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(lstm_units, input_shape=input_shape))

    # Add Dense layers
    for size in hidden_sizes:
        model.add(Dense(size, activation="relu"))

    # Add output layer
    model.add(Dense(out_channels, activation="softmax"))

    # Configure trainable layers if specified
    if trainable_layers is not None:
        for idx, layer in enumerate(model.layers):
            layer.trainable = idx in trainable_layers

    # Compile the model
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=lr_rate), metrics=["accuracy"])

    return model


# Export P2PFL model
def model_build_fn(*args, **kwargs) -> KerasModel:
    """
    Export the model build function for P2PFL.

    Args:
        *args: Positional arguments for create_lstm_model.
        **kwargs: Keyword arguments for create_lstm_model and compression settings.

    Returns:
        KerasModel: Wrapped LSTM model for P2PFL.

    """
    compression = kwargs.pop("compression", None)
    model = create_lstm_model(*args, **kwargs)
    return KerasModel(model, compression=compression)
