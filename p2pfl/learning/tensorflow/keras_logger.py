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

"""Keras Logger for P2PFL."""

import tensorflow as tf

from p2pfl.management.logger import Logger as P2PLogger


class FederatedLogger(tf.keras.callbacks.Callback):
    """
    Keras Logger for Federated Learning. Handles local training logging.

    Args:
        node_name: Name of the node.

    """

    def __init__(self, node_name: str) -> None:
        """Initialize the callback."""
        super().__init__()
        self.self_name = node_name
        self.step = 0  # Initialize training step counter

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        return

        if logs is not None:
            for k, v in logs.items():
                P2PLogger.log_metric(self.self_name, k, v, self.step)

    def on_train_batch_end(self, batch, logs=None):
        """Log metrics at the end of each batch (optional)."""
        self.step += 1  # Increment step counter after each batch
        if logs is not None:
            for k, v in logs.items():
                P2PLogger.log_metric(self.self_name, k, v, self.step)
