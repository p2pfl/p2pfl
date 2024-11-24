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

"""Callback for SCAFFOLD operations (Keras)."""

from typing import Any, Dict, List, Optional

import tensorflow as tf  # type: ignore
from keras import callbacks  # type: ignore

from p2pfl.learning.frameworks.callback import P2PFLCallback


class SCAFFOLDCallback(callbacks.Callback, P2PFLCallback):
    """
    Callback for SCAFFOLD operations to use with TensorFlow Keras.

    At the beginning of the training, the callback stores the global model and the initial learning rate.
    After optimization steps, it applies control variate adjustments.

    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.c_i: List[tf.Variable] = []
        self.c: List[tf.Variable] = []
        self.initial_model_params: List[tf.Variable] = []
        self.saved_lr: Optional[float] = None
        self.K: int = 0
        self.additional_info: Dict[str, Any] = {}

    @staticmethod
    def get_name() -> str:
        """Return the name of the callback."""
        return "scaffold"

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Store the global model and the initial learning rate."""
        if not self.c_i:
            self.c_i = [tf.Variable(tf.zeros_like(param), trainable=False) for param in self.model.trainable_variables]

        if self.K == 0:
            if "global_c" not in self.additional_info:
                self.additional_info["global_c"] = None

            c = self.additional_info["global_c"]
            if c is None:
                self.c = [tf.Variable(tf.zeros_like(param), trainable=False) for param in self.model.trainable_variables]
            else:
                self.c = [tf.Variable(tf.convert_to_tensor(c_np), trainable=False) for c_np in c]

        self.initial_model_params = [tf.Variable(param.numpy()) for param in self.model.trainable_variables]
        self.K = 0

    def on_train_batch_begin(self, batch: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Store the learning rate.

        Args:
            batch: The batch.
            logs: The logs.

        """
        optimizer = self.model.optimizer
        if hasattr(optimizer, "learning_rate"):
            lr = optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                self.saved_lr = tf.keras.backend.get_value(lr(self.model.optimizer.iterations))
            else:
                self.saved_lr = tf.keras.backend.get_value(lr)
        else:
            raise AttributeError("The optimizer does not have a learning rate attribute.")

    def on_train_batch_end(self, batch: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Modify model by applying control variate adjustments after the optimizer step.

        Args:
            batch: The batch.
            logs: The logs.

        """
        if self.saved_lr is None:
            raise AttributeError("Learning rate has not been set.")

        eta_l = self.saved_lr
        for param, c_i_var, c_var in zip(self.model.trainable_variables, self.c_i, self.c):
            adjustment = eta_l * c_i_var - eta_l * c_var
            param.assign_add(adjustment)
        self.K += 1

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Restore the global model.

        Args:
            logs: The logs.

        """
        y_i = [param.numpy() for param in self.model.trainable_variables]
        x_g = self.initial_model_params

        if not x_g or self.saved_lr is None:
            raise AttributeError("Necessary attributes are not initialized.")

        previous_c_i = [c_i_var.numpy() for c_i_var in self.c_i]

        for idx, (_, x, y) in enumerate(zip(self.c_i, x_g, y_i)):
            adjustment = (x - y) / (self.K * self.saved_lr)
            self.c_i[idx].assign_add(adjustment)

        # Compute delta y_i and delta c_i
        delta_y_i = [y - x for y, x in zip(y_i, x_g)]
        delta_c_i = [c_new.numpy() - c_old for c_new, c_old in zip(self.c_i, previous_c_i)]

        self.additional_info["delta_y_i"] = delta_y_i
        self.additional_info["delta_c_i"] = delta_c_i
