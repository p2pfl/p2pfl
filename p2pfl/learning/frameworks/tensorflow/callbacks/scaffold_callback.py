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

from typing import Any

import tensorflow as tf  # type: ignore
from keras import callbacks  # type: ignore
from tensorflow.keras.optimizers import Optimizer

from p2pfl.learning.frameworks.callback import P2PFLCallback


class ScaffoldOptimizerWrapper(Optimizer):
    """Wraps an optimizer to only redefine apply_gradients, delegating other calls."""

    def __init__(self, optimizer, c_i, c, eta_l):
        """
        Initialize the wrapper.

        Args:
            optimizer: The optimizer to wrap.
            c_i: The control variate for the current iteration.
            c: The global control variate.
            eta_l: The learning rate.

        """
        self._optimizer = optimizer  # Use a different name to avoid recursion in __getattr__
        self.c_i = c_i
        self.c = c
        self.eta_l = eta_l

    @tf.function
    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Apply gradients with SCAFFOLD adjustments."""
        adjustment = []
        for (grad, var), c_i_var, c_var in zip(grads_and_vars, self.c_i, self.c, strict=False):
            if grad is not None:
                adjusted_grad = grad + self.eta_l * (c_i_var - c_var)
                adjustment.append((adjusted_grad, var))
            else:
                adjustment.append((grad, var))
        self._optimizer.apply_gradients(adjustment, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attribute/method calls to the original optimizer."""
        return getattr(self._optimizer, name)


class SCAFFOLDCallback(callbacks.Callback, P2PFLCallback):
    """
    Callback for SCAFFOLD operations to use with TensorFlow Keras.

    At the beginning of the training, the callback initializes control variates and substitutes
    the optimizer with a custom one to apply control variate adjustments.
    After training, it updates the local control variate (c_i) and computes the deltas.

    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.c_i: list[tf.Variable] = []
        self.c: list[tf.Variable] = []
        self.initial_model_params: list[tf.Variable] = []
        self.saved_lr: float | None = None
        self.K: int = 0
        self.additional_info: dict[str, Any] = {}

    @staticmethod
    def get_name() -> str:
        """Return the name of the callback."""
        return "scaffold"

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Initialize control variates and replace the optimizer with custom one."""
        optimizer = self.model.optimizer
        if hasattr(optimizer, "learning_rate"):
            lr = optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                self.saved_lr = tf.keras.backend.get_value(lr(optimizer.iterations))
            else:
                self.saved_lr = tf.keras.backend.get_value(lr)
        else:
            raise AttributeError("The optimizer does not have a learning rate attribute.")

        if not self.c_i:
            self.c_i = [tf.Variable(tf.zeros_like(param), trainable=False) for param in self.model.trainable_variables]

        global_c = self.additional_info.get("global_c")
        if global_c is not None:
            self.c = [tf.Variable(tf.convert_to_tensor(c_np), trainable=False) for c_np in global_c]
        else:
            if not self.c:
                self.c = [tf.Variable(tf.zeros_like(param), trainable=False) for param in self.model.trainable_variables]

        self.initial_model_params = [tf.Variable(param.numpy()) for param in self.model.trainable_variables]
        self.K = 0

        self.model.optimizer = ScaffoldOptimizerWrapper(
            optimizer=optimizer,
            c_i=self.c_i,
            c=self.c,
            eta_l=self.saved_lr,
        )  # type: ignore

    def on_train_batch_end(self, batch: Any, logs: dict[str, Any] | None = None) -> None:
        """
        Increment the local step counter after each batch.

        Args:
            batch: The batch.
            logs: The logs.

        """
        self.K += 1

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """
        Update local control variate (c_i) and compute deltas.

        Args:
            logs: The logs.

        """
        y_i = [param.numpy() for param in self.model.trainable_variables]
        x_g = self.initial_model_params

        if not x_g or self.saved_lr is None:
            raise AttributeError("Necessary attributes are not initialized.")

        previous_c_i = [c_i_var.numpy() for c_i_var in self.c_i]

        for idx, c_i_var in enumerate(self.c_i):
            adjustment = (x_g[idx] - y_i[idx]) / (self.K * self.saved_lr)
            c_i_var.assign_add(adjustment)

        # Compute delta y_i and delta c_i
        delta_y_i = [y_i_param - x_g_param for y_i_param, x_g_param in zip(y_i, x_g, strict=False)]
        delta_c_i = [c_i_new.numpy() - c_i_old for c_i_new, c_i_old in zip(self.c_i, previous_c_i, strict=False)]

        self.additional_info["delta_y_i"] = delta_y_i
        self.additional_info["delta_c_i"] = delta_c_i

    def set_additional_info(self, info: dict[str, Any]) -> None:
        """Set additional information required for SCAFFOLD."""
        self.additional_info = info
