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

import tensorflow as tf
from keras import callbacks

from p2pfl.learning.callbacks.decorators import register


@register(callback_key='scaffold', framework='keras')
class SCAFFOLDCallback(callbacks.Callback):
    """
    Callback for SCAFFOLD operations to use with TensorFlow Keras.

    At the beginning of the training, the callback stores the global model and the initial learning rate.
    After optimization steps, it applies control variate adjustments.

    """

    def __init__(self):
        """Initialize the callback."""
        super().__init__()
        self.c_i: Optional[List[tf.Variable]] = None
        self.c: Optional[List[tf.Variable]] = None
        self.initial_model_params: Optional[List[tf.Variable]] = None
        self.saved_lr: Optional[float] = None
        self.K: int = 0
        self.additional_info: Dict[str, Any] = {}

    def on_train_begin(self, logs=None):
        """Store the global model and the initial learning rate."""
        if self.c_i is None:
            self.c_i = [tf.Variable(tf.zeros_like(param), trainable=False)
                       for param in self.model.trainable_variables]

        if self.K == 0:
            if 'global_c' not in self.additional_info:
                self.additional_info['global_c'] = None

            c = self.additional_info['global_c']
            if c is None:
                self.c = [tf.Variable(tf.zeros_like(param), trainable=False)
                          for param in self.model.trainable_variables]
            else:
                self.c = [
                    tf.Variable(tf.convert_to_tensor(c_np), trainable=False)
                    for c_np in c
                ]

        self.initial_model_params = [tf.Variable(param.numpy()) for param in self.model.trainable_variables]
        self.K = 0

    def on_train_batch_begin(self, batch, logs=None):
        """
        Store the learning rate.

        Args:
            batch: The batch.
            logs: The logs.

        """
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'learning_rate'):
            lr = optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                self.saved_lr = tf.keras.backend.get_value(lr(self.model.optimizer.iterations))
            else:
                self.saved_lr = tf.keras.backend.get_value(lr)
        else:
            raise AttributeError("The optimizer does not have a learning rate attribute.")


    def on_train_batch_end(self, batch, logs=None):
        """
        Modify model by applying control variate adjustments after the optimizer step.

        Args:
            batch: The batch.
            logs: The logs.

        """
        eta_l = self.saved_lr
        # y_i ← y_i + eta_l * c_i - eta_l * c
        for param, c_i, c in zip(self.model.trainable_variables, self.c_i, self.c):
            adjustment = eta_l * c_i - eta_l * c
            param.assign_add(adjustment)
        self.K += 1

    def on_train_end(self, logs=None):
        """
        Restore the global model.

        Args:
            logs: The logs.

        """
        y_i = [param.numpy() for param in self.model.trainable_variables]
        x_g = self.initial_model_params
        previous_c_i = [c_i.numpy() for c_i in self.c_i]

        for idx, (_, x, y) in enumerate(zip(self.c_i, x_g, y_i)):
            adjustment = (x - y) / (self.K * self.saved_lr)
            self.c_i[idx].assign_add(adjustment)

        # Compute delta y_i and delta c_i
        delta_y_i = [y - x for y, x in zip(y_i, x_g)]
        delta_c_i = [c_new.numpy() - c_old for c_new, c_old in zip(self.c_i, previous_c_i)]

        self.additional_info['delta_y_i'] = delta_y_i
        self.additional_info['delta_c_i'] = delta_c_i

    def set_global_c(self, global_c_np: Optional[List[tf.Tensor]]):
        """
        Set the global control variate from an aggregator.

        Args:
            global_c_np: The global control variate.

        """
        self.additional_info['global_c'] = global_c_np

