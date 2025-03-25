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

"""AsyDFL callback for P2PFL."""

from typing import Optional

import tensorflow as tf  # type: ignore

from p2pfl.learning.frameworks.learner import P2PFLCallback
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger as P2PLogger


class EdgeNodeUpdateCallback(tf.keras.callbacks.Callback, P2PFLCallback):
    """
    Custom callback to update the local model based on a de-biased model's gradient.

    This assumes that the gradient of the de-biased model is available during training.

    Args:
        model: The model of the learner.
        learner: The Learner instance associated with the callback.

    """

    def __init__(self, learner, de_biased_model: Optional[P2PFLModel] = None):
        """
        Initialize the callback.

        Args:
            learner: The Learner instance associated with the callback.
            de_biased_model: The de-biased model to use for gradient computation.

        """
        super().__init__()
        self.learner = learner  # Reference to the learner
        self.de_biased_model = de_biased_model  # Optional de-biased model
        self.model = learner.get_model()

    def on_train_batch_end(self, batch, logs=None):
        """Calculate gradients from the de-biased model at the end of each batch during training."""
        if self.de_biased_model:
            # Get the gradients from the de-biased model
            gradients = self._calculate_gradients(self.de_biased_model)
            # Update the local model parameters based on these gradients
            self._update_local_model(gradients)

    def on_epoch_end(self, epoch, logs=None):
        """Execute actions at the end of each epoch."""
        self.learner.update_callbacks_with_model_info()

    def _calculate_gradients(self, de_biased_model: P2PFLModel):
        """
        Calculate the gradients based on the de-biased model.

        You may want to compute the gradient wrt to a loss function here.

        """
        # Compute gradients
        with tf.GradientTape() as tape:
            predictions = self.model(self.learner.get_data().inputs)
            loss = self.model.loss(predictions, self.learner.get_data().labels)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients

    def _update_local_model(self, gradients):
        """
        Update the learner's local model using the computed gradients.

        Here, you can apply the gradients to update the model.

        """
        optimizer = tf.keras.optimizers.Adam()
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
