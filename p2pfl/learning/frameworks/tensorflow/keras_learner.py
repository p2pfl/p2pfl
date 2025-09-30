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

"""Keras learner for P2PFL."""

import numpy as np
import tensorflow as tf  # type: ignore

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.tensorflow.callbacks.keras_logger import FederatedLogger
from p2pfl.learning.frameworks.tensorflow.keras_dataset import KerasExportStrategy
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import allow_no_addr_check
from p2pfl.utils.seed import set_seed


class KerasLearner(Learner):
    """
    Learner for TensorFlow/Keras models in P2PFL.

    Args:
        model: The KerasModel instance.
        data: The P2PFLDataset instance.
        addr: The address of this node.

    """

    def __init__(self, model: P2PFLModel | None = None, data: P2PFLDataset | None = None, aggregator: Aggregator | None = None) -> None:
        """Initialize the KerasLearner."""
        super().__init__(model, data, aggregator)

    @allow_no_addr_check
    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        super().set_model(model)
        self.get_model().model.compile(
            optimizer=self.get_model().model.optimizer,
            loss=self.get_model().model.loss,
            metrics=["sparse_categorical_accuracy"],
        )

    def set_addr(self, addr: str) -> str:
        """Set the addr of the node."""
        self.callbacks.append(FederatedLogger(addr))
        return super().set_addr(addr)

    def __get_tf_model_data(self, train: bool = True) -> tuple[tf.keras.Model, tf.data.Dataset]:
        # Get Model
        tf_model = self.get_model().get_model()
        if not isinstance(tf_model, tf.keras.Model):
            raise ValueError("The model must be a TensorFlow Keras model")
        # Get Data
        data = self.get_data().export(KerasExportStrategy, train=train)
        if not isinstance(data, tf.data.Dataset):
            raise ValueError("The data must be a TensorFlow Dataset")
        return tf_model, data

    def fit(self) -> P2PFLModel:
        """Fit the model."""
        set_seed(Settings.general.SEED, self.get_framework())
        try:
            if self.epochs > 0:
                model, data = self.__get_tf_model_data(train=True)
                model.fit(
                    data,
                    epochs=self.epochs,
                    callbacks=self.callbacks,  # type: ignore
                )

            # Set model contribution
            self.get_model().set_contribution([self.addr], self.get_data().get_num_samples(train=True))

            # Set callback info
            self.add_callback_info_to_model()

            return self.get_model()
        except Exception as e:
            logger.error(self.addr, f"Error in training with Keras: {e}")
            raise e

    def interrupt_fit(self) -> None:
        """Interrupt the training process."""
        # Keras doesn't have a direct way to interrupt fit.
        # Need to implement a custom callback or use a flag to stop training.
        logger.error(self.addr, "Interrupting training (not fully implemented for Keras).")

    def evaluate(self) -> dict[str, float]:
        """Evaluate the Keras model."""
        try:
            if self.epochs > 0:
                model, data = self.__get_tf_model_data(train=False)
                results = model.evaluate(data, verbose=0)
                if not isinstance(results, list):
                    results = [results]
                results_dict = dict(zip(model.metrics_names, results, strict=False))
                for k, v in results_dict.items():
                    logger.log_metric(self.addr, k, v)
                return results_dict
            else:
                return {}
        except Exception as e:
            logger.error(self.addr, f"Evaluation error with Keras: {e}")
            raise e

    def get_framework(self) -> str:
        """
        Retrieve the learner name.

        Returns:
            The name of the learner class.

        """
        return Framework.TENSORFLOW.value
