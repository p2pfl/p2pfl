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

from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.tensorflow.keras_dataset import KerasExportStrategy
from p2pfl.learning.tensorflow.keras_logger import FederatedLogger
from p2pfl.learning.tensorflow.keras_model import KerasModel
from p2pfl.management.logger import logger


class KerasLearner(NodeLearner):
    """
    Learner for TensorFlow/Keras models in P2PFL.

    Args:
        model: The KerasModel instance.
        data: The P2PFLDataset instance.
        self_addr: The address of this node.

    """

    def __init__(self, model: KerasModel, data: P2PFLDataset, self_addr: str = "unknown-node") -> None:
        """Initialize the KerasLearner."""
        self.model = model
        self.data = data
        self.__self_addr = self_addr
        self.epochs = 1  # Default epochs

        # Compile the model (you might need to customize this)
        self.model.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def set_model(self, model: Union[P2PFLModel, List[np.ndarray], bytes]) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        if isinstance(model, KerasModel):
            self.model = model
        elif isinstance(model, (list, bytes)):
            self.model.set_parameters(model)

    def get_model(self) -> KerasModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        return self.model

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner.

        Args:
            data: The data of the learner.

        """
        self.data = data

    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The data of the learner.

        """
        return self.data

    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs.

        Args:
            epochs: The number of epochs.

        """
        self.epochs = epochs

    def __get_tf_model_data(self, train: bool = True) -> Tuple[tf.keras.Model, tf.data.Dataset]:
        # Get Model
        tf_model = self.model.get_model()
        if not isinstance(tf_model, tf.keras.Model):
            raise ValueError("The model must be a TensorFlow Keras model")
        # Get Data
        data = self.data.export(KerasExportStrategy, train=train)
        if not isinstance(data, tf.data.Dataset):
            raise ValueError("The data must be a TensorFlow Dataset")
        return tf_model, data

    def fit(self) -> None:
        """Fit the model."""
        try:
            if self.epochs > 0:
                model, data = self.__get_tf_model_data(train=True)
                model.fit(
                    data,
                    epochs=self.epochs,
                    callbacks=[FederatedLogger(self.__self_addr)],
                )
            # Set model contribution
            self.model.set_contribution([self.__self_addr], self.data.get_num_samples(train=True))
        except Exception as e:
            logger.error(self.__self_addr, f"Error in training with Keras: {e}")

    def interrupt_fit(self) -> None:
        """Interrupt the training process."""
        # Keras doesn't have a direct way to interrupt fit.
        # Need to implement a custom callback or use a flag to stop training.
        logger.error(self.__self_addr, "Interrupting training (not fully implemented for Keras).")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the Keras model."""
        try:
            if self.epochs > 0:
                model, data = self.__get_tf_model_data(train=False)
                results = model.evaluate(data, verbose=0)
                if not isinstance(results, list):
                    results = [results]
                results_dict = dict(zip(model.metrics_names, results))
                for k, v in results_dict.items():
                    logger.log_metric(self.__self_addr, k, v)
                return results_dict
            else:
                return {}
        except Exception as e:
            logger.error(self.__self_addr, f"Evaluation error with Keras: {e}")
            raise e
