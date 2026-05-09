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

import asyncio

import numpy as np
import tensorflow as tf  # type: ignore

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.tensorflow.callbacks.keras_logger import FederatedLogger
from p2pfl.learning.frameworks.tensorflow.keras_dataset import KerasExportStrategy
from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel
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
        aggregator: The aggregator instance.

    """

    # Serializes TF operations across all instances in the same event loop.
    # TensorFlow deadlocks when multiple models run concurrently via asyncio.to_thread.
    _tf_lock = asyncio.Lock()

    def __init__(
        self,
        model: KerasModel | None = None,
        data: P2PFLDataset | None = None,
        aggregator: Aggregator | None = None,
    ) -> None:
        """Initialize the KerasLearner."""
        super().__init__(model, data, aggregator)
        self._batch_idx: int = 0
        self._np_cache: dict[bool, tuple[np.ndarray, np.ndarray]] = {}

    @allow_no_addr_check
    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        super().set_model(model)
        self.get_model().get_model().compile(
            optimizer=self.get_model().get_model().optimizer,
            loss=self.get_model().get_model().loss,
            metrics=["sparse_categorical_accuracy"],
        )

    def set_address(self, address: str) -> str:
        """Set the address of the node."""
        self.callbacks.append(FederatedLogger(address))
        return super().set_address(address)

    def __get_tf_model(self) -> tf.keras.Model:
        tf_model = self.get_model().get_model()
        if not isinstance(tf_model, tf.keras.Model):
            raise ValueError("The model must be a TensorFlow Keras model")
        return tf_model

    def __get_numpy_data(self, train: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if train not in self._np_cache:
            data = self.get_data().export(KerasExportStrategy, train=train)
            if not isinstance(data, tuple) or len(data) != 2:
                raise ValueError("KerasExportStrategy must return (x, y) numpy tuple")
            self._np_cache[train] = data
        return self._np_cache[train]

    def __gradient_step(self, model: tf.keras.Model, bx: np.ndarray, by: np.ndarray) -> float:
        """One gradient-tape training step.  Returns the scalar loss."""
        x_t = tf.constant(bx)
        y_t = tf.cast(tf.constant(by), tf.int64)
        with tf.GradientTape() as tape:
            preds = model(x_t, training=True)
            loss = model.compute_loss(x=x_t, y=y_t, y_pred=preds)
            assert loss is not None
        grads = tape.gradient(loss, model.trainable_variables)
        assert model.optimizer is not None
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=False))
        return float(loss.numpy())

    def __evaluate_numpy(self, model: tf.keras.Model, x: np.ndarray, y: np.ndarray, batch_size: int) -> dict[str, float]:
        """Evaluate model using direct forward pass (avoids Keras execution engine)."""
        total_loss = 0.0
        total_correct = 0
        n = len(x)
        for start in range(0, n, batch_size):
            bx = tf.constant(x[start : start + batch_size])
            by = tf.cast(tf.constant(y[start : start + batch_size]), tf.int64)
            preds = model(bx, training=False)
            batch_loss = model.compute_loss(x=bx, y=by, y_pred=preds)
            assert batch_loss is not None
            total_loss += float(batch_loss.numpy()) * len(bx)
            total_correct += int(tf.reduce_sum(tf.cast(tf.argmax(preds, axis=-1) == by, tf.int32)).numpy())
        return {"loss": total_loss / n, "sparse_categorical_accuracy": total_correct / n}

    async def fit(self) -> KerasModel:
        """Fit the model."""
        set_seed(Settings.general.SEED, self.get_framework())
        try:
            if self.epochs > 0:
                model = self.__get_tf_model()
                x, y = self.__get_numpy_data(train=True)
                batch_size = self.get_data().batch_size

                def _train_epochs():
                    last_loss = 0.0
                    for _ in range(self.epochs):
                        for start in range(0, len(x), batch_size):
                            last_loss = self.__gradient_step(model, x[start : start + batch_size], y[start : start + batch_size])
                    return last_loss

                async with KerasLearner._tf_lock:
                    last_loss = await asyncio.to_thread(_train_epochs)
                self.get_model().last_training_loss = last_loss

            self.get_model().set_contribution([self.address], self.get_data().get_num_samples(train=True))
            self.add_callback_info_to_model()
            return self.get_model()
        except Exception as e:
            logger.error(self.address, f"Error in training with Keras: {e}")
            raise e

    async def train_on_batch(self):
        """Train the model on the next batch manually."""
        set_seed(Settings.general.SEED, self.get_framework())
        try:
            x, y = self.__get_numpy_data(train=True)
            batch_size = self.get_data().batch_size
            start = self._batch_idx * batch_size
            if start >= len(x):
                self._batch_idx = 0
                start = 0
            end = min(start + batch_size, len(x))
            bx, by = x[start:end], y[start:end]
            self._batch_idx += 1

            model = self.__get_tf_model()
            async with KerasLearner._tf_lock:
                loss = await asyncio.to_thread(self.__gradient_step, model, bx, by)

            self.get_model().last_training_loss = loss
            self.get_model().set_contribution([self.address], self.get_data().get_num_samples(train=True))
            self.add_callback_info_to_model()
            return self.get_model()
        except Exception as e:
            logger.error(self.address, f"Error in training with Keras: {e}")
            raise e

    async def interrupt_fit(self) -> None:
        """Interrupt the training process."""
        logger.error(self.address, "Interrupting training (not fully implemented for Keras).")

    async def evaluate(self) -> dict[str, float]:
        """Evaluate the Keras model."""
        try:
            if self.epochs > 0:
                model = self.__get_tf_model()
                x, y = self.__get_numpy_data(train=False)
                batch_size = self.get_data().batch_size
                async with KerasLearner._tf_lock:
                    results_dict = await asyncio.to_thread(self.__evaluate_numpy, model, x, y, batch_size)
                for k, v in results_dict.items():
                    logger.log_metric(self.address, k, v)
                return results_dict
            else:
                return {}
        except Exception as e:
            logger.error(self.address, f"Evaluation error with Keras: {e}")
            raise e

    def get_framework(self) -> str:
        """
        Retrieve the learner name.

        Returns:
            The name of the learner class.

        """
        return Framework.TENSORFLOW.value
