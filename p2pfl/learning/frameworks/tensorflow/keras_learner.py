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

"""Keras learners for P2PFL."""

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

    Uses native Keras APIs (model.fit, model.evaluate) for training and evaluation.

    Args:
        model: The KerasModel instance.
        data: The P2PFLDataset instance.
        aggregator: The aggregator instance.

    """

    def __init__(
        self,
        model: KerasModel | None = None,
        data: P2PFLDataset | None = None,
        aggregator: Aggregator | None = None,
    ) -> None:
        """Initialize the KerasLearner."""
        super().__init__(model, data, aggregator)
        self._batch_iterator = None

    @allow_no_addr_check
    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        self._train_x = None
        self._train_y = None
        self._train_batch_size = None
        self._train_idx = 0
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

    def _get_tf_model(self) -> tf.keras.Model:
        tf_model = self.get_model().get_model()
        if not isinstance(tf_model, tf.keras.Model):
            raise ValueError("The model must be a TensorFlow Keras model")
        return tf_model

    def _get_tf_data(self, train: bool = True) -> tuple:
        return self.get_data().export(KerasExportStrategy, train=train)

    async def fit(self) -> KerasModel:
        """Fit the model."""
        set_seed(Settings.general.SEED, self.get_framework())
        try:
            if self.epochs > 0:
                model = self._get_tf_model()
                x, y, batch_size = self._get_tf_data(train=True)

                history = model.fit(
                    x,
                    y,
                    epochs=self.epochs,
                    batch_size=batch_size,
                    callbacks=self.callbacks,  # type: ignore
                    steps_per_epoch=self.steps_per_epoch,
                    verbose=0,
                )
                self.get_model().last_training_loss = history.history["loss"][-1]

            # Set model contribution
            self.get_model().set_contribution([self.address], self.get_data().get_num_samples(train=True))

            # Set callback info
            self.add_callback_info_to_model()

            return self.get_model()
        except Exception as e:
            logger.error(self.address, f"Error in training with Keras: {e}")
            raise e

    async def train_on_batch(self):
        """Train the model on the next batch manually."""
        set_seed(Settings.general.SEED, self.get_framework())
        if not hasattr(self, "_train_x") or self._train_x is None:
            x, y, batch_size = self._get_tf_data(train=True)
            self._train_x, self._train_y = x, y
            self._train_batch_size = batch_size
            self._train_idx = 0

        try:
            model = self._get_tf_model()
            bs = self._train_batch_size
            start = self._train_idx
            end = start + bs

            if start >= len(self._train_x):
                self._train_idx = 0
                start, end = 0, bs

            batch_x = self._train_x[start:end]
            batch_y = self._train_y[start:end]
            self._train_idx = end

            loss = model.train_on_batch(batch_x, batch_y)
            self.get_model().last_training_loss = float(loss[0]) if isinstance(loss, list | tuple) else float(loss)

            # Set model contribution
            self.get_model().set_contribution([self.address], self.get_data().get_num_samples(train=True))

            # Set callback info
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
                model = self._get_tf_model()
                x, y, batch_size = self._get_tf_data(train=False)

                results = model.evaluate(x, y, batch_size=batch_size, verbose=0)
                if not isinstance(results, list):
                    results = [results]
                results_dict = dict(zip(model.metrics_names, results, strict=False))
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


class EagerKerasLearner(KerasLearner):
    """
    Keras learner using eager-mode GradientTape.

    Avoids Keras internal tf.data thread pool deadlocks that occur
    on macOS when running inside Ray workers. Uses model.metrics for
    arbitrary metric support in evaluation.
    """

    async def fit(self) -> KerasModel:
        """Fit the model using GradientTape."""
        set_seed(Settings.general.SEED, self.get_framework())
        try:
            if self.epochs > 0:
                model = self._get_tf_model()
                x, y, batch_size = self._get_tf_data(train=True)

                loss_fn = model.loss
                if isinstance(loss_fn, str):
                    loss_fn = tf.keras.losses.get(loss_fn)
                optimizer = model.optimizer

                n = len(x)
                last_loss = 0.0
                for _ in range(self.epochs):
                    for step, start in enumerate(range(0, n, batch_size)):
                        if self.steps_per_epoch is not None and step >= self.steps_per_epoch:
                            break
                        xb = tf.constant(x[start : start + batch_size])
                        yb = tf.constant(y[start : start + batch_size])
                        with tf.GradientTape() as tape:
                            preds = model(xb, training=True)
                            loss = loss_fn(yb, preds)
                        grads = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=False))
                        last_loss = float(loss)
                self.get_model().last_training_loss = last_loss

            self.get_model().set_contribution([self.address], self.get_data().get_num_samples(train=True))
            self.add_callback_info_to_model()

            return self.get_model()
        except Exception as e:
            logger.error(self.address, f"Error in training with Keras: {e}")
            raise e

    async def train_on_batch(self):
        """Train on a single batch using GradientTape."""
        set_seed(Settings.general.SEED, self.get_framework())
        if not hasattr(self, "_train_x") or self._train_x is None:
            x, y, batch_size = self._get_tf_data(train=True)
            self._train_x, self._train_y = x, y
            self._train_batch_size = batch_size
            self._train_idx = 0

        try:
            model = self._get_tf_model()
            bs = self._train_batch_size
            start = self._train_idx
            end = start + bs

            if start >= len(self._train_x):
                self._train_idx = 0
                start, end = 0, bs

            xb = tf.constant(self._train_x[start:end])
            yb = tf.constant(self._train_y[start:end])
            self._train_idx = end

            loss_fn = model.loss
            if isinstance(loss_fn, str):
                loss_fn = tf.keras.losses.get(loss_fn)
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                loss = loss_fn(yb, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=False))
            self.get_model().last_training_loss = float(loss)

            self.get_model().set_contribution([self.address], self.get_data().get_num_samples(train=True))
            self.add_callback_info_to_model()

            return self.get_model()
        except Exception as e:
            logger.error(self.address, f"Error in training with Keras: {e}")
            raise e

    async def evaluate(self) -> dict[str, float]:
        """Evaluate the model using forward pass and model.metrics for arbitrary metric support."""
        try:
            if self.epochs > 0:
                model = self._get_tf_model()
                x, y, batch_size = self._get_tf_data(train=False)

                loss_fn = model.loss
                if isinstance(loss_fn, str):
                    loss_fn = tf.keras.losses.get(loss_fn)

                # Create fresh metric instances from the model's compile config.
                # Access _user_metrics on the CompileMetrics object to get the
                # original metric names/strings, avoiding Keras internal wrappers.
                compile_metrics = getattr(model, "_compile_metrics", None)
                user_metrics = getattr(compile_metrics, "_user_metrics", []) if compile_metrics else []
                metrics: list[tf.keras.metrics.Metric] = [tf.keras.metrics.get(m) if isinstance(m, str) else m for m in user_metrics]

                total_loss = 0.0
                total_samples = 0
                n = len(x)
                for start in range(0, n, batch_size):
                    xb = tf.constant(x[start : start + batch_size])
                    yb = tf.constant(y[start : start + batch_size])
                    preds = model(xb, training=False)
                    yb_flat = tf.reshape(yb, [-1])
                    loss = loss_fn(yb_flat, preds)
                    batch_n = len(xb)
                    total_loss += float(loss) * batch_n
                    total_samples += batch_n
                    for metric in metrics:
                        metric.update_state(yb_flat, preds)

                if total_samples == 0:
                    return {}

                results_dict: dict[str, float] = {"loss": total_loss / total_samples}
                results_dict.update({m.name: float(m.result()) for m in metrics})

                for k, v in results_dict.items():
                    logger.log_metric(self.address, k, v)
                return results_dict
            else:
                return {}
        except Exception as e:
            logger.error(self.address, f"Evaluation error with Keras: {e}")
            raise e
