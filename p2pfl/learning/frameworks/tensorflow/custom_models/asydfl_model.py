""" Custom Keras Model with de-biased gradient updates. """
import importlib
from typing import Optional

import numpy as np
import tensorflow as tf

from p2pfl.management.logger import logger


@tf.keras.utils.register_keras_serializable(package="p2pfl")
class DeBiasedAsyDFLKerasModel(tf.keras.Model):
    """
    Custom Keras Model with de-biased gradient updates.

    Args:
        base_model: The base model.
        de_biased_model: The de-biased model.

    """
    def __init__(self, model: tf.keras.Model, push_sum_weight: float = 1.0, last_training_loss: Optional[float] = None, **kwargs):
        """Initialize the model."""
        super().__init__(**kwargs)
        self.model = model
        self.push_sum_weight = push_sum_weight
        self.last_training_loss = last_training_loss

        self.loss = self.model.loss

    @property
    def optimizer(self):
        """
        Get the optimizer of the model.

        Returns:
            The optimizer.

        """
        return self.model.optimizer

    @optimizer.setter
    def optimizer(self, value):
        """
        Set the optimizer of the model.

        Args:
            value: The optimizer.

        """
        self.model.optimizer = value

    def train_step(self, data):
        """
        Apply a de-biasing adjustment in a custom training step.

        Args:
            data: The training data.

        Returns:
            Dict: The training metrics.

        """
        # Unpack the data
        x, y = data

        # Scale trainable variables by μ_t before the forward pass
        for var in self.model.trainable_variables:
            # Apply scaling before computing loss
            var.assign(var / self.push_sum_weight)  # Scale ω_t by μ_t

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass
            # Compute the loss value
            loss = self.compute_loss(y=y, y_pred=y_pred, training=True)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Restore trainable variables by multiplying them back by μ_t
        for var in self.model.trainable_variables:
            var.assign(var * self.push_sum_weight)  # Restore ω_t by multiplying with μ_t

        # Update weights using the computed gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False) -> tf.Tensor:
        """
        Call the model with the given inputs.

        Args:
            inputs: The input data.
            training: Whether the model is training.

        Returns:
            The model output.

        """
        return self.model(inputs, training=training)

    def get_config(self) -> dict:
        """
        Return the configuration of the model for saving.

        Returns:
            The configuration of the model.

        """
        config = super().get_config()
        config.update({
            "model": tf.keras.utils.serialize_keras_object(self.model),
            "push_sum_weight": self.push_sum_weight,
            "last_training_loss": self.last_training_loss
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Create an instance from the configuration dictionary.

        Args:
            config: The configuration dictionary.

        Returns:
            The model instance.

        """
        def load_model_from_config(model_config):
            """Dynamically load a model from its config."""
            module_name = model_config["module"]
            class_name = model_config["class_name"]

            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the model class
            model_class = getattr(module, class_name)

            # Deserialize model instance
            return model_class.from_config(model_config["config"])

        # Load base_model and de_biased_model dynamically
        config["model"] = load_model_from_config(config["model"])
        return cls(**config)

    def get_weights(self) -> list[np.ndarray]:
        """ Get the weights of the model. """
        return self.model.get_weights()

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """
        Set the weights of the model.

        Args:
            weights: The weights.

        """
        self.model.set_weights(weights)
