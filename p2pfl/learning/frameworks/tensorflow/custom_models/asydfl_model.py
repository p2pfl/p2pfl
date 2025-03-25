""" Custom Keras Model with de-biased gradient updates. """
from typing import Optional

import tensorflow as tf

from p2pfl.management.logger import logger


class DeBiasedAsyDFLKerasModel(tf.keras.Model):
    """
    Custom Keras Model with de-biased gradient updates.

    Args:
        base_model: The base model.
        de_biased_model: The de-biased model.

    """
    def __init__(self, base_model: tf.keras.Model, de_biased_model: Optional[tf.keras.Model] = None, **kwargs):
        """Initialize the model."""
        super().__init__(**kwargs)
        self.base_model = base_model
        if de_biased_model is None:
            self.de_biased_model = tf.keras.models.clone_model(base_model)
        else:
            self.de_biased_model = de_biased_model
        self.last_training_loss = None

    @property
    def optimizer(self):
        """
        Get the optimizer of the model.

        Returns:
            The optimizer.

        """
        return self.base_model.optimizer

    @optimizer.setter
    def optimizer(self, value):
        """
        Set the optimizer of the model.

        Args:
            value: The optimizer.

        """
        self.base_model.optimizer = value

    def train_step(self, data):
        """
        Apply a de-biasing adjustment in a custom training step.

        Args:
            data: The training data.

        Returns:
            Dict: The training metrics.

        """
        x, y = data

        logger.info("test",f"{self.de_biased_model=}")

        with tf.GradientTape() as tape:
            predictions = self.de_biased_model(x, training=True)
            self.last_training_loss = self.compiled_loss(y, predictions)

        # Compute gradients
        gradients = tape.gradient(self.last_training_loss, self.de_biased_model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

        # Compute metrics
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        """
        Call the model with the given inputs.

        Args:
            inputs: The input data.
            training: Whether the model is training.

        Returns:
            The model output.

        """
        return self.base_model(inputs, training=training)

    def get_config(self):
        """
        Return the configuration of the model for saving.

        Returns:
            The configuration of the model.

        """
        config = super().get_config()
        config.update({
            "base_model": self.base_model,
            "de_biased_model": self.de_biased_model
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
        return cls(**config)
