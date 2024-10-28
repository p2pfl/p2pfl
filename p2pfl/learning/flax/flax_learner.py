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

"""Flax Learner for P2PFL."""

from typing import Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from flax.training import train_state

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.flax.flax_dataset import FlaxExportStrategy
from p2pfl.learning.flax.flax_model import FlaxModel
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


class FlaxLearner(NodeLearner):
    """
    Learner for Flax models in P2PFL.

    Args:
        model: The FlaxModel instance.
        data: The P2PFLDataset instance.
        self_addr: The address of this node.

    """

    def __init__(self, model: FlaxModel, data: P2PFLDataset, self_addr: str = "unknown-node") -> None:
        """Initialize the FlaxLearner."""
        self.model = model
        self.data = data
        self.__self_addr = self_addr
        self.epochs = 1  # Default epochs

        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate=1e-3)
        self.state = train_state.TrainState.create(apply_fn=self.model.model.apply, params=self.model.model_params, tx=self.optimizer)  # type: ignore

    def set_model(self, model: Union[P2PFLModel, List[np.ndarray], bytes]) -> None:
        """Set the model of the learner."""
        if isinstance(model, FlaxModel):
            self.model = model
        elif isinstance(model, (list, bytes)):
            self.model.set_parameters(model)

    def get_model(self) -> P2PFLModel:
        """Get the model of the learner."""
        return self.model

    def set_data(self, data: P2PFLDataset) -> None:
        """Set the data of the learner."""
        self.data = data

    def get_data(self) -> P2PFLDataset:
        """Get the data of the learner."""
        return self.data

    def set_epochs(self, epochs: int) -> None:
        """Set the number of epochs."""
        self.epochs = epochs

    def __get_flax_model_data(self, train: bool = True) -> Tuple:
        # Get model
        flax_model = self.model.get_model()
        # Get data
        jnp_dataloader = self.data.export(FlaxExportStrategy, train=train)

        return flax_model, jnp_dataloader

    def train_step(self, state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
        """Perform a single training step."""

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits,
                labels=y.reshape(
                    1,
                ),
            ).mean()
            return loss

        grads = jax.grad(loss_fn)(state.params)  # Compute gradients
        new_state = state.apply_gradients(grads=grads)  # type: ignore
        return new_state

    def fit(self) -> None:
        """Fit the model."""
        try:
            if self.epochs > 0:
                # Get training and validation data
                model, dataloader = self.__get_flax_model_data(train=True)

                # Training loop
                for epoch in range(self.epochs):
                    # Training phase
                    for x, y in dataloader:
                        x = x.reshape(1, *x.shape)
                        y = jnp.array([y])
                        self.state = self.train_step(self.state, x, y)

                    # End of epoch: Log training progress
                    logger.log_metric(self.__self_addr, "epoch", epoch)
                    print(f"Epoch {epoch + 1}/{self.epochs} completed.")

                # Set model contribution
                self.model.set_contribution([self.__self_addr], self.data.get_num_samples(train=True))
        except Exception as e:
            logger.error(self.__self_addr, f"Error in training with Flax: {e}")
            raise e

    def evaluate_step(self, state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
        """Evaluate the model on a batch of data."""
        logits = state.apply_fn({"params": state.params}, x)
        predictions = jnp.argmax(logits, axis=-1)
        print(f"predictions: {predictions}, true: {y}")
        accuracy = jnp.mean(
            predictions
            == y.reshape(
                1,
            )
        )
        return accuracy

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the Flax model."""
        try:
            if self.epochs > 0:
                model, dataloader = self.__get_flax_model_data(train=False)

                accuracies = []
                for x, y in dataloader:
                    accuracy = self.evaluate_step(self.state, x, y)  # Perform evaluation step
                    accuracies.append(accuracy)

                avg_accuracy = float(jnp.mean(jnp.array(accuracies)))
                logger.log_metric(self.__self_addr, "accuracy", avg_accuracy)
                return {"accuracy": avg_accuracy}
            else:
                return {}
        except Exception as e:
            logger.error(self.__self_addr, f"Evaluation error with Flax: {e}")
            raise e
