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

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax  # type: ignore
import tqdm
from flax.core import FrozenDict
from flax.training import train_state

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.flax.flax_dataset import FlaxExportStrategy
from p2pfl.learning.frameworks.flax.flax_model import FlaxModel
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


class FlaxLearner(Learner):
    """
    Learner for Flax models in P2PFL.

    Args:
        model: The FlaxModel instance.
        data: The P2PFLDataset instance.
        self_addr: The address of this node.

    """

    def __init__(
        self,
        model: P2PFLModel | None = None,
        data: P2PFLDataset | None = None,
        aggregator: Aggregator | None = None,
    ) -> None:
        """Initialize the FlaxLearner."""
        super().__init__(model, data, aggregator)
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate=1e-3)
        self.state = train_state.TrainState.create(
            apply_fn=self.flax_model.model.apply, params={"params": self.flax_model.model_params}, tx=self.optimizer
        )  # type: ignore
        raise NotImplementedError("FlaxLearner is not yet implemented. HAY COSAS QUE ESTAN MAL")

    @property
    def flax_model(self) -> FlaxModel:
        """Retrieve the Flax model."""
        return cast(FlaxModel, self.get_model())

    def __get_flax_data(self, train: bool = True) -> tuple:
        return self.get_data().export(FlaxExportStrategy, train=train)

    @staticmethod
    def __calculate_loss_acc(state: train_state.TrainState, params: dict, x: jnp.ndarray, y: jnp.ndarray):
        logits = state.apply_fn(params, x)
        predictions = jnp.argmax(logits, axis=-1)
        # Calculate the loss and accuracy
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=y.reshape(
                1,
            ),
        ).mean()
        acc = jnp.mean(
            predictions
            == y.reshape(
                1,
            )
        )
        return loss, acc

    def train_step(self, state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray) -> tuple[train_state.TrainState, float, float]:
        """Perform a single training step."""

        # TODO: test jit
        def compute_grads(params: FrozenDict[str, Any]) -> tuple[train_state.TrainState, float, float]:
            grad_fn = jax.value_and_grad(self.__calculate_loss_acc, argnums=1, has_aux=True)
            (loss, acc), grads = grad_fn(state, params, x, y)
            new_state = state.apply_gradients(grads=grads)  # type: ignore
            return new_state, loss, acc

        nwe_state, loss, acc = compute_grads(state.params)
        return nwe_state, loss, acc

    def fit(self) -> P2PFLModel:
        """Fit the model."""
        try:
            if self.epochs > 0:
                # Get training data
                dataloader = self.__get_flax_data(train=True)
                total_loss = 0.0
                total_acc = 0.0
                num_batches = 0

                # Training loop
                for epoch in tqdm.tqdm(range(self.epochs), total=self.epochs, desc="Training"):
                    # Training phase
                    for x, y in dataloader:
                        self.state, loss, acc = self.train_step(self.state, x, y)
                        total_loss += loss
                        total_acc += acc
                        num_batches += 1

                    avg_loss = total_loss / num_batches
                    avg_acc = total_acc / num_batches
                    # End of epoch: Log training progress
                    logger.log_metric(self.addr, "epoch", epoch)
                    logger.log_metric(self.addr, "loss", avg_loss)
                    logger.log_metric(self.addr, "accuracy", avg_acc)

            # Set model contribution
            self.flax_model.set_contribution([self.addr], self.get_data().get_num_samples(train=True))

            return self.flax_model

        except Exception as e:
            logger.error(self.addr, f"Error in training with Flax: {e}")
            raise e

    def evaluate(self) -> dict[str, float]:
        """Evaluate the Flax model."""

        # TODO: test jit
        def eval_step(state, x, y) -> float:
            _, acc = self.__calculate_loss_acc(state, state.params, x, y)
            return acc

        try:
            if self.epochs > 0:
                dataloader = self.__get_flax_data(train=False)

                accuracies = []
                for x, y in dataloader:
                    accuracy = eval_step(self.state, x, y)  # Perform evaluation step
                    accuracies.append(accuracy)

                avg_accuracy = float(jnp.mean(jnp.array(accuracies)))
                logger.log_metric(self.addr, "accuracy", avg_accuracy)
                return {"accuracy": avg_accuracy}
            else:
                return {}
        except Exception as e:
            logger.error(self.addr, f"Evaluation error with Flax: {e}")
            raise e

    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        # Flax doesn't have a direct way to interrupt fit.
        # Need to implement a custom callback or use a flag to stop training.
        logger.error(self.addr, "Interrupting training (not fully implemented for Flax).")

    def get_framework(self) -> str:
        """
        Retrieve the learner name.

        Returns:
            The name of the learner class.

        """
        return Framework.FLAX.value
