#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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
"""Virtual Node Learner."""

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.simulation.actor_pool import SuperActorPool
from p2pfl.management.logger import logger


class VirtualNodeLearner(Learner):
    """Decorator for the learner to be used in the simulation."""

    def __init__(self, learner: Learner) -> None:
        """Initialize the learner."""
        self.learner = learner
        self.actor_pool = SuperActorPool()

    def set_addr(self, addr: str) -> str:
        """Set the addr of the node."""
        self.learner.set_addr(addr)
        return super().set_addr(addr)

    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """
        Set the model of the learner (not weights).

        Args:
            model: The model of the learner.

        """
        self.learner.set_model(model)

    def get_model(self) -> P2PFLModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        return self.learner.get_model()

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner. It is used to fit the model.

        Args:
            data: The data of the learner.

        """
        self.learner.set_data(data)

    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The data of the learner.

        """
        return self.learner.get_data()

    def indicate_aggregator(self, aggregator: Aggregator) -> None:
        """
        Indicate to the learner the aggregators that are being used in order to instantiate the callbacks.

        Args:
            aggregator: The aggregator used in the learning process.

        """
        return self.learner.indicate_aggregator(aggregator)

    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.

        """
        self.learner.set_epochs(epochs)

    def update_callbacks_with_model_info(self) -> None:
        """Update the callbacks with the model additional information."""
        self.learner.update_callbacks_with_model_info()

    def add_callback_info_to_model(self) -> None:
        """Add the additional information from the callbacks to the model."""
        self.learner.add_callback_info_to_model()

    def fit(self) -> P2PFLModel:
        """Fit the model."""
        try:
            self.actor_pool.submit_learner_job(
                lambda actor, addr, learner: actor.fit.remote(addr, learner),
                (str(self.addr), self.learner),
            )
            model: P2PFLModel = self.actor_pool.get_learner_result(str(self.addr), None)[1]
            self.learner.set_model(model)
            return model
        except Exception as ex:
            logger.error(self.addr, f"An error occurred during remote fit: {ex}")
            raise ex

    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        # TODO: Need to implement this!
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        try:
            self.actor_pool.submit_learner_job(
                lambda actor, addr, learner: actor.evaluate.remote(addr, learner),
                (str(self.addr), self.learner),
            )
            result: dict[str, float] = self.actor_pool.get_learner_result(str(self.addr), None)[1]
            return result
        except Exception as ex:
            logger.error(self.addr, f"An error occurred during remote evaluation: {ex}")
            raise ex

    def get_framework(self) -> str:
        """Return the framework of the wrapped learner."""
        return self.learner.get_framework()
