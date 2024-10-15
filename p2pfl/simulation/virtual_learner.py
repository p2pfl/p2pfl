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

import numpy as np
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.simulation.actor_pool import SuperActorPool
from typing import Any, List, Optional, Tuple, Union


class VirtualNodeLearner(NodeLearner):
    """
    Decorator for the learner to be used in the simulation.
    """
    def __init__(self, 
                 learner: NodeLearner,
                 addr: str) -> None:
        """Initialize the learner."""
        self.learner = learner
        self.actor_pool = SuperActorPool()
        self.addr = addr

    def set_addr(self, addr: str) -> None:
        """
        Set the address of the learner.

        Args:
            addr: The address of the learner.

        """
        self.learner.set_addr(addr)
        self.addr = addr

    def set_model(self, model: Union[P2PFLModel, List[np.ndarray], bytes]) -> None:
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
        self.learner.get_data()

    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs of the model.

        Args:
            epochs: The number of epochs of the model.

        """
        self.learner.set_epochs(epochs)

    def fit(self) -> None:
        """Fit the model."""
        try:
            self.actor_pool.submit_learner_job(
                lambda actor, addr, learner: actor.fit.remote(addr, learner),
                (str(self.addr),self.learner),
            )
            self.learner.set_model(self.actor_pool.get_learner_result(str(self.addr), None))
        except Exception as ex:
            print(f"An error occurred during remote fit: {ex}")
            raise ex

    def interrupt_fit(self) -> None:
        """Interrupt the fit process."""
        self.actor_pool.submit(
            lambda actor: actor.interrupt_fit.remote(),
            (),
        )

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        try:
            self.actor_pool.submit_learner_job(
                lambda actor, addr, learner: actor.evaluate.remote(
                    addr, learner
                ),
                (str(self.addr), self.learner),
            )
            result = self.actor_pool.get_learner_result(str(self.addr), None)
            return result
        except Exception as ex:
            print(f"An error occurred during remote evaluation: {ex}")
            raise ex