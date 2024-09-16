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

from p2pfl.learning.learner import NodeLearner
from p2pfl.simulation.actor_pool import SuperActorPool
from typing import Any, Optional, Tuple


class VirtualNodeLearner(NodeLearner):
    """
    Decorator for the learner to be used in the simulation.
    """
    def __init__(self, 
                 learner: NodeLearner,
                 model: Any,
                 data: Any,
                 addr: str,
                 epochs: int
        ):
        self.learner = learner(model, data, addr, epochs)
        self.actor_pool = SuperActorPool()
        self.addr = addr

    def set_model(self, model: Any) -> None:
        self.learner.set_model(model)

    def set_data(self, data: Any) -> None:
        self.learner.set_data(data)

    def encode_parameters(self, params: Optional[Any] = None) -> bytes:
        return self.learner.encode_parameters(params)

    def decode_parameters(self, data: bytes) -> Any:
        return self.learner.decode_parameters(data)

    def check_parameters(self, params: Any) -> bool:
        return self.learner.check_parameters(params)

    def set_parameters(self, params: Any) -> None:
        self.learner.set_parameters(params)

    def get_parameters(self) -> Any:
        return self.learner.get_parameters()

    def set_epochs(self, epochs: int) -> None:
        self.learner.set_epochs(epochs)

    def fit(self) -> None:
        try:
            self.actor_pool.submit_learner_job(
                lambda actor, addr, learner: actor.fit.remote(addr, learner),
                (str(self.addr),self.learner),
            )
            #self.model = self.actor_pool.get_learner_result(str(self.addr), None)
        except Exception as ex:
            print(f"An error occurred during fit: {ex}")
            raise ex

    def interrupt_fit(self) -> None:
        self.actor_pool.submit(
            lambda actor: actor.interrupt_fit.remote(),
            (),
        )

    def evaluate(self) -> Tuple[float, float]:
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
            print(f"An error occurred during evaluation: {ex}")
            raise ex

    def get_num_samples(self) -> Tuple[int, int]:
        try:
            self.actor_pool.submit_learner_job(
                lambda actor, addr, learner: actor.get_num_samples.remote(addr, learner),
                (str(self.addr), self.learner),
            )
            result = self.actor_pool.get_learner_result(str(self.addr), None)
            return result
        except Exception as ex:
            print(f"An error occurred while getting the number of samples: {ex}")
            raise ex