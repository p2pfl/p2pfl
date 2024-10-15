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

from typing import Tuple
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
import ray

@ray.remote
class VirtualLearnerActor:
    def terminate(self) -> None:
        """Manually terminate Actor object."""
        logger.debug(self.__class__.__name__, f"Manually terminating {self.__class__.__name__}")
        ray.actor.exit_actor()

    def fit(self,
        learner: NodeLearner,
        addr: str
    ) -> Tuple[str,P2PFLModel]:
        """Fit the model."""
        try:
            model = learner.fit()

        except Exception as ex:
            raise ex
        
        return addr, model
    
    def evaluate(self,
        learner: NodeLearner,
        addr: str
    ) -> Tuple[str, Tuple[float,float]]:
        """Evaluate the model."""
        try:
            results = learner.evaluate()

        except Exception as ex:
            raise ex
        
        return addr, results
    
    def interrupt_fit(self,
        learner: NodeLearner,
        addr: str
    ) -> Tuple[str, None]:
        """Interrupt the fit."""
        try:
            learner.interrupt_fit()

        except Exception as ex:
            raise ex
        
        return addr, None