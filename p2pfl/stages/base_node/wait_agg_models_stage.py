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
"""Wait aggregated models stage."""

from typing import Optional, Type, Union

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class WaitAggregatedModelsStage(Stage):
    """Wait aggregated models stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "WaitAggregatedModelsStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None, aggregator: Optional[Aggregator] = None, **kwargs
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or aggregator is None:
            raise Exception("Invalid parameters on WaitAggregatedModelsStage.")
        logger.info(state.addr, "Waiting aregation.")
        """
        Quizá pueda ser interesante que la lógica de espera esté aquí
        """
        aggregator.set_waiting_aggregated_model(state.train_set)
        return StageFactory.get_stage("GossipModelStage")
