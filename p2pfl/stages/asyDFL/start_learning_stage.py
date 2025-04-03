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
"""Start learning stage."""

from typing import Optional, Type, Union

from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.asyDFL.stage_factory import AsyDFLStageFactory
from p2pfl.stages.stage import Stage


class StartLearningStage(Stage):
    """Start learning stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "StartLearningStage"

    @staticmethod
    def execute(
        rounds: Optional[int] = None,
        epochs: Optional[int] = None,
        state: Optional[NodeState] = None,
        learner: Optional[Learner] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        aggregator: Optional[Aggregator] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if rounds is None or epochs is None or state is None or learner is None or communication_protocol is None or aggregator is None:
            raise Exception("Invalid parameters on StartLearningStage.")

        # Init
        state.set_experiment("experiment", rounds)
        learner.set_epochs(epochs)
        logger.experiment_started(state.addr, state.experiment)

        # Set train set
        state.train_set = communication_protocol.get_neighbors(only_direct=False)
        aggregator.set_nodes_to_aggregate(state.train_set)

        # Initialize asynchronous DFL variables
        logger.info(state.addr, "Initializing local model and parameters...")
        #learner.get_model().get_model().de_biased_model = learner.get_model().clone_model()  # χ(0) = ω(0)
        out_neighbors = communication_protocol.get_neighbors(only_direct=True)
        state.mixing_weights = {neighbor: 1.0 / len(out_neighbors) if out_neighbors else 1.0 for neighbor in out_neighbors}  # pt_j,i(0)
        state.push_times = {} # tp_j,i(0)
        for neighbor_id in out_neighbors:
            state.push_times[neighbor_id] = 0
        state.reception_times = {} # tl_j,i(0)
        state.losses = {} # f_i(ω(0))
        state.push_sum_weights = {} # μ_j(0)
        state.push_sum_weights[state.addr] = 1.0 # μ(0)
        state.tau = 2  # τ

        # Setup learner
        learner.set_steps_per_epoch(1)
        learner.set_epochs(1)

        # Vote
        return AsyDFLStageFactory.get_stage("LocalUpdateStage")
