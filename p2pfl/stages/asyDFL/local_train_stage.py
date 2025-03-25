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
"""Local Train stage."""

from typing import Optional, Type, Union

from p2pfl.communication.commands.message.asyDFL.ctx_info_updating_command import ContextInformationUpdatingCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.asyDFL.stage_factory import AsyDFLStageFactory
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop


class LocalUpdateStage(Stage):
    """Train stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "LocalUpdateStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        learner: Optional[Learner] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or communication_protocol is None or learner is None:
            raise Exception("Invalid parameters on LocalUpdateStage.")

        try:
            # Check for early stopping before training
            check_early_stop(state)

            # De-bias the update model (Equation 3)
            de_biased_model = learner.get_model()
            de_biased_model.get_model().de_biased_model = [
                param / state.push_sum_weights[state.addr] for param in learner.get_model().get_parameters()
            ]
            learner.set_model(de_biased_model)

            logger.info(state.addr, "🏋️‍♀️ Updating local model...")
            learner.fit()

            check_early_stop(state)

            # Evaluate and send metrics
            LocalUpdateStage.__train(state, learner, communication_protocol)

            check_early_stop(state)

            # Next stage
            return AsyDFLStageFactory.get_stage("NeighborSelectionStage")

        except EarlyStopException:
            return None

    @staticmethod
    def __train(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol) -> None:
        logger.info(state.addr, "🏋️‍♀️ Updating local model...")
        learner.fit()

        logger.info(state.addr, "📢 Broadcasting loss values.")
        flattened_loss = str(learner.model.get_model().last_training_loss)
        communication_protocol.broadcast(
            communication_protocol.build_msg(
                ContextInformationUpdatingCommand.get_name(),
                flattened_loss,
                round=state.round,
            )
        )
