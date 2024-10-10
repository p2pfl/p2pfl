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
"""Wait aggregated models stage."""

from typing import Optional, Type, Union

from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
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
        state: Optional[NodeState] = None, communication_protocol: Optional[CommunicationProtocol] = None, **kwargs
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or communication_protocol is None:
            raise Exception("Invalid parameters on WaitAggregatedModelsStage.")
        # clear here instead of aquiring in vote_train_set_stage
        state.wait_aggregated_model_event.clear()
        logger.info(state.addr, "⏳ Waiting aggregation.")
        # Wait for aggregation to finish, if time over timeout log a warning message
        event_set = state.wait_aggregated_model_event.wait(timeout=Settings.AGGREGATION_TIMEOUT)

        if event_set:
            # The event was set before the timeout
            logger.info(state.addr, "✅ Aggregation event received.")
        else:
            # The timeout occurred before the event was set
            logger.warning(state.addr, "⏰ Aggregation timeout occurred.")

        # Get aggregated model
        logger.debug(
            state.addr,
            f"Broadcast aggregation done for round {state.round}",
        )
        # Share that aggregation is done
        communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

        return StageFactory.get_stage("GossipModelStage")
