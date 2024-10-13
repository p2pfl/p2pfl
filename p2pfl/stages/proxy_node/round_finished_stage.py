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

"""Round Finished Stage."""

from typing import Optional, Type, Union

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.protocols.p2p.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.learner import NodeLearner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class RoundFinishedStage(Stage):
    """Round Finished Stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "RoundFinishedStage_proxy"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        learner: Optional[NodeLearner] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        aggregator: Optional[Aggregator] = None,
        edge_communication_protocol = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on RoundFinishedStage.")

        # Set Next Round
        aggregator.clear()
        state.increase_round()
        logger.round_finished(state.addr)

        # Next Step or Finish
        logger.info(
            state.addr,
            f"🎉 Round {state.round} of {state.total_rounds} finished.",
        )
        if state.round is None or state.total_rounds is None:
            raise ValueError("Round or total rounds not set.")

        if state.round < state.total_rounds:
            return StageFactory.get_stage("VoteTrainSetStage_proxy")
        else:
            # At end, all nodes compute metrics
            RoundFinishedStage.__evaluate(state, learner, communication_protocol, edge_communication_protocol)
            # Finish
            state.clear()
            logger.info(state.addr, "😋 Training finished!!")
            return None

    @staticmethod
    def __evaluate(
        state: NodeState,
        learner: NodeLearner,
        communication_protocol: CommunicationProtocol,
        edge_communication_protocol
    ) -> None:
        # Send
        logger.info(state.addr, "🔬 Sending eval...")
        val_results = edge_communication_protocol.broadcast_message(
            "validate",
            weights=learner.get_model().encode_parameters(),
            timeout=120
        )
        # Transform
        val_results = {
            k: dict(zip(v.message[::2], v.message[1::2]))
            for k, v in val_results.items() if v is not None
        }
        logger.info(state.addr, f"📈 Evaluated. Results: {val_results}")
        # Promediate
        results = {}
        for _, metrics in val_results.items():
            for metric, value in metrics.items():
                if metric not in results:
                    results[metric] = float(value)
                else:
                    results[metric] += float(value)
        results = {k: v / len(val_results) for k, v in results.items()}

        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )

