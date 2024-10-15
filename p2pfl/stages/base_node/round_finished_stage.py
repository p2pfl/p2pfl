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

"""Round Finished Stage."""

from typing import Callable, Optional, Type, Union

from p2pfl.commands.metrics_command import MetricsCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class RoundFinishedStage(Stage):
    """Round Finished Stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "RoundFinishedStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        aggregator: Optional[Aggregator] = None,
        early_stopping_fn: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or early_stopping_fn is None:
            raise Exception("Invalid parameters on RoundFinishedStage.")

        # Check if early stopping
        if early_stopping_fn():
            logger.info(state.addr, "Early stopping.")
            return None

        # Set Next Round
        aggregator.clear()
        state.increase_round()
        logger.round_finished(state.addr)

        # Next Step or Finish
        logger.info(
            state.addr,
            f"Round {state.round} of {state.total_rounds} finished.",
        )
        if state.round is None or state.total_rounds is None:
            raise Exception("Round or total rounds not set.")
        if state.round < state.total_rounds:
            return StageFactory.get_stage("TrainStage")
        else:
            # At end, all nodes compute metrics
            RoundFinishedStage.__evaluate(state, communication_protocol)
            # Finish
            state.clear()
            state.model_initialized_lock.acquire()
            logger.info(state.addr, "Training finished!!.")
            return None

    @staticmethod
    def __evaluate(state: NodeState, communication_protocol: CommunicationProtocol) -> None:
        logger.info(state.addr, "Evaluating...")
        if state.learner is None:
            raise Exception("Learner not initialized.")
        results = state.learner.evaluate()
        logger.info(state.addr, f"Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )
