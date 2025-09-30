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
"""Gossip model stage."""

from typing import Any

from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class GossipModelStage(Stage):
    """Gossip model stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "GossipModelStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        aggregator: Aggregator | None = None,
        learner: Learner | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or aggregator is None or communication_protocol is None or learner is None:
            raise Exception("Invalid parameters on GossipModelStage.")

        GossipModelStage.__gossip_model_difusion(state, communication_protocol, learner)
        return StageFactory.get_stage("RoundFinishedStage")

    @staticmethod
    def __gossip_model_difusion(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
    ) -> None:
        logger.info(state.addr, "🗣️ Gossiping aggregated model.")
        fixed_round = state.round
        if fixed_round is None:
            raise Exception("Learner not initialized")

        def candidate_condition(node: str) -> bool:
            return state.nei_status[node] < fixed_round

        def get_candidates_fn() -> list[str]:
            return [n for n in communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]

        def status_fn() -> Any:
            return get_candidates_fn()

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized")
            encoded_model = learner.get_model().encode_parameters()
            return (
                communication_protocol.build_weights(FullModelCommand.get_name(), state.round, encoded_model),
                FullModelCommand.get_name(),
                state.round,
                [str(state.round)],
            )

        # Gossip
        communication_protocol.gossip_weights(
            lambda: check_early_stop(state, raise_exception=False),
            get_candidates_fn,
            status_fn,
            model_fn,
        )
