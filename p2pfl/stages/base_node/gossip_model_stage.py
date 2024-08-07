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
"""Gossip model stage."""

from typing import Any, Callable, List, Optional, Type, Union

from p2pfl.commands.add_model_command import AddModelCommand
from p2pfl.commands.models_ready_command import ModelsReadyCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class GossipModelStage(Stage):
    """Gossip model stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "GossipModelStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        aggregator: Optional[Aggregator] = None,
        early_stopping_fn: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or aggregator is None or early_stopping_fn is None or communication_protocol is None:
            raise Exception("Invalid parameters on GossipModelStage.")

        if not early_stopping_fn():
            GossipModelStage.__wait_aggregated_model(state, communication_protocol, aggregator)

        if not early_stopping_fn():
            GossipModelStage.__gossip_model_difusion(state, communication_protocol, aggregator)

        return StageFactory.get_stage("RoundFinishedStage")

    @staticmethod
    def __wait_aggregated_model(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        params = aggregator.wait_and_get_aggregation()

        # Set parameters and communate it to the training process
        if params is not None:
            if state.learner is None:
                raise Exception("Learner not initialized")
            state.learner.set_parameters(params)
            logger.debug(
                state.addr,
                f"Broadcast aggregation done for round {state.round}",
            )
            # Share that aggregation is done
            communication_protocol.broadcast(
                communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round)
            )
        else:
            raise Exception("Aggregation finished with no parameters")

    @staticmethod
    def __gossip_model_difusion(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        def early_stopping_fn():
            return state.round is None

        # Wait a model (init or aggregated)

        logger.info(state.addr, "Gossiping aggregated model.")
        fixed_round = state.round
        if fixed_round is None:
            raise Exception("Learner not initialized")

        def candidate_condition(node: str) -> bool:
            return state.nei_status[node] < fixed_round

        def get_candidates_fn() -> List[str]:
            return [n for n in communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]

        def status_fn() -> Any:
            return get_candidates_fn()

        def model_fn(node: str) -> Any:
            if state.learner is None:
                raise Exception("Learner not initialized")
            if state.round is None:
                raise Exception("Round not initialized")
            model = state.learner.get_parameters()
            contributors = aggregator.get_aggregated_models()
            weight = 1
            encoded_model = state.learner.encode_parameters(params=model)
            return communication_protocol.build_weights(
                AddModelCommand.get_name(),
                state.round,
                encoded_model,
                contributors,
                weight,
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
        )
