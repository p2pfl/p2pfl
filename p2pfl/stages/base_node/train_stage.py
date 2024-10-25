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
"""Train stage."""

from typing import Any, List, Optional, Set, Type, Union

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.learner import NodeLearner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "TrainStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        learner: Optional[NodeLearner] = None,
        aggregator: Optional[Aggregator] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on TrainStage.")

        try:
            check_early_stop(state)

            # Set Models To Aggregate
            aggregator.set_nodes_to_aggregate(state.train_set)

            check_early_stop(state)

            # Evaluate and send metrics
            TrainStage.__evaluate(state, learner, communication_protocol)

            check_early_stop(state)

            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            learner.fit()

            check_early_stop(state)

            # Aggregate Model
            models_added = aggregator.add_model(learner.get_model())
            # send model added msg ---->> redundant (a node always owns its model)
            # TODO: print("Broadcast redundante")
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=state.round,
                )
            )
            TrainStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

            check_early_stop(state)

            # Set aggregated model
            agg_model = aggregator.wait_and_get_aggregation()
            learner.set_model(agg_model)

            # Share that aggregation is done
            communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

            # Next stage
            return StageFactory.get_stage("GossipModelStage")
        except EarlyStopException:
            return None

    @staticmethod
    def __evaluate(state: NodeState, learner: NodeLearner, communication_protocol: CommunicationProtocol) -> None:
        logger.info(state.addr, "🔬 Evaluating...")
        results = learner.evaluate()
        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
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

    @staticmethod
    def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> List[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStage.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStage.__get_aggregated_models(n, state),
                )  # reemplazar por Aggregator - borrarlo de node
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> Any:
            try:
                model = aggregator.get_partial_aggregation(TrainStage.__get_aggregated_models(node, state))
            except NoModelsToAggregateError:
                return None
            if state.round is None:
                raise Exception("Round not initialized.")

            return communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True,
        )

    @staticmethod
    def __get_aggregated_models(node: str, state: NodeState) -> List[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []

    @staticmethod
    def __get_remaining_nodes(node: str, state: NodeState) -> Set[str]:
        return set(state.train_set) - set(TrainStage.__get_aggregated_models(node, state))
