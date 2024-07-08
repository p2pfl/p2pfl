from typing import Any, Callable, List, Union

from p2pfl.commands.add_model_command import AddModelCommand
from p2pfl.commands.metrics_command import MetricsCommand
from p2pfl.commands.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class TrainStage(Stage):
    @staticmethod
    def name():
        return "TrainStage"

    @staticmethod
    def execute(
        state: NodeState = None,
        communication_protocol: CommunicationProtocol = None,
        aggregator: Aggregator = None,
        early_stopping_fn: Callable[[], bool] = None,
        **kwargs,
    ) -> Union["Stage", None]:
        if (
            state is None
            or communication_protocol is None
            or aggregator is None
            or early_stopping_fn is None
        ):
            raise Exception("Invalid parameters on TrainStage.")

        # Set nodes to agg
        if not early_stopping_fn():
            # Set Models To Aggregate
            aggregator.set_nodes_to_aggregate(state.train_set)

        # Evaluate and send metrics
        if not early_stopping_fn():
            TrainStage.__evaluate(state, communication_protocol)

        # Train
        if not early_stopping_fn():
            TrainStage.__train(state)

        # Aggregate Model
        if not early_stopping_fn():
            models_added = aggregator.add_model(
                state.learner.get_parameters(),
                [state.addr],
                state.learner.get_num_samples()[0],
            )
            # send model added msg ---->> redundant (a node always owns its model)
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=state.round,
                )
            )
            TrainStage.__gossip_model_aggregation(
                state, communication_protocol, aggregator
            )

        # Next stage
        return StageFactory.get_stage("GossipModelStage")

    def __train(state: NodeState) -> None:
        logger.info(state.addr, "Training...")
        state.learner.fit()

    def __evaluate(
        state: NodeState, communication_protocol: CommunicationProtocol
    ) -> None:
        logger.info(state.addr, "Evaluating...")
        results = state.learner.evaluate()
        logger.info(state.addr, f"Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "Broadcasting metrics.")
            flattened_metrics = [item for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )

    def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        CAREFULL: full connected trainset to increase aggregation speed. On real scenarios, this won't be possible, private networks and firewalls.
        Needed because the trainset can split the networks (and neighbors that are not in the trainset won't receive the aggregation).
        """

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> List[str]:
            return [
                n
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n not in aggregator.get_aggregated_models())
                and (n in state.train_set)
            ]

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
            model, contributors, weight = aggregator.get_partial_aggregation(
                TrainStage.__get_aggregated_models(
                    node, state
                )  # reemplazar por Aggregator - borrarlo de node
            )
            if model is None:
                return None
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
            create_connection=True,
        )

    def __get_aggregated_models(node: str, state: NodeState) -> List[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []
