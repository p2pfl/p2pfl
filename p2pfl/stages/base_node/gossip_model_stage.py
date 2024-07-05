from typing import Any, List, Callable, Union
from p2pfl.commands.add_model_command import AddModelCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.management.logger import logger
from p2pfl.commands.models_ready_command import ModelsReadyCommand
from p2pfl.stages.stage_factory import StageFactory


class GossipModelStage(Stage):

    @staticmethod
    def name():
        return "GossipModelStage"
    
    @staticmethod
    def execute(
        state: NodeState = None,
        communication_protocol: CommunicationProtocol = None,
        aggregator: Aggregator = None,
        early_stopping_fn: Callable[[], bool] = None,
        **kwargs
    ) -> Union["Stage", None]:
        if (
            state is None
            or aggregator is None
            or early_stopping_fn is None
            or communication_protocol is None
        ):
            raise Exception("Invalid parameters on GossipModelStage.")

        if not early_stopping_fn():
            GossipModelStage.__wait_aggregated_model(
                state, communication_protocol, aggregator
            )

        if not early_stopping_fn():
            GossipModelStage.__gossip_model_difusion()

        return StageFactory.get_stage("RoundFinishedStage")

    def __wait_aggregated_model(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        params = aggregator.wait_and_get_aggregation()

        # Set parameters and communate it to the training process
        if params is not None:
            state.learner.set_parameters(params)
            logger.debug(
                state.addr,
                f"Broadcast aggregation done for round {state.round}",
            )
            # Share that aggregation is done
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsReadyCommand.get_name(), [], round=state.round
                )
            )
        else:
            raise Exception("Aggregation finished with no parameters")

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

        def candidate_condition(node: str) -> bool:
            return state.nei_status[node] < fixed_round

        def get_candidates_fn() -> List[str]:
            return [
                n
                for n in communication_protocol.get_neighbors(only_direct=True)
                if candidate_condition(n)
            ]

        def status_fn() -> Any:
            return get_candidates_fn()

        def model_fn(node: str) -> Any:
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
