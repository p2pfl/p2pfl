from typing import Callable, Union

from p2pfl.commands.metrics_command import MetricsCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class RoundFinishedStage(Stage):
    @staticmethod
    def name():
        return "RoundFinishedStage"

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

    def __evaluate(state: NodeState, communication_protocol: CommunicationProtocol) -> None:
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
