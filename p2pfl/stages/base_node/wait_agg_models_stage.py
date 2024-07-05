from typing import Union
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.management.logger import logger
from p2pfl.stages.stage_factory import StageFactory


class WaitAggregatedModelsStage(Stage):

    @staticmethod
    def name():
        return "WaitAggregatedModelsStage"

    @staticmethod
    def execute(
        state: NodeState = None, aggregator: Aggregator = None, **kwargs
    ) -> Union["Stage", None]:
        if state is None or aggregator is None:
            raise Exception("Invalid parameters on WaitAggregatedModelsStage.")
        logger.info(state.addr, "Waiting aregation.")
        """
        Quizá pueda ser interesante que la lógica de espera esté aquí
        """
        aggregator.set_waiting_aggregated_model(state.train_set)
        print("going to the GossipModelStage")
        return StageFactory.get_stage("GossipModelStage")
