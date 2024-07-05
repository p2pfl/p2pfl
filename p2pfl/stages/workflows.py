from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory
from p2pfl.management.logger import logger


class StageWokflow:
    def __init__(self, first_stage: Stage):
        self.current_stage = first_stage

    def run(self, **kwargs):

        # get state
        state: NodeState = kwargs.get("state")

        while True:
            logger.debug(state.addr, f"Running stage: {(self.current_stage.name())}")
            self.current_stage = self.current_stage.execute(**kwargs)
            if self.current_stage is None:
                break


class LearningWorkflow(StageWokflow):
    def __init__(self):
        super().__init__(StageFactory.get_stage("TrainStage"))
