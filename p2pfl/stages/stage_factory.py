class StageFactory:
    """
    Main goal: Avoid cyclic imports.
    """

    @staticmethod
    def get_stage(stage_name: str):
        if stage_name == "StartLearningStage":
            from p2pfl.stages.base_node.start_learning_stage import StartLearningStage

            return StartLearningStage
        elif stage_name == "RoundFinishedStage":
            from p2pfl.stages.base_node.round_finished_stage import RoundFinishedStage

            return RoundFinishedStage
        elif stage_name == "WaitAggregatedModelsStage":
            from p2pfl.stages.base_node.wait_agg_models_stage import (
                WaitAggregatedModelsStage,
            )

            return WaitAggregatedModelsStage
        elif stage_name == "GossipModelStage":
            from p2pfl.stages.base_node.gossip_model_stage import GossipModelStage

            return GossipModelStage
        elif stage_name == "TrainStage":
            from p2pfl.stages.base_node.train_stage import TrainStage

            return TrainStage
        elif stage_name == "VoteTrainSetStage":
            from p2pfl.stages.base_node.vote_train_set_stage import VoteTrainSetStage

            return VoteTrainSetStage
        else:
            raise Exception("Invalid stage name.")
