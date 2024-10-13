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

"""Stage factory."""

from typing import Type

from p2pfl.stages.stage import Stage

#
# OJO WORKFLOWS DUPLICADOS, ALGUNOS SON IGUALES PARA EL PROXY Y PARA EL BASE NODE
#

class StageFactory:
    """Factory class to create stages. Main goal: Avoid cyclic imports."""

    @staticmethod
    def get_stage(stage_name: str) -> Type[Stage]:
        """Return the stage class."""
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

        # PROXY
        

        if stage_name == "StartLearningStage_proxy":
            from p2pfl.stages.proxy_node.start_learning_stage import StartLearningStage as StartLearningStage_proxy

            return StartLearningStage_proxy
        elif stage_name == "RoundFinishedStage_proxy":
            from p2pfl.stages.proxy_node.round_finished_stage import RoundFinishedStage as RoundFinishedStage_proxy

            return RoundFinishedStage_proxy
        elif stage_name == "WaitAggregatedModelsStage_proxy":
            from p2pfl.stages.proxy_node.wait_agg_models_stage import WaitAggregatedModelsStage as WaitAggregatedModelsStage_proxy

            return WaitAggregatedModelsStage_proxy
        elif stage_name == "GossipModelStage_proxy":
            from p2pfl.stages.proxy_node.gossip_model_stage import GossipModelStage as GossipModelStage_proxy

            return GossipModelStage_proxy
        elif stage_name == "TrainStage_proxy":
            from p2pfl.stages.proxy_node.train_stage import TrainStage as TrainStage_proxy

            return TrainStage_proxy
        elif stage_name == "VoteTrainSetStage_proxy":
            from p2pfl.stages.proxy_node.vote_train_set_stage import VoteTrainSetStage as VoteTrainSetStage_proxy

            return VoteTrainSetStage_proxy
        else:
            raise Exception("Invalid stage name.")
