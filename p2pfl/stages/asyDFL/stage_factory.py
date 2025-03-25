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

from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class AsyDFLStageFactory(StageFactory):
    """Factory class to create stages. Main goal: Avoid cyclic imports."""

    @staticmethod
    def get_stage(stage_name: str) -> type[Stage]:
        """Return the stage class."""
        if stage_name == "StartLearningStage":
            from p2pfl.stages.asyDFL.start_learning_stage import StartLearningStage

            return StartLearningStage
        elif stage_name == "LocalUpdateStage":
            from p2pfl.stages.asyDFL.local_train_stage import LocalUpdateStage

            return LocalUpdateStage
        elif stage_name == "NeighborSelectionStage":
            from p2pfl.stages.asyDFL.neighbor_selection_stage import (
                NeighborSelectionStage,
            )

            return NeighborSelectionStage
        elif stage_name == "RepeatStage":
            from p2pfl.stages.asyDFL.repeat_stage import RepeatStage

            return RepeatStage
        else:
            raise Exception("Invalid stage name.")
