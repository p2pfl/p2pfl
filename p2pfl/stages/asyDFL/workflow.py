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
from p2pfl.stages.asyDFL.stage_factory import AsyDFLStageFactory
from p2pfl.stages.workflows import StageWorkflow


class AsyDFLWorkflow(StageWorkflow):
    """Class to run a federated learning workflow."""

    def __init__(self) -> None:
        """Initialize the federated learning workflow."""
        super().__init__(AsyDFLStageFactory.get_stage("StartLearningStage"))
