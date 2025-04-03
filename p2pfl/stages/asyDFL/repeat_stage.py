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
"""Start learning stage."""

from typing import Optional, Type, Union

from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.asyDFL.stage_factory import AsyDFLStageFactory
from p2pfl.stages.stage import Stage


class RepeatStage(Stage): # TODO: Implement better graph structure
    """Repeat stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "RepeatStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state.round < state.total_rounds:
            # Train
            return AsyDFLStageFactory.get_stage("LocalUpdateStage")
        else:
            # Finish
            state.clear()
            logger.info(state.addr, "😋 Training finished!!")
            return None
