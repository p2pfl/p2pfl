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
"""Workflows."""

from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class StageWokflow:
    """Class to run a workflow of stages."""

    def __init__(self, first_stage: type[Stage]) -> None:
        """Initialize the workflow."""
        self.current_stage = first_stage
        self.history: list[str] = []
        self.finished = False

    def run(self, **kwargs) -> None:
        """Run the workflow."""
        self.finished = False
        # get state (need info from state)
        state: NodeState | None = kwargs.get("state")
        if state:
            while True:
                logger.debug(state.addr, f"🏃 Running stage: {(self.current_stage.name())}")
                self.history.append(self.current_stage.name())
                next_stage = self.current_stage.execute(**kwargs)
                if next_stage is None or check_early_stop(state, raise_exception=False):
                    self.finished = True
                    break
                self.current_stage = next_stage
        else:
            raise ValueError("State not found in kwargs")


class LearningWorkflow(StageWokflow):
    """Class to run a federated learning workflow."""

    def __init__(self) -> None:
        """Initialize the federated learning workflow."""
        super().__init__(StageFactory.get_stage("StartLearningStage"))
