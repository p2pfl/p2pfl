#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""StopLearning command."""

import contextlib

from p2pfl.communication.commands.command import Command
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class StopLearningCommand(Command):
    """StopLearning command."""

    def __init__(self, state: NodeState, aggregator: Aggregator, learner: Learner) -> None:
        """Initialize the command."""
        self.state = state
        self.aggregator = aggregator
        self.learner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "stop_learning"

    def execute(self, source: str, round: int, **kwargs) -> None:
        """
        Execute the command. Start learning thread.

        Args:
            source: The source of the command.
            round: The round of the command.
            **kwargs: The command keyword arguments.

        """
        logger.info(self.state.addr, "Stopping learning received")
        # Leraner
        self.learner.interrupt_fit()
        # Aggregator
        self.aggregator.clear()
        # State
        self.state.clear()
        logger.experiment_finished(self.state.addr)
        # Try to free wait locks
        with contextlib.suppress(Exception):
            self.state.wait_votes_ready_lock.release()
