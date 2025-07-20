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

"""InitModel command."""

from collections.abc import Callable

from p2pfl.communication.commands.command import Command
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class InitModelCommand(Command):
    """InitModelCommand."""

    def __init__(self, state: NodeState, stop: Callable[[], None], aggregator: Aggregator, learner: Learner) -> None:
        """Initialize InitModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.learner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "init_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: bytes | None = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None:
            logger.error(self.state.addr, "Invalid InitModelCommand message")
            return

        # Check if Learning is running
        if self.state.round is not None:
            # Check source
            if round != self.state.round:
                logger.debug(
                    self.state.addr,
                    f"Model initiallization in a late round ({round} != {self.state.round}).",
                )
                return

            # Check moment (not init and invalid round)
            if not self.state.model_initialized_lock.locked():
                logger.info(
                    self.state.addr,
                    "🕳  Model initizalization message when the model is already initialized. Ignored.",
                )
                return

            try:
                # Set new weights
                self.learner.set_model(weights)
                # Release lock - wrap in try/except to handle concurrent releases
                try:
                    self.state.model_initialized_lock.release()
                    logger.info(self.state.addr, "🤖 Model Weights Initialized")
                except RuntimeError:
                    # This likely means another concurrent INIT_MODEL message already released the lock.
                    logger.debug(
                        self.state.addr,
                        "Attempted to release model_initialized_lock, but it was already unlocked. Likely due to concurrent initialization",
                    )

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.state.addr, "Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError:
                logger.error(self.state.addr, "Models not matching.")
                self.stop()

            except Exception as e:
                logger.error(self.state.addr, f"Unknown error adding initial model: {e}")
                self.stop()

        else:
            logger.debug(self.state.addr, "Tried to add a model while learning is not running")
