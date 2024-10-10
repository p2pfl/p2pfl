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

"""FullModelCommand."""

from typing import Callable, Optional

from p2pfl.communication.commands.command import Command
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.learner import NodeLearner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class FullModelCommand(Command):
    """FullModelCommand."""

    def __init__(self, state: NodeState, stop: Callable[[], None], aggregator: Aggregator, learner: NodeLearner) -> None:
        """Initialize FullModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.learner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "add_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: Optional[bytes] = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None:
            raise ValueError("Weights, contributors and weight are required")

        # Check if Learning is running
        if self.state.round is not None:
            # Check source
            if round != self.state.round:
                logger.debug(
                    self.state.addr,
                    f"Model reception in a late round ({round} != {self.state.round}).",
                )
                return

            try:
                logger.info(self.state.addr, "📦 Aggregated model received.")
                # Decode and set model
                self.learner.set_model(weights)
                # Release here caused the simulation to crash before
                self.state.wait_aggregated_model_event.set()

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.state.addr, "❌ Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError:
                logger.error(self.state.addr, "❌ Models not matching.")
                self.stop()

            except Exception as e:
                logger.error(self.state.addr, f"❌ Unknown error adding model: {e}")
                self.stop()
        else:
            logger.debug(self.state.addr, "❌ Tried to add a model while learning is not running")
