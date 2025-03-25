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

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from p2pfl.communication.commands.command import Command
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class InitModelCommand(Command):
    """InitModelCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize InitModelCommand."""
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "init_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: Optional[bytes] = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None:
            logger.error(self.__node.state.addr, "Invalid InitModelCommand message")
            return

        # Check if Learning is running
        if self.__node.state.round is not None:
            # Check source
            if round != self.__node.state.round:
                logger.debug(
                    self.__node.state.addr,
                    f"Model initiallization in a late round ({round} != {self.state.round}).",
                )
                return

            # Check moment (not init and invalid round)
            if not self.__node.state.model_initialized_lock.locked():
                logger.error(
                    self.__node.state.addr,
                    "Model initizalization message when the model is already initialized. Ignored.",
                )
                return

            try:
                # Set new weights
                self.__node.learner.set_model(weights)
                # Release lock
                self.__node.state.model_initialized_lock.release()
                logger.info(self.__node.state.addr, "🤖 Model Weights Initialized")

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.__node.state.addr, "Error decoding parameters.")
                self.__node.stop()

            except ModelNotMatchingError:
                logger.error(self.__node.state.addr, "Models not matching.")
                self.__node.stop()

            except Exception as e:
                logger.error(self.__node.state.addr, f"Unknown error adding model: {e}")
                self.__node.stop()

        else:
            logger.debug(self.__node.state.addr, "Tried to add a model while learning is not running")
