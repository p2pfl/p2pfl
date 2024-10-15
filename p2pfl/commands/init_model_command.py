#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

from typing import List, Optional

from p2pfl.commands.command import Command
from p2pfl.commands.model_initialized_command import ModelInitializedCommand
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger


class InitModelCommand(Command):
    """InitModelCommand."""

    def __init__(
        self,
        state,
        stop,
        aggregator,
        comm_proto,
    ) -> None:
        """Initialize InitModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.communication_protocol = comm_proto

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "init_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: Optional[bytes] = None,
        contributors: Optional[List[str]] = None,
        weight: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            weights: The weights of the model.
            contributors: The contributors of the command.
            weight: The weight of the model (ammount of samples).
            **kwargs: The command arguments.

        """
        if weights is None or contributors is None or weight is None:
            logger.error(self.state.addr, "Invalid message")
            return

        # Check if Learning is running
        if self.state.learner is not None:
            # Check source
            if round != self.state.round:
                logger.debug(
                    self.state.addr,
                    f"Model reception in a late round ({round} != {self.state.round}).",
                )
                return

            # Check moment (not init and invalid round)
            if not self.state.model_initialized_lock.locked():
                logger.error(
                    self.state.addr,
                    "Model initizalization message when the model is already initialized. Ignored.",
                )
                return

            try:
                model = self.state.learner.decode_parameters(weights)
                self.state.learner.set_parameters(model)
                self.state.model_initialized_lock.release()
                logger.info(self.state.addr, "Model Weights Initialized")
                # Communicate Initialization
                self.communication_protocol.broadcast(
                    self.communication_protocol.build_msg(ModelInitializedCommand.get_name())
                )

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.state.addr, "Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError:
                logger.error(self.state.addr, "Models not matching.")
                self.stop()

            except Exception as e:
                logger.error(self.state.addr, f"Unknown error adding model: {e}")
                self.stop()

        else:
            logger.debug(self.state.addr, "Tried to add a model while learning is not running")
