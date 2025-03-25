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

"""ModelsReady command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class ModelsReadyCommand(Command):
    """ModelsReady command."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        self._node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "models_ready"

    def execute(self, source: str, round: int, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            **kwargs: The command keyword arguments.

        """
        # revisar validación al igual que en VoteTrainSetCommand
        ########################################################
        # try to improve clarity in message moment check
        ########################################################
        if self._node.state.round is not None:
            if round in [self._node.state.round - 1, self._node.state.round]:
                self._node.state.nei_status[source] = self._node.state.round
            else:
                # Ignored
                logger.error(
                    self._node.state.addr,
                    f"Models ready from {source} in a late round. Ignored. {round} " + f"!= {self._node.state.round} / {self._node.state.round-1}",
                )
        else:
            logger.warning(self._node.state.addr, "Models ready received when learning is not running")
