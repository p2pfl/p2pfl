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

"""ModelsAggregated command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class ModelsAggregatedCommand(Command):
    """ModelsAggregated command."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        self._node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "models_aggregated"

    def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: List of models that contribute to the aggregated model.
            **kwargs: The command keyword arguments.

        """
        if round == self._node.state.round:
            # esto meterlo en estado o agg
            self._node.state.models_aggregated[source] = list(args)
        else:
            logger.debug(
                self._node.state.addr,
                f"Models Aggregated message from {source} in a late round. Ignored. {round} != {self._node.state.round}",
            )
