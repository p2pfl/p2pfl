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

"""Metrics command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class MetricsCommand(Command):
    """MetricsCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        super().__init__()
        self._node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "metrics"

    def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: Metric values (pairs of key and values).
            **kwargs: The command keyword arguments.

        """
        for i in range(0, len(args), 2):
            key = args[i]
            value = float(args[i + 1])
            logger.log_metric(source, metric=key, value=value, round=round)
