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

"""StartLearning command."""

from typing import Callable, Optional

from p2pfl.commands.command import Command


class StartLearningCommand(Command):
    """StartLearning command."""

    def __init__(self, start_learning_fn: Callable[[int, int], None]) -> None:
        """Initialize the command."""
        super().__init__()
        self.__learning_fn = start_learning_fn

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "start_learning"

    def execute(
        self,
        source: str,
        round: int,
        learning_rounds: Optional[int] = None,
        learning_epochs: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Execute the command. Start learning thread.

        Args:
            source: The source of the command.
            round: The round of the command.
            learning_rounds: The number of learning rounds.
            learning_epochs: The number of learning epochs.
            **kwargs: The command keyword arguments.

        """
        if learning_rounds is None or learning_epochs is None:
            raise ValueError("Learning rounds and epochs are required")
        self.__learning_fn(int(learning_rounds), int(learning_epochs))
