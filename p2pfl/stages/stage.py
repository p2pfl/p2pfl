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
"""Stage."""

from typing import Type, Union

from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class Stage:
    """Abstract class for a stage."""

    @staticmethod
    def name() -> str:
        """Return the name of the stage."""
        raise NotImplementedError("Stage name not implemented.")

    @staticmethod
    def execute() -> Union[Type["Stage"], None]:
        """Execute the stage."""
        raise NotImplementedError("Stage execute not implemented.")


class EarlyStopException(Exception):
    """Custom exception for early stopping."""

    pass


def check_early_stop(state: NodeState, raise_exception: bool = True) -> bool:
    """
    Check if early stopping is required.

    Args:
        state (NodeState): Node state
        raise_exception (bool): Whether to raise an exception for early stopping.

    Returns:
        bool: True if early stopping is required, False otherwise.

    Raises:
        EarlyStopException: Early stopping

    """
    if state.round is None:
        logger.info(state.addr, "Stopping Wokflow.")
        if raise_exception:
            raise EarlyStopException("Early stopping.")
        return True
    return False
