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

"""P2PFL Callbacks for federated learning."""

from abc import ABC, abstractmethod
from typing import Any


class P2PFLCallback(ABC):
    """
    A callback for the P2PFL learning process.

    The callback can generate additional information that can be used by the aggregator to affect the learning process.

    In order to affect the learning process, the callback must be registered with the `CallbackFactory`.
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        self.additional_info: Any = None

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Get the name of the callback."""
        pass

    def get_info(self) -> Any:
        """Get the additional information stored in the callback."""
        return self.additional_info

    def set_info(self, additional_info: Any) -> None:
        """Set the additional information stored in the callback."""
        self.additional_info = additional_info
