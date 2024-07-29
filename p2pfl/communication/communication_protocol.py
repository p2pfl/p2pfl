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

"""Communication protocol."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from p2pfl.commands.command import Command


class CommunicationProtocol(ABC):
    """Communication protocol interface."""

    @abstractmethod
    def __init__(self, address: str) -> None:
        """Initialize the communication protocol."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the communication protocol."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the communication protocol."""
        pass

    @abstractmethod
    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        """Add a command to the communication protocol."""
        pass

    @abstractmethod
    def build_msg(self, msg: str, args: Optional[List[str]] = None, round: Optional[int] = None) -> str:
        """Build a message."""
        pass

    @abstractmethod
    def build_weights(
        self, cmd: str, round: int, serialized_model: bytes, contributors: Optional[List[str]] = None, weight: int = 1
    ) -> Any:
        """Build weights."""
        pass

    @abstractmethod
    def send(self, nei: str, message: Any) -> None:
        """Send a message to a neighbor."""
        pass

    @abstractmethod
    def broadcast(self, message: Any) -> None:
        """Broadcast a message to all neighbors."""
        pass

    @abstractmethod
    def connect(self, addr: str, non_direct: bool = False) -> bool:
        """Connect to a neighbor."""
        pass

    @abstractmethod
    def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        """Disconnect from a neighbor."""
        pass

    @abstractmethod
    def get_neighbors(self, only_direct: bool = False) -> Dict[str, Any]:
        """Get the neighbors."""
        pass

    @abstractmethod
    def get_address(self) -> str:
        """Get the address."""
        pass

    @abstractmethod
    def wait_for_termination(self) -> None:
        """Wait for termination."""
        pass

    @abstractmethod
    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], List[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], Any],
        period: Optional[float] = None,
        create_connection: bool = False,
    ) -> None:
        """Gossip model weights."""
        pass
