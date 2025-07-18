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

"""Communication protocol."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from p2pfl.communication.commands.command import Command
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check


class CommunicationProtocol(ABC, NodeComponent):
    """
    Communication protocol interface.

    Args:
        addr: The address.
        commands: The commands.

    """

    def __init__(self, commands: list[Command] | None = None, *args, **kwargs) -> None:
        """Initialize the communication protocol."""
        # (addr) Super
        NodeComponent.__init__(self)

    @abstractmethod
    def start(self) -> None:
        """Start the communication protocol."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the communication protocol."""
        pass

    @allow_no_addr_check
    @abstractmethod
    def add_command(self, cmds: Command | list[Command]) -> None:
        """
        Add a command to the communication protocol.

        Args:
            cmds: The command to add.

        """
        pass

    @abstractmethod
    def build_msg(self, cmd: str, args: list[str] | None = None, round: int | None = None) -> Any:
        """
        Build a message.

        Args:
            cmd: The message.
            args: The arguments.
            round: The round.

        """
        pass

    @abstractmethod
    def build_weights(self, cmd: str, round: int, serialized_model: bytes, contributors: list[str] | None = None, weight: int = 1) -> Any:
        """
        Build weights.

        Args:
            cmd: The command.
            round: The round.
            serialized_model: The serialized model.
            contributors: The model contributors.
            weight: The weight of the model (amount of samples used).

        """
        pass

    @abstractmethod
    def send(
        self,
        nei: str,
        msg: Any,
        raise_error: bool = False,
        remove_on_error: bool = True,
    ) -> None:
        """
        Send a message to a neighbor.

        Args:
            nei: The neighbor to send the message.
            msg: The message to send.
            raise_error: If raise error.
            remove_on_error: If remove on error.

        """
        pass

    @abstractmethod
    def broadcast(self, msg: Any, node_list: list[str] | None = None) -> None:
        """
        Broadcast a message to all neighbors.

        Args:
            msg: The message to broadcast.
            node_list: Optional node list.

        """
        pass

    @abstractmethod
    def connect(self, addr: str, non_direct: bool = False) -> bool:
        """
        Connect to a neighbor.

        Args:
            addr: The address to connect to.
            non_direct: The non direct flag.

        """
        pass

    @abstractmethod
    def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        """
        Disconnect from a neighbor.

        Args:
            nei: The neighbor to disconnect from.
            disconnect_msg: The disconnect message flag.

        """
        pass

    @abstractmethod
    def get_neighbors(self, only_direct: bool = False) -> dict[str, Any]:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        pass

    def get_address(self) -> str:
        """
        Get the address.

        Returns:
            The address.

        """
        return self.addr

    @abstractmethod
    def wait_for_termination(self) -> None:
        """Wait for termination."""
        pass

    @abstractmethod
    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], list[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], tuple[Any, str, int, list[str]]],
        period: float | None = None,
        create_connection: bool = False,
    ) -> None:
        """
        Gossip model weights.

        Args:
            early_stopping_fn: The early stopping function.
            get_candidates_fn: The get candidates function.
            status_fn: The status function.
            model_fn: The model function.
            period: The period.
            create_connection: The create connection flag.

        """
        pass
