#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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

"""In-memory communication protocol."""

import random
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.message.heartbeat_command import HeartbeatCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.exceptions import ProtocolNotStartedError
from p2pfl.communication.protocols.gossiper import Gossiper
from p2pfl.communication.protocols.heartbeater import Heartbeater
from p2pfl.communication.protocols.memory.memory_client import InMemoryClient
from p2pfl.communication.protocols.memory.memory_neighbors import InMemoryNeighbors
from p2pfl.communication.protocols.memory.memory_server import InMemoryServer
from p2pfl.settings import Settings

#
# Need to simplify this protocol, just wrapp the grpc one to avoid modifying InMemoryCommunicationProtocol every time
#


def running(func):
    """Ensure that the server is running before executing a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._server.is_running():
            raise ProtocolNotStartedError("The protocol has not been started.")
        return func(self, *args, **kwargs)

    return wrapper


class InMemoryCommunicationProtocol(CommunicationProtocol):
    """
    In-memory communication protocol.

    Args:
        addr: Address of the node.
        commands: Commands to add to the communication protocol.

    .. todo:: Remove this copy-paste code and use a in-memory wrapper for the grpc communication protocol.

    """

    def __init__(self, addr: Optional[str] = None, commands: Optional[List[Command]] = None) -> None:
        """Initialize the in-memory communication protocol."""
        # Address
        if addr:
            self.addr = addr
        else:
            self.addr = f"node-{random.randint(0,99999999)}"
        # Neighbors
        self._neighbors = InMemoryNeighbors(self.addr)
        # Client
        self._client = InMemoryClient(self.addr, self._neighbors)
        # Gossip
        self._gossiper = Gossiper(self.addr, self._client)
        # Server
        self._server = InMemoryServer(self.addr, self._gossiper, self._neighbors, commands)
        # Hearbeat
        self._heartbeater = Heartbeater(self.addr, self._neighbors, self._client)
        # Commands
        self._server.add_command(HeartbeatCommand(self._heartbeater))
        if commands is None:
            commands = []
        self._server.add_command(commands)

    def get_address(self) -> str:
        """
        Get the address.

        Returns:
            The address.

        """
        return self.addr

    def start(self) -> None:
        """Start the communication protocol."""
        self._server.start()
        self._heartbeater.start()
        self._gossiper.start()

    @running
    def stop(self) -> None:
        """Stop the communication protocol."""
        self._server.stop()
        self._heartbeater.stop()
        self._gossiper.stop()
        self._neighbors.clear_neighbors()

    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        """
        Add a command to the communication protocol.

        Args:
            cmds: The command to add.

        """
        self._server.add_command(cmds)

    @running
    def connect(self, addr: str, non_direct: bool = False) -> bool:
        """
        Connect to a neighbor.

        Args:
            addr: The address to connect to.
            non_direct: The non direct flag.

        """
        return self._neighbors.add(addr, non_direct=non_direct)

    @running
    def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        """
        Disconnect from a neighbor.

        Args:
            nei: The neighbor to disconnect from.
            disconnect_msg: The disconnect message flag.

        """
        self._neighbors.remove(nei, disconnect_msg=disconnect_msg)

    def build_msg(self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None) -> Any:
        """
        Build a message.

        Args:
            cmd: The message.
            args: The arguments.
            round: The round.

        """
        if args is None:
            args = []
        return self._client.build_message(cmd, args, round)

    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[List[str]] = None,
        weight: int = 1,
    ) -> Any:
        """
        Build weights.

        Args:
            cmd: The command.
            round: The round.
            serialized_model: The serialized model.
            contributors: The model contributors.
            weight: The weight of the model (amount of samples used).

        """
        if contributors is None:
            contributors = []
        return self._client.build_weights(cmd, round, serialized_model, contributors, weight)

    @running
    def send(
        self,
        nei: str,
        msg: Union[
            Dict[str, Union[str, int, List[str], bytes]],
            Dict[str, Union[str, int, bytes, List[str]]],
        ],
        raise_error: bool = False,
        remove_on_error: bool = True,
    ) -> None:
        """
        Send a message to a neighbor.

        Args:
            nei: The neighbor to send the message.
            msg: The message to sen
            raise_error: If raise error.
            remove_on_error: If remove on error.d.

        """
        self._client.send(nei, msg, raise_error=raise_error, remove_on_error=remove_on_error)

    @running
    def broadcast(
        self,
        msg: Dict[str, Union[str, int, List[str], bytes]],
        node_list: Optional[List[str]] = None,
    ) -> None:
        """
        Broadcast a message to all neighbors.

        Args:
            msg: The message to broadcast.
            node_list: Optional node list.

        """
        self._client.broadcast(msg, node_list)

    @running
    def get_neighbors(self, only_direct: bool = False) -> Dict[str, Any]:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        return self._neighbors.get_all(only_direct)

    @running
    def wait_for_termination(self) -> None:
        """Wait for termination."""
        self._server.wait_for_termination()

    @running
    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], List[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], Any],
        period: Optional[float] = None,
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
        if period is None:
            period = Settings.GOSSIP_MODELS_PERIOD
        self._gossiper.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            period,
            create_connection,
        )
