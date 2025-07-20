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

"""GRPC communication protocol."""

import random
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.message.heartbeat_command import HeartbeatCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.communication.protocols.exceptions import CommunicationError, ProtocolNotStartedError
from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.heartbeater import Heartbeater
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.communication.protocols.protobuff.server import ProtobuffServer
from p2pfl.settings import Settings
from p2pfl.utils.node_component import allow_no_addr_check


def running(func):
    """Ensure that the server is running before executing a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._server.is_running():
            raise ProtocolNotStartedError("The protocol has not been started.")
        return func(self, *args, **kwargs)

    return wrapper


class ProtobuffCommunicationProtocol(CommunicationProtocol):
    """
    Protobuff communication protocol.

    Args:
        addr: Address of the node.
        commands: Commands to add to the communication protocol.

    .. todo:: https://grpc.github.io/grpc/python/grpc_asyncio.html
    .. todo:: Decouple the heeartbeat command.

    """

    def __init__(
        self,
        commands: list[Command] | None = None,
    ) -> None:
        """Initialize the GRPC communication protocol."""
        # (addr) Super
        CommunicationProtocol.__init__(self)
        # Neighbors
        self._neighbors = Neighbors(self.bluid_client)
        # Gossip
        self._gossiper = Gossiper(self._neighbors, self.build_msg)
        # GRPC
        self._server = self.build_server(self._gossiper, self._neighbors, commands)
        # Hearbeat
        self._heartbeater = Heartbeater(self._neighbors, self.build_msg)
        # Commands
        self.add_command(HeartbeatCommand(self._heartbeater))
        if commands is None:
            commands = []
        self.add_command(commands)

    @allow_no_addr_check
    @abstractmethod
    def bluid_client(self, *args, **kwargs) -> ProtobuffClient:
        """Build client function."""
        pass

    @allow_no_addr_check
    @abstractmethod
    def build_server(self, *args, **kwargs) -> ProtobuffServer:
        """Build server function."""
        pass

    def set_addr(self, addr: str) -> str:
        """Set the addr of the node."""
        # Delegate on server
        addr = self._server.set_addr(addr)
        # Update components
        self._neighbors.set_addr(addr)
        self._gossiper.set_addr(addr)
        self._heartbeater.set_addr(addr)
        # Set on super
        return super().set_addr(addr)

    def start(self) -> None:
        """Start the GRPC communication protocol."""
        self._server.start()
        self._heartbeater.start()
        self._gossiper.start()

    @running
    def stop(self) -> None:
        """Stop the GRPC communication protocol."""
        self._heartbeater.stop()
        self._gossiper.stop()
        self._neighbors.clear_neighbors()
        self._server.stop()

    @allow_no_addr_check
    def add_command(self, cmds: Command | list[Command]) -> None:
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

    def build_msg(self, cmd: str, args: list[str] | None = None, round: int | None = None, direct: bool = False) -> node_pb2.RootMessage:
        """
        Build a RootMessage to send to the neighbors.

        Args:
            cmd: Command of the message.
            args: Arguments of the message.
            round: Round of the message.
            direct: If direct message.

        Returns:
            RootMessage to send.

        """
        if round is None:
            round = -1
        if args is None:
            args = []
        args = [str(a) for a in args]

        if direct:
            return node_pb2.RootMessage(
                source=self.addr,
                round=round,
                cmd=cmd,
                direct_message=node_pb2.DirectMessage(
                    args=args,
                ),
            )
        else:
            hs = hash(str(cmd) + str(args) + str(datetime.now()) + str(random.randint(0, 100000)))
            return node_pb2.RootMessage(
                source=self.addr,
                round=round,
                cmd=cmd,
                gossip_message=node_pb2.GossipMessage(
                    ttl=Settings.gossip.TTL,
                    hash=hs,
                    args=args,
                ),
            )

    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: list[str] | None = None,
        weight: int = 1,
    ) -> node_pb2.RootMessage:
        """
        Build a RootMessage with a Weights payload to send to the neighbors.

        Args:
            cmd: Command of the message.
            round: Round of the message.
            serialized_model: Serialized model to send.
            contributors: List of contributors.
            weight: Weight of the message (number of samples).

        Returns:
            RootMessage to send.

        """
        if contributors is None:
            contributors = []
        return node_pb2.RootMessage(
            source=self.addr,
            round=round,
            cmd=cmd,
            weights=node_pb2.Weights(
                weights=serialized_model,
                contributors=contributors,
                num_samples=weight,
            ),
        )

    @running
    def send(
        self,
        nei: str,
        msg: node_pb2.RootMessage,
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
        try:
            self._neighbors.get(nei).send(msg, raise_error=raise_error, disconnect_on_error=remove_on_error)
        except CommunicationError as e:
            if remove_on_error:
                self._neighbors.remove(nei)
            if raise_error:
                raise e

    @running
    def broadcast(self, msg: node_pb2.RootMessage, node_list: list[str] | None = None) -> None:
        """
        Broadcast a message to all neighbors.

        Args:
            msg: The message to broadcast.
            node_list: Optional node list.

        """
        neis = self._neighbors.get_all(only_direct=True)
        neis_clients = [nei[0] for nei in neis.values()]
        for nei in neis_clients:
            nei.send(msg)

    @running
    def get_neighbors(self, only_direct: bool = False) -> dict[str, Any]:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        return self._neighbors.get_all(only_direct)

    @running
    def wait_for_termination(self) -> None:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        self._server.wait_for_termination()

    @running
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
        if period is None:
            period = Settings.gossip.MODELS_PERIOD
        self._gossiper.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            period,
            create_connection,
        )
