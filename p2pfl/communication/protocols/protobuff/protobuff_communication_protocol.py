#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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

import asyncio
import random
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.infrastructure import HeartbeatCommand
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
        self._neighbors: Neighbors = Neighbors(self.build_client)
        # Gossip
        self._gossiper = Gossiper(self._neighbors, self.build_msg)
        # GRPC
        self._server: ProtobuffServer = self.build_server(self._gossiper, self._neighbors, commands)
        # Hearbeat
        self._heartbeater: Heartbeater = Heartbeater(self._neighbors, self.build_msg)
        # Commands
        self.add_command(HeartbeatCommand(self._heartbeater))
        if commands is None:
            commands = []
        self.add_command(commands)

    @allow_no_addr_check
    @abstractmethod
    def build_client(self, *args, **kwargs) -> ProtobuffClient:
        """Build client function."""
        pass

    @allow_no_addr_check
    @abstractmethod
    def build_server(self, *args, **kwargs) -> ProtobuffServer:
        """Build server function."""
        pass

    def set_address(self, address: str) -> str:
        """Set the address of the node."""
        # Delegate on server
        address = self._server.set_address(address)
        # Update components
        self._neighbors.set_address(address)
        self._heartbeater.set_address(address)
        self._gossiper.set_address(address)
        # Set on super
        return super().set_address(address)

    async def start(self) -> None:
        """Start the GRPC communication protocol."""
        await self._server.start()
        await self._heartbeater.start()
        await self._gossiper.start()

    @running
    async def stop(self) -> None:
        """Stop the GRPC communication protocol."""
        # Run the stop methods of async tasks, awaiting their completion
        await self._heartbeater.stop()
        await self._gossiper.stop()

        # Cancel in-flight message processing tasks and clear buffers
        await self._server.cleanup_tasks()

        # Clear neighbors and stop the server
        await self._neighbors.clear_neighbors()
        await self._server.stop()

    @allow_no_addr_check
    def add_command(self, cmds: Command | list[Command]) -> None:
        """
        Add a command to the communication protocol.

        Args:
            cmds: The command to add.

        """
        self._server.add_command(cmds)

    @allow_no_addr_check
    def remove_command(self, cmd: str | Command) -> None:
        """
        Remove a command from the communication protocol.

        Args:
            cmd: The command to remove.

        """
        self._server.remove_command(cmd)

    @running
    async def connect(self, addr: str, non_direct: bool = False) -> bool:
        """
        Connect to a neighbor.

        Args:
            addr: The address to connect to.
            non_direct: The non direct flag.

        """
        return await self._neighbors.add(addr, non_direct=non_direct)

    @running
    async def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        """
        Disconnect from a neighbor.

        Args:
            nei: The neighbor to disconnect from.
            disconnect_msg: The disconnect message flag.

        """
        await self._neighbors.remove(nei, disconnect_msg=disconnect_msg)

    def build_msg(self, cmd: str, args: list[str] | None = None, round: int | None = None, direct: bool = False) -> node_pb2.RootMessage:
        """
        Build a Message to send to the neighbors.

        Args:
            cmd: Command of the message.
            args: Arguments of the message.
            round: Round of the message.
            direct: If True, builds a point-to-point message (no propagation, can return response).
                If False (default), builds a gossip message (propagates with TTL, fire-and-forget).

        Returns:
            Message to send.

        """
        if round is None:
            round = -1
        if args is None:
            args = []
        args = [str(a) for a in args]

        if direct:
            return node_pb2.RootMessage(
                source=self.address,
                round=round,
                cmd=cmd,
                direct_message=node_pb2.DirectMessage(
                    args=args,
                ),
            )
        else:
            hs = hash(str(cmd) + str(args) + str(datetime.now()) + str(random.randint(0, 100000)))
            return node_pb2.RootMessage(
                source=self.address,
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
            source=self.address,
            round=round,
            cmd=cmd,
            weights=node_pb2.Weights(
                weights=serialized_model,
                contributors=contributors,
                num_samples=weight,
            ),
        )

    @running
    async def send(
        self,
        nei: str,
        msg: node_pb2.RootMessage,
        raise_error: bool = False,
        remove_on_error: bool = True,
        temporal_connection: bool = False,
    ) -> str:
        """
        Send a message to a neighbor.

        Args:
            nei: The neighbor to send the message.
            msg: The message to send.
            raise_error: If raise error.
            remove_on_error: If remove on error.
            temporal_connection: If temporal connection.

        Returns:
            The response from the neighbor (for direct messages).

        """
        try:
            return await self._neighbors.get(nei).send(
                msg, temporal_connection=temporal_connection, raise_error=raise_error, disconnect_on_error=remove_on_error
            )
        except CommunicationError as e:
            if remove_on_error:
                await self._neighbors.remove(nei)
            if raise_error:
                raise e
            return ""

    @running
    async def broadcast(self, msg: node_pb2.RootMessage, node_list: list[str] | None = None) -> None:
        """
        Broadcast a message to all neighbors.

        Args:
            msg: The message to broadcast.
            node_list: Optional node list.

        """
        neis = self._neighbors.get_all(only_direct=True)
        neis_clients = [nei[0] for nei in neis.values()]

        await asyncio.gather(*(nei.send(msg) for nei in neis_clients))

    @running
    async def gossip(
        self,
        nei: str,
        msg: node_pb2.RootMessage,
        raise_error: bool = False,
        remove_on_error: bool = True,
        temporal_connection: bool = False,
    ) -> None:
        """
        Gossip a message to a neighbor.

        Args:
            nei: The neighbor to gossip the message.
            msg: The message to gossip.
            raise_error: If raise error.
            remove_on_error: If remove on error.
            temporal_connection: If temporal connection.

        """
        msg.gossip_message.ttl = Settings.gossip.TTL

        await self.send(nei, msg, raise_error=raise_error, remove_on_error=remove_on_error, temporal_connection=temporal_connection)

    @running
    async def broadcast_gossip(self, msg: node_pb2.RootMessage, node_list: list[str] | None = None) -> None:
        """
        Gossip a message to all neighbors.

        Args:
            msg: The message to gossip.
            node_list: Optional node list.

        """
        msg.gossip_message.ttl = Settings.gossip.TTL

        neis = self._neighbors.get_all(only_direct=True)
        neis_clients = [nei[0] for nei in neis.values()]

        await asyncio.gather(*(nei.send(msg, disconnect_on_error=False) for nei in neis_clients))

    @running
    def get_neighbors(self, only_direct: bool = False) -> dict[str, Any]:
        """
        Get the neighbors.

        Args:
            only_direct: The only direct flag.

        """
        return self._neighbors.get_all(only_direct)

    @running
    async def wait_for_termination(self) -> None:
        """Wait for the server to terminate."""
        await self._server.wait_for_termination()

    @running
    async def gossip_weights(
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
        await self._gossiper.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            period,
            create_connection,
        )
