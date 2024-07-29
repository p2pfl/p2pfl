#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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

from typing import Any, Callable, Dict, List, Optional, Union

from p2pfl.commands.command import Command
from p2pfl.commands.heartbeat_command import HeartbeatCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.communication.gossiper import Gossiper
from p2pfl.communication.grpc.address import AddressParser
from p2pfl.communication.grpc.grpc_client import GrpcClient
from p2pfl.communication.grpc.grpc_neighbors import GrpcNeighbors
from p2pfl.communication.grpc.grpc_server import GrpcServer
from p2pfl.communication.grpc.proto import node_pb2
from p2pfl.communication.heartbeater import Heartbeater
from p2pfl.settings import Settings


class GrpcCommunicationProtocol(CommunicationProtocol):
    """GRPC communication protocol."""

    def __init__(self, addr: str = "127.0.0.1", commands: Optional[List[Command]] = None) -> None:
        """Initialize the GRPC communication protocol."""
        # Parse IP address
        parsed_address = AddressParser(addr)
        self.addr = parsed_address.get_parsed_address()
        if not self.addr:  # esto rompe si esta mal? -> meterlo a test
            raise Exception(f"Address ({addr}) cannot be parsed.")

        """ De momento no aporta nada no?
        if parsed_address.unix_domain:
            # Using Unix domain socket
            self._socket_family = socket.AF_UNIX
        else:
            self._socket_family = socket.AF_INET
        """

        # Neighbors
        self._neighbors = GrpcNeighbors(self.addr)
        # GRPC Client
        self._client = GrpcClient(self.addr, self._neighbors)
        # Gossip
        self._gossiper = Gossiper(self.addr, self._client)
        # GRPC
        self._server = GrpcServer(self.addr, self._gossiper, self._neighbors, commands)
        # Hearbeat
        self._heartbeater = Heartbeater(self.addr, self._neighbors, self._client)
        # Commands
        self._server.add_command(HeartbeatCommand(self._heartbeater))
        if commands is None:
            commands = []
        self._server.add_command(commands)

    def get_address(self) -> str:
        """Get the address."""
        return self.addr

    def start(self) -> None:
        """Start the GRPC communication protocol."""
        self._server.start()
        self._heartbeater.start()
        self._gossiper.start()

    def stop(self) -> None:
        """Stop the GRPC communication protocol."""
        self._server.stop()
        self._heartbeater.stop()
        self._gossiper.stop()
        self._neighbors.clear_neighbors()

    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        """Add a command."""
        self._server.add_command(cmds)

    def connect(self, addr: str, non_direct: bool = False) -> bool:
        """Connect to a neighbor."""
        return self._neighbors.add(addr, non_direct=non_direct)

    def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        """Disconnect from a neighbor."""
        self._neighbors.remove(nei, disconnect_msg=disconnect_msg)

    def build_msg(self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None) -> Any:
        """Build a message."""
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
        """Build a weights message."""
        if contributors is None:
            contributors = []
        return self._client.build_weights(cmd, round, serialized_model, contributors, weight)

    def send(self, nei: str, msg: Union[node_pb2.Message, node_pb2.Weights]) -> None:
        """Send a message."""
        self._client.send(nei, msg)

    def broadcast(self, msg: node_pb2.Message, node_list: Optional[List[str]] = None) -> None:
        """Broadcast a message."""
        self._client.broadcast(msg, node_list)

    def get_neighbors(self, only_direct: bool = False) -> Dict[str, Any]:
        """Get neighbors."""
        return self._neighbors.get_all(only_direct)

    def wait_for_termination(self) -> None:
        """Wait for termination."""
        self._server.wait_for_termination()

    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], List[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], Any],
        period: Optional[float] = None,
        create_connection: bool = False,
    ) -> None:
        """Gossip weights."""
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
