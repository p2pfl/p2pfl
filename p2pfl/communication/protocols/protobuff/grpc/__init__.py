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

"""GRPC implementation of the `CommunicationProtocol`."""

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.grpc.client import GrpcClient
from p2pfl.communication.protocols.protobuff.grpc.server import GrpcServer
from p2pfl.communication.protocols.protobuff.protobuff_communication_protocol import ProtobuffCommunicationProtocol
from p2pfl.utils.node_component import allow_no_addr_check


class GrpcCommunicationProtocol(ProtobuffCommunicationProtocol):
    """GRPC communication protocol."""

    def __init__(self, commands: list[Command] | None = None) -> None:
        """Initialize the GRPC communication protocol."""
        super().__init__(commands)

    @allow_no_addr_check
    def bluid_client(self, *args, **kwargs) -> GrpcClient:
        """Build client function."""
        return GrpcClient(*args, **kwargs)

    @allow_no_addr_check
    def build_server(self, *args, **kwargs) -> GrpcServer:
        """Build server function."""
        return GrpcServer(*args, **kwargs)
