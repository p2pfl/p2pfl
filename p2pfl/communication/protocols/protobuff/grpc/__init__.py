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

from typing import Optional

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.grpc.address import AddressParser
from p2pfl.communication.protocols.protobuff.grpc.client import GrpcClient
from p2pfl.communication.protocols.protobuff.grpc.server import GrpcServer
from p2pfl.communication.protocols.protobuff.protobuff_communication_protocol import ProtobuffCommunicationProtocol


class GrpcCommunicationProtocol(ProtobuffCommunicationProtocol):
    """GRPC communication protocol."""

    def __init__(self, addr: str = "127.0.0.1", commands: Optional[list[Command]] = None) -> None:
        """Initialize the GRPC communication protocol."""
        super().__init__(AddressParser(addr).get_parsed_address(), commands)

    def bluid_client(self, *args, **kwargs) -> GrpcClient:
        """Build client function."""
        return GrpcClient(*args, **kwargs)

    def build_server(self, *args, **kwargs) -> GrpcServer:
        """Build server function."""
        return GrpcServer(*args, **kwargs)
