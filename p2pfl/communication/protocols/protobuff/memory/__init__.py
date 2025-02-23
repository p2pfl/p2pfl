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

"""In-Memory implementation of the `CommunicationProtocol`."""

from typing import Optional

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.memory.client import MemoryClient
from p2pfl.communication.protocols.protobuff.memory.server import MemoryServer
from p2pfl.communication.protocols.protobuff.protobuff_communication_protocol import ProtobuffCommunicationProtocol
from p2pfl.utils.singleton import SingletonMeta


# Singleton Address counter
class AddressCounter(metaclass=SingletonMeta):
    """Singleton address counter."""

    def __init__(self) -> None:
        """Initialize the address counter."""
        self.__counter = 0

    def get(self) -> str:
        """Get the address."""
        self.__counter += 1
        return f"node-{self.__counter}"


class MemoryCommunicationProtocol(ProtobuffCommunicationProtocol):
    """GRPC communication protocol."""

    def __init__(self, addr: str = "", commands: Optional[list[Command]] = None) -> None:
        """Initialize the GRPC communication protocol."""
        # Address
        if addr == "":
            addr = AddressCounter().get()
        # Super
        super().__init__(addr, commands)

    def bluid_client(self, *args, **kwargs) -> MemoryClient:
        """Build client function."""
        return MemoryClient(*args, **kwargs)

    def build_server(self, *args, **kwargs) -> MemoryServer:
        """Build server function."""
        return MemoryServer(*args, **kwargs)
