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

"""In-memory server."""

import threading
from typing import Optional

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.memory.singleton_dict import SingletonDict
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.server import ProtobuffServer
from p2pfl.management.logger import logger


class MemoryServer(ProtobuffServer):
    """
    Implementation of the server side logic of Memory communication protocol.

    Args:
        addr: Address of the server.
        gossiper: Gossiper instance.
        neighbors: Neighbors instance.
        commands: List of commands to be executed by the server.

    """

    def __init__(
        self,
        addr: str,
        gossiper: Gossiper,
        neighbors: Neighbors,
        commands: Optional[list[Command]] = None,
    ) -> None:
        """Initialize the in-memory server."""
        # Super
        super().__init__(addr, gossiper, neighbors, commands)

        # Server
        self.__singleton_dict = SingletonDict()
        self.__terminated = threading.Event()

    ####
    # Management
    ####

    def start(self, wait: bool = False) -> None:
        """
        Start the in-memory server.

        Args:
            wait: If True, wait for termination.

        """
        if self.__singleton_dict is None:
            raise Exception("ServerSingleton instance not created")
        self.__singleton_dict[self.addr] = self
        self.__terminated.set()
        logger.info(self.addr, f"InMemoryServer started at {self.addr}")

    def stop(self) -> None:
        """Stop the in-memory server."""
        del self.__singleton_dict[self.addr]
        self.__terminated.clear()
        logger.info(self.addr, f"InMemoryServer stopped at {self.addr}")

    def wait_for_termination(self) -> None:
        """Wait for termination."""
        self.__terminated.wait()

    def is_running(self) -> bool:
        """
        Check if the server is running.

        Returns:
            True if the server is running, False otherwise.

        """
        return self.__terminated.is_set()
