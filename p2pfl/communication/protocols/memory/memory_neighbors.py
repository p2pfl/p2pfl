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

"""In-memory neighbors."""

import time
from typing import Optional, Tuple

from p2pfl.communication.protocols.memory.server_singleton import ServerSingleton
from p2pfl.communication.protocols.neighbors import Neighbors
from p2pfl.management.logger import logger


class InMemoryNeighbors(Neighbors):
    """Implementation of the neighbors side of an in-memory communication protocol."""

    def refresh_or_add(self, addr: str, time: float) -> None:
        """
        Refresh or add a neighbor.

        Args:
            addr: Address of the neighbor.
            time: Time of the last heartbeat.

        """
        # Update if exists
        if addr in self.neis:
            with self.neis_lock:
                # Update time
                self.neis[addr] = (
                    self.neis[addr][0],
                    self.neis[addr][1],
                    time,
                )
        else:
            # Add
            self.add(addr, non_direct=True)

    def connect(self, addr: str, non_direct: bool = False, handshake_msg: bool = True) -> Tuple[None, Optional[str], float]:
        """
        Connect to a neighbor.

        Args:
            addr: Address of the neighbor to connect.
            non_direct: If the connection is direct or not.
            handshake_msg: If a handshake message is needed.

        """
        if non_direct:
            logger.debug(self.self_addr, f"🔍 Found node {addr}")
            return self.__build_non_direct_neighbor(addr)
        else:
            logger.info(self.self_addr, f"🤝 Adding {addr}")
            return self.__build_direct_neighbor(addr, handshake_msg)

    def __build_direct_neighbor(self, addr: str, handshake_msg: bool) -> Tuple[None, Optional[str], float]:
        try:
            # Get server
            server = ServerSingleton()[addr]

            # Simulate the handshake process
            if handshake_msg:
                response = server.handshake({"addr": self.self_addr})
                if response.get("error"):
                    logger.info(self.self_addr, f"Cannot add a neighbor: {response['error']}")
                    raise Exception(f"Cannot add a neighbor: {response['error']}")

            return (None, server, time.time())

        except Exception as e:
            logger.info(self.self_addr, f"Crash while adding a neighbor: {e}")
            # Re-raise exception
            raise e

    def __build_non_direct_neighbor(self, _: str) -> Tuple[None, None, float]:
        return (None, None, time.time())

    def disconnect(self, addr: str, disconnect_msg: bool = True) -> None:
        """
        Disconnect from a neighbor.

        Args:
            addr: Address of the neighbor to disconnect.
            disconnect_msg: If a disconnect message is needed.

        """
        try:
            # If the other node still connected, disconnect
            _, node_server, _ = self.get(addr)

            if disconnect_msg and node_server is not None:
                node_server.disconnect({"addr": self.self_addr})

        except Exception as e:
            logger.error(self.self_addr, f"Error while disconnecting from {addr}: {e}")
