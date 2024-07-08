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

import time
from typing import Any, Dict, Optional, Tuple

from p2pfl.communication.neighbors import Neighbors
from p2pfl.management.logger import logger
from .server_singleton import ServerSingleton

class InMemoryNeighbors(Neighbors):

    def refresh_or_add(self, addr: str, time: time) -> None:
        # Update if exists
        if addr in self.neis.keys():
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

    def connect(
        self, addr: str, non_direct: bool = False, handshake_msg: bool = True
    ) -> Tuple[Optional[None], Optional[None], float]:
        if non_direct:
            return self.__build_non_direct_neighbor(addr)
        else:
            return self.__build_direct_neighbor(addr, handshake_msg)

    def __build_direct_neighbor(self, addr: str, handshake_msg: bool) -> Optional[Tuple[str, str, float]]:
        try:
            # Get server
            server = ServerSingleton()[addr]

            # Simulate the handshake process
            if handshake_msg:
                response = server.handshake({"addr":self.self_addr})
                if response.get("error"):
                    logger.info(self.self_addr, f"Cannot add a neighbor: {response['error']}")
                    return None

            # Add neighbor
            self.neis[addr] = (None, server, time.time())
            
            return (None, server, time.time())

        except Exception as e:
            logger.info(self.self_addr, f"Crash while adding a neighbor: {e}")
            # Re-raise exception
            raise e

    def __build_non_direct_neighbor(self, _: str) -> Tuple[Optional[None], Optional[None], float]:
        return (None, None, time.time())

    def disconnect(self, addr: str, disconnect_msg: bool = True) -> None:
        try:
            # If the other node still connected, disconnect
            _, node_server, _ = self.get(addr)

            if disconnect_msg:
                if node_server is not None:
                    node_server.disconnect({"addr":self.self_addr})

        except Exception as e:
            logger.error(self.self_addr,
                         f"Error while disconnecting from {addr}: {e}")