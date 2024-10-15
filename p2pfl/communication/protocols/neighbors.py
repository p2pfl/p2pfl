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

"""Protocol agnostic neighbor management."""

import threading
from typing import Any, Dict

from p2pfl.management.logger import logger


class Neighbors:
    """
    Neighbor management class for agnostic communication protocol.

    Args:
        self_addr: Address of the node.

    """

    def __init__(self, self_addr) -> None:
        """Initialize the neighbor management class."""
        self.self_addr = self_addr
        self.neis: Dict[str, Any] = {}
        self.neis_lock = threading.Lock()

    def connect(self, addr: str) -> Any:
        """
        Connect to a neighbor.

        Args:
            addr: Address of the neighbor to connect.

        """
        raise NotImplementedError

    def disconnect(self, addr: str) -> None:
        """
        Disconnect from a neighbor.

        Args:
            addr: Address of the neighbor to disconnect.

        """
        raise NotImplementedError

    def refresh_or_add(self, addr: str, time: float) -> None:
        """
        Refresh or add a neighbor.

        Args:
            addr: Address of the neighbor.
            time: Time of the last heartbeat.

        """
        raise NotImplementedError

    def add(self, addr: str, *args, **kargs) -> bool:
        """
        Add a neighbor to the neighbors list.

        Args:
            addr: Address of the neighbor to add.
            args: Additional arguments for the connect method (focused reimplementation).
            kargs: Additional keyword arguments for the connect method (focused reimplementation).

        """
        # Cannot add itself
        if addr == self.self_addr:
            logger.info(self.self_addr, "❌ Cannot add itself")
            return False

        # Lock
        self.neis_lock.acquire()

        # Cannot add duplicates
        if self.exists(addr):
            logger.info(self.self_addr, f"❌ Cannot add duplicates. {addr} already exists.")
            self.neis_lock.release()
            return False

        # Add
        try:
            self.neis[addr] = self.connect(addr, *args, **kargs)
        except Exception as e:
            logger.error(self.self_addr, f"❌ Cannot add {addr}: {e}")
            self.neis_lock.release()
            return False

        # Release
        self.neis_lock.release()
        return True

    def remove(self, addr: str, *args, **kargs) -> None:
        """
        Remove a neighbor from the neighbors list.

        Be careful, this method does not close the connection, is agnostic to the connection state.

        Args:
            addr: Address of the neighbor to remove.
            args: Additional arguments for the disconnect method (focused reimplementation).
            kargs: Additional keyword arguments for the disconnect method (focused reimplementation).

        """
        self.neis_lock.acquire()
        # Disconnect
        self.disconnect(addr, *args, **kargs)
        # Remove neighbor
        if addr in self.neis:
            del self.neis[addr]
        self.neis_lock.release()

    def get(self, addr: str) -> Any:
        """
        Get a neighbor from the neighbors list.

        Args:
            addr: Address of the neighbor to get.

        """
        return self.neis[addr]

    def get_all(self, only_direct: bool = False) -> Dict[str, Any]:
        """
        Get all neighbors from the neighbors list.

        Args:
            only_direct: Flag to get only direct neighbors.

        """
        # Copy neighbors dict
        neis = self.neis.copy()
        # Filter
        if only_direct:
            return {k: v for k, v in neis.items() if v[1]}
        return neis

    def exists(self, addr: str) -> bool:
        """
        Check if a neighbor exists in the neighbors list.

        Args:
            addr: Address of the neighbor to check.

        """
        return addr in self.neis

    def clear_neighbors(self) -> None:
        """Clear all neighbors."""
        while len(self.neis) > 0:
            self.remove(list(self.neis.keys())[0])
