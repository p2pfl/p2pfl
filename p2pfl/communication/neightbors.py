import threading
from typing import Dict

from p2pfl.management.logger import logger

"""
Duda, el channel sirve para algo?¿ Sólo para cerrar conexion no?

- mirar de hacer protected los atributos
"""


class Neighbors:
    def __init__(self, self_addr) -> None:
        self.self_addr = self_addr
        self.neis = {}
        self.neis_lock = threading.Lock()

    def connect(self, addr: str) -> any:
        raise NotImplementedError

    def disconnect(self, addr: str) -> None:
        raise NotImplementedError

    def add(self, addr: str, *args, **kargs) -> bool:
        # Log
        logger.info(self.self_addr, f"Adding {addr}")

        # Cannot add itself
        if addr == self.self_addr:
            logger.info(self.self_addr, "Cannot add itself")
            return False

        # Lock
        self.neis_lock.acquire()

        # Cannot add duplicates
        if self.exists(addr):
            logger.info(
                self.self_addr, f"Cannot add duplicates. {addr} already exists."
            )
            self.neis_lock.release()
            return False

        # Add
        try:
            self.neis[addr] = self.connect(addr, *args, **kargs)
        except Exception as e:
            logger.error(self.self_addr, f"Cannot add {addr}: {e}")
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
            addr (str): Address of the neighbor.
        """
        self.neis_lock.acquire()
        # Disconnect
        self.disconnect(addr, *args, **kargs)
        # Remove neighbor
        if addr in self.neis.keys():
            del self.neis[addr]
        self.neis_lock.release()

    def get(self, addr: str) -> any:
        return self.neis[addr]

    def get_all(self, only_direct: bool = False) -> Dict[str, any]:
        # Copy neighbors dict
        neis = self.neis.copy()
        # Filter
        if only_direct:
            return {k: v for k, v in neis.items() if v[1]}
        return neis

    def exists(self, addr: str) -> bool:
        return addr in self.neis.keys()

    def clear_neighbors(self) -> None:
        while len(self.neis) > 0:
            self.remove(list(self.neis.keys())[0])
