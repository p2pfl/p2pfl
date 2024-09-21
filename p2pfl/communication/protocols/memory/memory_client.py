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

"""In-memory client."""

import random
import time
from typing import Dict, List, Optional, Union

from p2pfl.communication.protocols.client import Client
from p2pfl.communication.protocols.exceptions import CommunicationError, NeighborNotConnectedError
from p2pfl.communication.protocols.memory.memory_neighbors import InMemoryNeighbors
from p2pfl.communication.protocols.memory.server_singleton import ServerSingleton
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class InMemoryClient(Client):
    """
    Implementation of the client side of an in-memory communication protocol.

    Args:
        self_addr: Address of the node.
        neighbors: Neighbors of the node.

    """

    def __init__(self, self_addr: str, neighbors: InMemoryNeighbors) -> None:
        """Initialize the in-memory client."""
        self.__self_addr = self_addr
        self.__neighbors = neighbors

    def build_message(
        self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None
    ) -> Dict[str, Union[str, int, List[str]]]:
        """
        Build a message to send to the neighbors.

        Args:
            cmd: Command of the message.
            args: Arguments of the message.
            round: Round of the message.

        Returns:
            Message to send.

        """
        if round is None:
            round = -1
        if args is None:
            args = []
        hs = hash(str(cmd) + str(args) + str(time.time()) + str(random.randint(0, 100000)))
        args = [str(a) for a in args]
        return {
            "source": self.__self_addr,
            "ttl": Settings.TTL,
            "hash": hs,
            "cmd": cmd,
            "args": args,
            "round": round,
        }

    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[List[str]] = None,
        weight: int = 1,
    ) -> Dict[str, Union[str, int, bytes, List[str]]]:
        """
        Build a weight message to send to the neighbors.

        Args:
            cmd: Command of the message.
            round: Round of the message.
            serialized_model: Serialized model to send.
            contributors: List of contributors.
            weight: Weight of the message.

        """
        if contributors is None:
            contributors = []
        return {
            "source": self.__self_addr,
            "round": round,
            "weights": serialized_model,
            "contributors": contributors,
            "weight": weight,
            "cmd": cmd,
        }

    def send(
        self,
        nei: str,
        msg: Dict[str, Union[str, int, List[str], bytes]],
        create_connection: bool = False,
        raise_error: bool = False,
        remove_on_error: bool = True,
    ) -> None:
        """
        Send a message to a neighbor.

        Args:
            nei (string): Neighbor address.
            msg (node_pb2.Message or node_pb2.Weights): Message to send.
            create_connection (bool): Create a connection if not exists.
            raise_error (bool): Raise error if an error occurs.
            remove_on_error (bool): Remove neighbor if an error occurs.

        """
        try:
            # Get neighbor
            try:
                node_server = self.__neighbors.get(nei)[1]
            except KeyError as e:
                raise NeighborNotConnectedError(f"Neighbor {nei} not found.") from e

            # Check if direct connection
            if node_server is None and create_connection:
                node_server = ServerSingleton()[nei]

            # Simulate sending a message by invoking the neighbor's receive function
            if node_server is not None:
                res = node_server.send_weights(msg) if "weight" in msg else node_server.send_message(msg)
            else:
                raise NeighborNotConnectedError("Neighbor not directly connected (Stub not defined and create_connection is false).")
            if "error" in res:
                raise CommunicationError(f"Error while sending a message: {msg['cmd']!r}: {res['error']!r}")

        except Exception as e:
            logger.info(
                self.__self_addr,
                f"Cannot send message {msg['cmd']!r} to {nei}. Error: {str(e)}",
            )
            if remove_on_error:
                self.__neighbors.remove(nei, disconnect_msg=True)
            # Re-raise
            if raise_error:
                raise e

    def broadcast(
        self,
        msg: Dict[str, Union[str, int, List[str], bytes]],
        node_list: Optional[List[str]] = None,
    ) -> None:
        """
        Broadcast a message to all the neighbors.

        Args:
            msg: Message to send.
            node_list: List of neighbors to send the message. If None, send to all the neighbors.

        """
        # Node list
        nodes = node_list if node_list is not None else self.__neighbors.get_all(only_direct=True)

        # Send
        for n in nodes:
            self.send(n, msg)
