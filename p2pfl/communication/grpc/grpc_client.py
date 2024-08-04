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

"""GRPC client."""

import random
from datetime import datetime
from typing import List, Optional, Union

import grpc

from p2pfl.communication.client import Client
from p2pfl.communication.exceptions import NeighborNotConnectedError
from p2pfl.communication.grpc.grpc_neighbors import GrpcNeighbors
from p2pfl.communication.grpc.proto import node_pb2, node_pb2_grpc
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class GrpcClient(Client):
    """
    Implementation of the client side (i.e. who initiates the communication) of the GRPC communication protocol.

    Args:
        self_addr: Address of the node.
        neighbors: Neighbors of the node.

    """

    def __init__(self, self_addr: str, neighbors: GrpcNeighbors) -> None:
        """Initialize the GRPC client."""
        self.__self_addr = self_addr
        self.__neighbors = neighbors

    ####
    # Message Building
    ####

    def build_message(
        self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None
    ) -> node_pb2.Message:
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
        hs = hash(str(cmd) + str(args) + str(datetime.now()) + str(random.randint(0, 100000)))
        args = [str(a) for a in args]

        return node_pb2.Message(
            source=self.__self_addr,
            ttl=Settings.TTL,
            hash=hs,
            cmd=cmd,
            args=args,
            round=round,
        )

    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[List[str]] = None,
        weight: int = 1,
    ) -> node_pb2.Weights:
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
        return node_pb2.Weights(
            source=self.__self_addr,
            round=round,
            weights=serialized_model,
            contributors=contributors,
            weight=weight,
            cmd=cmd,
        )

    ####
    # Message Sending
    ####

    def send(
        self,
        nei: str,
        msg: Union[node_pb2.Message, node_pb2.Weights],
        create_connection: bool = False,
    ) -> None:
        """
        Send a message to a neighbor.

        Args:
            nei (string): Neighbor address.
            msg (node_pb2.Message or node_pb2.Weights): Message to send.
            create_connection (bool): Create a connection if not exists.

        """
        channel = None
        try:
            # Get neighbor
            try:
                node_stub = self.__neighbors.get(nei)[1]
            except KeyError as e:
                raise NeighborNotConnectedError(f"Neighbor {nei} not found.") from e

            # Check if direct connection
            if node_stub is None and create_connection:
                channel = grpc.insecure_channel(nei)
                node_stub = node_pb2_grpc.NodeServicesStub(channel)

            # Send
            if node_stub is not None:
                # Send message
                if isinstance(msg, node_pb2.Message):
                    res = node_stub.send_message(msg, timeout=Settings.GRPC_TIMEOUT)
                elif isinstance(msg, node_pb2.Weights):
                    res = node_stub.send_weights(msg, timeout=Settings.GRPC_TIMEOUT)
                else:
                    raise TypeError("Message type not supported.")
            else:
                raise NeighborNotConnectedError(
                    "Neighbor not directly connected (Stub not defined and create_connection is false)."
                )
            if res.error:
                if isinstance(msg, node_pb2.Message):
                    logger.error(
                        self.__self_addr,
                        f"Error while sending a message: {msg.cmd} {msg.args}: {res.error}",
                    )
                elif isinstance(msg, node_pb2.Weights):
                    logger.error(
                        self.__self_addr,
                        f"Error while sending a message: {msg.cmd}: {res.error}",
                    )
                else:
                    raise TypeError("Message type not supported.")
                self.__neighbors.remove(nei, disconnect_msg=True)
        except Exception as e:
            # Remove neighbor
            logger.info(
                self.__self_addr,
                f"Cannot send message {msg.cmd} to {nei}. Error: {str(e)}",
            )
            self.__neighbors.remove(nei)

        finally:
            if channel is not None:
                channel.close()

    def broadcast(self, msg: node_pb2.Message, node_list: Optional[List[str]] = None) -> None:
        """
        Broadcast a message to all the neighbors.

        Args:
            msg: Message to send.
            node_list: List of neighbors to send the message. If None, send to all the neighbors.

        """
        # Node list
        nodes = node_list if node_list is not None else self.__neighbors.get_all(only_direct=True).keys()

        # Send
        for n in nodes:
            self.send(n, msg)
