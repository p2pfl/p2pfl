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

"""GRPC client."""

import random
from datetime import datetime
from os.path import isfile
from typing import List, Optional

import grpc
from p2pfl.communication.protocols.client import Client
from p2pfl.communication.protocols.exceptions import CommunicationError, NeighborNotConnectedError
from p2pfl.communication.protocols.grpc.grpc_neighbors import GrpcNeighbors
from p2pfl.communication.protocols.grpc.proto import node_pb2, node_pb2_grpc
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

    def build_message(self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None) -> node_pb2.RootMessage:
        """
        Build a RootMessage to send to the neighbors.

        Args:
            cmd: Command of the message.
            args: Arguments of the message.
            round: Round of the message.

        Returns:
            RootMessage to send.

        """
        if round is None:
            round = -1
        if args is None:
            args = []
        hs = hash(str(cmd) + str(args) + str(datetime.now()) + str(random.randint(0, 100000)))
        args = [str(a) for a in args]

        return node_pb2.RootMessage(
            source=self.__self_addr,
            round=round,
            cmd=cmd,
            message=node_pb2.Message(
                ttl=Settings.TTL,
                hash=hs,
                args=args,
            ),
        )

    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[List[str]] = None,
        weight: int = 1,
    ) -> node_pb2.RootMessage:
        """
        Build a RootMessage with a Weights payload to send to the neighbors.

        Args:
            cmd: Command of the message.
            round: Round of the message.
            serialized_model: Serialized model to send.
            contributors: List of contributors.
            weight: Weight of the message (number of samples).

        Returns:
            RootMessage to send.

        """
        if contributors is None:
            contributors = []
        return node_pb2.RootMessage(
            source=self.__self_addr,
            round=round,
            cmd=cmd,
            weights=node_pb2.Weights(
                weights=serialized_model,
                contributors=contributors,
                num_samples=weight,
            ),
        )

    ####
    # Message Sending
    ####

    def send(
        self,
        nei: str,
        msg: node_pb2.RootMessage,
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
        channel = None
        try:
            # Get neighbor
            try:
                node_stub = self.__neighbors.get(nei)[1]
            except KeyError as e:
                raise NeighborNotConnectedError(f"Neighbor {nei} not found.") from e

            # Check if direct connection
            if node_stub is None and create_connection:
                if Settings.USE_SSL and isfile(Settings.SERVER_CRT):
                    with open(Settings.CLIENT_KEY) as key_file, open(Settings.CLIENT_CRT) as crt_file, open(Settings.CA_CRT) as ca_file:
                        private_key = key_file.read().encode()
                        certificate_chain = crt_file.read().encode()
                        root_certificates = ca_file.read().encode()
                    creds = grpc.ssl_channel_credentials(
                        root_certificates=root_certificates,
                        private_key=private_key,
                        certificate_chain=certificate_chain,
                    )
                    channel = grpc.secure_channel(nei, creds)
                else:
                    channel = grpc.insecure_channel(nei)
                node_stub = node_pb2_grpc.NodeServicesStub(channel)

            # Send
            if node_stub is not None:
                # Send message
                res = node_stub.send(msg, timeout=Settings.GRPC_TIMEOUT)
            else:
                raise NeighborNotConnectedError("Neighbor not directly connected (Stub not defined and create_connection is false).")
            if res.error:
                raise CommunicationError(f"Error while sending a message: {msg.cmd}: {res.error}")
        except Exception as e:
            # Remove neighbor
            logger.info(
                self.__self_addr,
                f"Cannot send message {msg.cmd} to {nei}. Error: {str(e)}",
            )
            if remove_on_error:
                self.__neighbors.remove(nei, disconnect_msg=True)
            # Re-raise
            if raise_error:
                raise e

        finally:
            if channel is not None:
                channel.close()

    def broadcast(self, msg: node_pb2.RootMessage, node_list: Optional[List[str]] = None) -> None:
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
