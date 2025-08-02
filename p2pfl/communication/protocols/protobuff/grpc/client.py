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

from os.path import isfile

import grpc

from p2pfl.communication.protocols.exceptions import CommunicationError, NeighborNotConnectedError
from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.communication.protocols.protobuff.proto import node_pb2, node_pb2_grpc
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class GrpcClient(ProtobuffClient):
    """
    Implementation of the client side (i.e. who initiates the communication) of the GRPC communication protocol.

    Args:
        self_addr: Address of the node.

    """

    def __init__(self, self_addr: str, nei_addr: str) -> None:
        """Initialize the GRPC client."""
        # Super
        super().__init__(self_addr, nei_addr)

        # GRPC
        self.channel: grpc.Channel | None = None
        self.stub: node_pb2_grpc.NodeServicesStub | None = None

    ####
    # Connection
    ####

    def is_connected(self) -> bool:
        """
        Check if a neighbor is connected.

        Returns:
            True if the neighbor is connected, False otherwise.

        """
        return (self.stub is not None) and (self.channel is not None)

    def connect(self, handshake_msg: bool = True) -> None:
        """Connect to a neighbor."""
        # Check if connected
        if self.is_connected():
            logger.debug(self.self_addr, f"Trying to connect to {self.nei_addr} an already connected neighbor.")
            return

        try:
            # Create channel (ssl or not)
            if Settings.ssl.USE_SSL and isfile(Settings.ssl.SERVER_CRT):
                with (
                    open(Settings.ssl.CLIENT_KEY) as key_file,
                    open(Settings.ssl.CLIENT_CRT) as crt_file,
                    open(Settings.ssl.CA_CRT) as ca_file,
                ):
                    private_key = key_file.read().encode()
                    certificate_chain = crt_file.read().encode()
                    root_certificates = ca_file.read().encode()
                creds = grpc.ssl_channel_credentials(
                    root_certificates=root_certificates, private_key=private_key, certificate_chain=certificate_chain
                )
                self.channel = grpc.secure_channel(self.nei_addr, creds)
            else:
                self.channel = grpc.insecure_channel(self.nei_addr)

            # Create stub
            self.stub = node_pb2_grpc.NodeServicesStub(self.channel)
            if not self.stub:
                raise Exception(f"Cannot create a stub for {self.nei_addr}")

            # Handshake
            if handshake_msg:
                res = self.stub.handshake(
                    node_pb2.HandShakeRequest(addr=self.self_addr),
                    timeout=Settings.general.GRPC_TIMEOUT,
                )
                if res.error:
                    logger.info(self.self_addr, f"Cannot add a neighbor: {res.error}")
                    self.channel.close()
                    raise Exception(f"Cannot add a neighbor: {res.error}")

        except Exception as e:
            # Set stub and channel to None
            self.stub = None
            self.channel = None
            # Re-raise exception
            raise e

    def disconnect(self, disconnect_msg: bool = True) -> None:
        """Disconnect from a neighbor."""
        # Check if connected
        if not self.is_connected():
            logger.debug(self.self_addr, f"Trying to disconnect from {self.nei_addr} a non-connected neighbor.")
            return

        # Disconnect
        try:
            # If the other node still connected, disconnect
            if disconnect_msg:
                self.stub.disconnect(node_pb2.HandShakeRequest(addr=self.self_addr))  # type: ignore
                # Close channel
                self.channel.close()  # type: ignore
        except Exception:
            pass
        self.channel = None
        self.stub = None

    ####
    # Message Sending
    ####

    def send(
        self,
        msg: node_pb2.RootMessage,
        temporal_connection: bool = False,
        raise_error: bool = False,
        disconnect_on_error: bool = True,
    ) -> str:
        """
        Send a message to the neighbor.

        Args:
            msg: Message to send.
            temporal_connection: If the connection isn't stablished and a temporal connection is needed for sending the message.
            raise_error: Raise error if an error occurs.
            disconnect_on_error: Disconnect if an error occurs.

        """
        # Check if connected
        if not self.is_connected():
            if temporal_connection:
                with self._temporal_connection_lock:
                    self._temporal_connection_uses += 1
                    if self._temporal_connection_uses == 1:
                        logger.debug(
                            self.self_addr, f"💔 Neighbor {self.nei_addr} not connected. Trying to send message with temporal connection"
                        )
                        self.connect(handshake_msg=False)
            else:
                raise NeighborNotConnectedError(f"Neighbor {self.nei_addr} not connected.")

        # Send
        try:
            res = self.stub.send(msg, timeout=Settings.general.GRPC_TIMEOUT)  # type: ignore

            # Log successful message sending
            if not res.error:
                self.log_successful_send(msg)
            else:
                raise CommunicationError(f"Error while sending a message: {msg.cmd}: {res.error}")

        except Exception as e:
            # Unexpected error
            logger.info(
                self.self_addr,
                f"Cannot send message {msg.cmd} to {self.nei_addr}. Error: {e}",
            )
            if temporal_connection:
                with self._temporal_connection_lock:
                    self._temporal_connection_uses -= 1
                    if self._temporal_connection_uses == 0:
                        self.disconnect(disconnect_msg=False)
            if raise_error:
                raise e
            else:
                return ""

        if res.error:
            logger.info(
                self.self_addr,
                f"Cannot send message {msg.cmd} to {self.nei_addr}. Error: {res.error}",
            )

        # Disconnect
        if temporal_connection:
            with self._temporal_connection_lock:
                self._temporal_connection_uses -= 1
                if self._temporal_connection_uses == 0:
                    self.disconnect(disconnect_msg=False)
        elif disconnect_on_error and res.error:
            self.disconnect(disconnect_msg=True)

        # Raise
        if res.error and raise_error:
            raise CommunicationError(f"Error while sending a message: {msg.cmd}: {res.error}")

        return res.response
