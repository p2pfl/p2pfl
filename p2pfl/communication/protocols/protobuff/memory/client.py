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

"""Memory client."""

from p2pfl.communication.protocols.exceptions import CommunicationError, NeighborNotConnectedError
from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.communication.protocols.protobuff.memory.singleton_dict import SingletonDict
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.communication.protocols.protobuff.server import ProtobuffServer
from p2pfl.management.logger import logger


class MemoryClient(ProtobuffClient):
    """
    Implementation of the client side of an in-memory communication protocol.

    Args:
        self_addr: Address of the node.

    """

    def __init__(self, self_addr: str, nei_addr: str) -> None:
        """Initialize the GRPC client."""
        # Super
        super().__init__(self_addr, nei_addr)

        # In-memory
        self.stub: ProtobuffServer | None = None

    ####
    # Connection
    ####

    def is_connected(self) -> bool:
        """
        Check if a neighbor is connected.

        Returns:
            True if the neighbor is connected, False otherwise.

        """
        return self.stub is not None

    def connect(self, handshake_msg: bool = True) -> None:
        """Connect to a neighbor."""
        if self.is_connected():
            logger.debug(self.self_addr, f"Trying to connect to {self.nei_addr} an already connected neighbor.")
            return

        try:
            # Create stub
            self.stub = SingletonDict()[self.nei_addr]

            # Handshake
            if handshake_msg:
                res = self.stub.handshake(  # type: ignore
                    node_pb2.HandShakeRequest(addr=self.self_addr),
                    None,  # type: ignore
                )
                if res.error:
                    logger.info(self.self_addr, f"Cannot add a neighbor: {res.error}")
                    raise Exception(f"Cannot add a neighbor: {res.error}")

        except KeyError as e:
            logger.info(self.self_addr, f"Neighbor {self.nei_addr} not found.")
            raise NeighborNotConnectedError(f"Neighbor {self.nei_addr} not found.") from e
        except Exception as e:
            # Set stub and channel to None
            self.stub = None
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
                self.stub.disconnect(node_pb2.HandShakeRequest(addr=self.self_addr), None)  # type: ignore
        except Exception:
            pass
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
        # Check if connected (threadsafe)
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
        res = self.stub.send(msg, None)  # type: ignore

        # Log successful message sending
        if not res.error:
            self.log_successful_send(msg)

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
