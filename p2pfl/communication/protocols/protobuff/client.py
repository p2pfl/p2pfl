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

"""Protocol agnostic client."""

import threading
from abc import ABC, abstractmethod

from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.management.logger import logger


class ProtobuffClient(ABC):
    """
    Client interface.

    It is used as a interface to help to decoulple communication protocols based on clien-server architecutes.

    """

    def __init__(self, self_addr: str, nei_addr: str) -> None:
        """Initialize the GRPC client."""
        self.self_addr = self_addr
        self.nei_addr = nei_addr
        self._temporal_connection_uses = 0
        self._temporal_connection_lock = threading.Lock()

    ####
    # Connection
    ####

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if a neighbor is connected.

        Returns:
            True if the neighbor is connected, False otherwise.

        """
        pass

    @abstractmethod
    def connect(self, handshake_msg: bool = True) -> None:
        """Connect to a neighbor."""
        pass

    @abstractmethod
    def disconnect(self, disconnect_msg: bool = True) -> None:
        """Disconnect from a neighbor."""
        pass

    def has_temporal_connection(self) -> bool:
        """
        Check if the client has a temporal connection.

        Returns:
            True if the client has a temporal connection, False otherwise.

        """
        return self._temporal_connection_uses > 0

    ####
    # Message Sending
    ####

    def log_successful_send(self, msg: node_pb2.RootMessage) -> None:
        """
        Log a successful message sending.

        Args:
            msg: The message that was sent.

        """
        # Log
        package_type = "message" if not msg.HasField("weights") else "weights"
        package_size = len(msg.SerializeToString())
        round_num = msg.round if msg.round >= 0 else None  # Pass None for negative rounds, the logger will handle it

        logger.log_communication(
            self.self_addr,
            "sent",
            msg.cmd,
            self.nei_addr,
            package_type,
            package_size,
            round_num,
        )

    @abstractmethod
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
        pass
