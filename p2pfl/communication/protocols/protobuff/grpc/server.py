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

"""GRPC server."""

from concurrent import futures
from os.path import isfile

import grpc

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.grpc.address import AddressParser
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2_grpc
from p2pfl.communication.protocols.protobuff.server import ProtobuffServer
from p2pfl.settings import Settings


class GrpcServer(ProtobuffServer):
    """
    Implementation of the server side logic of GRPC communication protocol.

    Args:
        addr: Address of the server.
        gossiper: Gossiper instance.
        neighbors: Neighbors instance.
        commands: List of commands to be executed by the server.

    """

    def __init__(
        self,
        gossiper: Gossiper,
        neighbors: Neighbors,
        commands: list[Command] | None = None,
    ) -> None:
        """Initialize the GRPC server."""
        # Super
        super().__init__(gossiper, neighbors, commands)

        # Server
        maxMsgLength = 1024 * 1024 * 1024
        self.__server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=2),
            options=[
                ("grpc.max_send_message_length", maxMsgLength),
                ("grpc.max_receive_message_length", maxMsgLength),
            ],
        )
        self.__server_started = False

    def set_addr(self, addr: str) -> str:
        """Parse and set the addr of the node."""
        return super().set_addr(AddressParser(addr).get_parsed_address())

    ####
    # Management
    ####

    def start(self, wait: bool = False) -> None:
        """
        Start the GRPC server.

        Args:
            wait: If True, wait for termination.

        """
        # Server
        node_pb2_grpc.add_NodeServicesServicer_to_server(self, self.__server)
        try:
            if Settings.ssl.USE_SSL and isfile(Settings.ssl.SERVER_KEY) and isfile(Settings.ssl.SERVER_CRT):
                with (
                    open(Settings.ssl.SERVER_KEY) as key_file,
                    open(Settings.ssl.SERVER_CRT) as crt_file,
                    open(Settings.ssl.CA_CRT) as ca_file,
                ):
                    private_key = key_file.read().encode()
                    certificate_chain = crt_file.read().encode()
                    root_certificates = ca_file.read().encode()
                server_credentials = grpc.ssl_server_credentials(
                    [(private_key, certificate_chain)], root_certificates=root_certificates, require_client_auth=True
                )
                self.__server.add_secure_port(self.addr, server_credentials)
            else:
                self.__server.add_insecure_port(self.addr)
        except Exception as e:
            raise Exception(f"Cannot bind the address ({self.addr}): {e}") from e
        self.__server.start()
        self.__server_started = True

    def stop(self) -> None:
        """Stop the GRPC server."""
        self.__server.stop(0)
        self.__server_started = False

    def wait_for_termination(self) -> None:
        """Wait for termination."""
        self.__server.wait_for_termination()

    def is_running(self) -> bool:
        """
        Check if the server is running.

        Returns:
            True if the server is running, False otherwise.

        """
        return self.__server_started
