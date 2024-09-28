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

import traceback
from concurrent import futures
from os.path import isfile
from typing import List, Optional, Union

import google.protobuf.empty_pb2

import grpc
from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.gossiper import Gossiper
from p2pfl.communication.protocols.grpc.grpc_neighbors import GrpcNeighbors
from p2pfl.communication.protocols.grpc.proto import node_pb2, node_pb2_grpc
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class GrpcServer(node_pb2_grpc.NodeServicesServicer):
    """
    Implementation of the server side of a GRPC communication protocol.

    Args:
        addr: Address of the server.
        gossiper: Gossiper instance.
        neighbors: Neighbors instance.
        commands: List of commands to be executed by the server.

    """

    def __init__(
        self,
        addr: str,
        gossiper: Gossiper,
        neighbors: GrpcNeighbors,
        commands: Optional[List[Command]] = None,
    ) -> None:
        """Initialize the GRPC server."""
        # Message handlers
        if commands is None:
            commands = []
        self.__commands = {c.get_name(): c for c in commands}

        # Address
        self.addr = addr

        # Server
        self.__server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        self.__server_started = False

        # Gossiper
        self.__gossiper = gossiper

        # Neighbors
        self.__neighbors = neighbors

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
            if Settings.USE_SSL and isfile(Settings.SERVER_KEY) and isfile(Settings.SERVER_CRT):
                with open(Settings.SERVER_KEY) as key_file, open(Settings.SERVER_CRT) as crt_file, open(Settings.CA_CRT) as ca_file:
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

    ####
    # GRPC Services
    ####

    def handshake(self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext) -> node_pb2.ResponseMessage:
        """
        GRPC service. It is called when a node connects to another.

        Args:
            request: Request message.
            _: Context.

        """
        if self.__neighbors.add(request.addr, non_direct=False, handshake_msg=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(error="Cannot add the node (duplicated or wrong direction)")

    def disconnect(self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext) -> google.protobuf.empty_pb2.Empty:
        """
        GRPC service. It is called when a node disconnects from another.

        Args:
            request: Request message.
            _: Context.

        """
        self.__neighbors.remove(request.addr, disconnect_msg=False)
        return google.protobuf.empty_pb2.Empty()

    def send(self, request: node_pb2.RootMessage, _: grpc.ServicerContext) -> node_pb2.ResponseMessage:
        """
        GRPC service. Handles both regular messages and model weights.

        Args:
            request: The RootMessage containing either a Message or Weights payload.
            _: Context.

        """
        # If message already proce
        # ssed, return
        if request.HasField("message") and not self.__gossiper.check_and_set_processed(request.message.hash):
            """
            if request.cmd != "beat" or (not Settings.EXCLUDE_BEAT_LOGS and request.source == "beat"):
                logger.debug(self.addr, f"🙅 Message already processed: {request.cmd} (id {request.message.hash})")
            """
            return node_pb2.ResponseMessage()

        # Process message/model
        if request.cmd != "beat" or (not Settings.EXCLUDE_BEAT_LOGS and request.cmd == "beat"):
            logger.debug(
                self.addr,
                f"📫 {request.cmd.upper()} received from {request.source}",
            )
        if request.cmd in self.__commands:
            try:
                if request.HasField("message"):
                    self.__commands[request.cmd].execute(request.source, request.round, *request.message.args)
                elif request.HasField("weights"):
                    self.__commands[request.cmd].execute(
                        request.source,
                        request.round,
                        weights=request.weights.weights,
                        contributors=request.weights.contributors,
                        num_samples=request.weights.num_samples,
                    )
                else:
                    error_text = f"Error while processing command: {request.cmd}: No message or weights"
                    logger.error(self.addr, error_text)
                    return node_pb2.ResponseMessage(error=error_text)
            except Exception as e:
                error_text = f"Error while processing command: {request.cmd}. {type(e).__name__}: {e}"
                logger.error(self.addr, error_text + f"\n{traceback.format_exc()}")
                return node_pb2.ResponseMessage(error=error_text)
        else:
            # disconnect node
            logger.error(self.addr, f"Unknown command: {request.cmd} from {request.source}")
            return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")

        # If message gossip
        if request.HasField("message") and request.message.ttl > 0:
            # Update ttl and gossip
            request.message.ttl -= 1
            pending_neis = [n for n in self.__neighbors.get_all(only_direct=True) if n != request.source]
            self.__gossiper.add_message(request, pending_neis)

        return node_pb2.ResponseMessage()

    ####
    # Commands
    ####

    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        """
        Add a command.

        Args:
            cmds: Command or list of commands to be added.

        """
        if isinstance(cmds, list):
            for cmd in cmds:
                self.__commands[cmd.get_name()] = cmd
        elif isinstance(cmds, Command):
            self.__commands[cmds.get_name()] = cmds
        else:
            raise Exception("Command not valid")
