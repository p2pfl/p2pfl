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

"""GRPC server."""

from concurrent import futures
from typing import List, Optional, Union

import google.protobuf.empty_pb2
import grpc

from p2pfl.commands.command import Command
from p2pfl.communication.gossiper import Gossiper
from p2pfl.communication.grpc.grpc_neighbors import GrpcNeighbors
from p2pfl.communication.grpc.proto import node_pb2, node_pb2_grpc
from p2pfl.management.logger import logger


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
            self.__server.add_insecure_port(self.addr)
        except Exception as e:
            raise Exception(f"Cannot bind the address ({self.addr}): {e}") from e
        self.__server.start()

    def stop(self) -> None:
        """Stop the GRPC server."""
        self.__server.stop(0)

    def wait_for_termination(self) -> None:
        """Wait for termination."""
        self.__server.wait_for_termination()

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

    def disconnect(
        self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext
    ) -> google.protobuf.empty_pb2.Empty:
        """
        GRPC service. It is called when a node disconnects from another.

        Args:
            request: Request message.
            _: Context.

        """
        self.__neighbors.remove(request.addr, disconnect_msg=False)
        return google.protobuf.empty_pb2.Empty()

    def send_message(self, request: node_pb2.Message, _: grpc.ServicerContext) -> node_pb2.ResponseMessage:
        """
        GRPC service. It is called when a node sends a message to another.

        Args:
            request: Request message.
            _: Context.

        """
        # If not processed
        if self.__gossiper.check_and_set_processed(request.hash):
            logger.debug(
                self.addr,
                f"Received message from {request.source} > {request.cmd} {request.args}",
            )
            # Gossip
            if request.ttl > 1:
                # Update ttl and gossip
                request.ttl -= 1
                pending_neis = [n for n in self.__neighbors.get_all(only_direct=True) if n != request.source]
                self.__gossiper.add_message(request, pending_neis)

            # Process message
            if request.cmd in self.__commands:
                try:
                    self.__commands[request.cmd].execute(request.source, request.round, *request.args)

                except Exception as e:
                    error_text = f"Error while processing command: {request.cmd} {request.args}: {e}"
                    logger.error(self.addr, error_text)
                    return node_pb2.ResponseMessage(error=error_text)
            else:
                # disconnect node
                logger.error(self.addr, f"Unknown command: {request.cmd} from {request.source}")
                return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")

        return node_pb2.ResponseMessage()

    def send_weights(self, request: node_pb2.Weights, _: grpc.ServicerContext) -> node_pb2.ResponseMessage:
        """
        GRPC service. It is called when a node sends weights to another.

        .. note:: Main diff with send_message (apart from the message type) is the gossip

        Args:
            request: Request message.
            _: Context.

        """
        # Process message
        if request.cmd in self.__commands:
            try:
                self.__commands[request.cmd].execute(
                    request.source,
                    request.round,
                    weights=request.weights,
                    contributors=request.contributors,
                    weight=request.weight,
                )
            except Exception as e:
                error_text = f"Error while processing model: {request.cmd}: {e}"
                logger.error(self.addr, error_text)
                return node_pb2.ResponseMessage(error=error_text)
        else:
            # disconnect node
            logger.error(self.addr, f"Unknown command: {request.cmd} from {request.source}")
            return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")
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
