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

import grpc
from concurrent import futures
from typing import List, Union
from p2pfl.management.logger import logger
from p2pfl.commands.command import Command
from p2pfl.communication.gossiper import Gossiper
from p2pfl.communication.grpc.proto import node_pb2
from p2pfl.communication.grpc.proto import node_pb2_grpc
from p2pfl.communication.grpc.neighbors import GrpcNeighbors
import google.protobuf.empty_pb2


class GrpcServer(node_pb2_grpc.NodeServicesServicer):

    ####
    # Init
    ####

    def __init__(
        self,
        addr: str,
        gossiper: Gossiper,
        neighbors: GrpcNeighbors,
        commands: List[Command] = {},
    ) -> None:

        # Message handlers
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
        Starts the GRPC server.
        """
        # Server
        node_pb2_grpc.add_NodeServicesServicer_to_server(self, self.__server)
        try:
            self.__server.add_insecure_port(self.addr)
        except Exception as e:
            raise Exception(f"Cannot bind the address ({self.addr}): {e}")
        self.__server.start()

    def stop(self) -> None:
        """
        Stops the GRPC server.
        """
        self.__server.stop(0)

    def wait_for_termination(self) -> None:
        """
        Waits for termination.
        """
        self.__server.wait_for_termination()

    ####
    # GRPC Services
    ####

    def handshake(
        self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext
    ) -> node_pb2.ResponseMessage:
        """
        GRPC service. It is called when a node connects to another.
        """
        if self.__neighbors.add(request.addr, non_direct=False, handshake_msg=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(
                error="Cannot add the node (duplicated or wrong direction)"
            )

    def disconnect(
        self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext
    ) -> google.protobuf.empty_pb2.Empty:
        """
        GRPC service. It is called when a node disconnects from another.
        """
        self.__neighbors.remove(request.addr, disconnect_msg=False)
        return google.protobuf.empty_pb2.Empty()

    def send_message(
        self, request: node_pb2.Message, _: grpc.ServicerContext
    ) -> node_pb2.ResponseMessage:
        """
        GRPC service. It is called when a node sends a message to another.
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
                pending_neis = [
                    n
                    for n in self.__neighbors.get_all(only_direct=True).keys()
                    if n != request.source
                ]
                self.__gossiper.add_message(request, pending_neis)

            # Process message
            if request.cmd in self.__commands.keys():
                try:
                    self.__commands[request.cmd].execute(
                        request.source, request.round, *request.args
                    )

                except Exception as e:
                    error_text = f"Error while processing command: {request.cmd} {request.args}: {e}"
                    logger.error(self.addr, error_text)
                    return node_pb2.ResponseMessage(error=error_text)
            else:
                # disconnect node
                logger.error(
                    self.addr, f"Unknown command: {request.cmd} from {request.source}"
                )
                return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")

        return node_pb2.ResponseMessage()

    # Note: main diff with send_message (apart from the message type) is the gossip
    #   -> ADDED GOAL OF THE MESSAGE TO INCREASE ROBUSNTNESS
    def send_weights(
        self, request: node_pb2.Weights, _: grpc.ServicerContext
    ) -> node_pb2.ResponseMessage:
        # Process message
        if request.cmd in self.__commands.keys():
            try:
                self.__commands[request.cmd].execute(
                    request.source,
                    request.round,
                    request.weights,
                    request.contributors,
                    request.weight,
                )
            except Exception as e:
                error_text = f"Error while processing model: {request.cmd}: {e}"
                logger.error(self.addr, error_text)
                return node_pb2.ResponseMessage(error=error_text)
        else:
            # disconnect node
            logger.error(
                self.addr, f"Unknown command: {request.cmd} from {request.source}"
            )
            return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")
        return node_pb2.ResponseMessage()

    ####
    # Commands
    ####

    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        """
        Adds a command.

        Args:

        """
        if isinstance(cmds, list):
            for cmd in cmds:
                self.__commands[cmd.get_name()] = cmd
        elif isinstance(cmds, Command):
            self.__commands[cmds.get_name()] = cmds
        else:
            raise Exception("Command not valid")
