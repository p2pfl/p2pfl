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

"""Protobuff server."""

import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union

import google.protobuf.empty_pb2
import grpc

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2, node_pb2_grpc
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check


class ProtobuffServer(ABC, node_pb2_grpc.NodeServicesServicer, NodeComponent):
    """
    Implementation of the server side logic of PROTOBUFF communication protocol.

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
        commands: Optional[list[Command]] = None,
    ) -> None:
        """Initialize the GRPC server."""
        # Message handlers
        if commands is None:
            commands = []
        self.__commands = {c.get_name(): c for c in commands}

        # (addr) Super
        NodeComponent.__init__(self)

        # Gossiper
        self._gossiper = gossiper

        # Neighbors
        self._neighbors = neighbors

    ####
    # Management
    ####

    @abstractmethod
    def start(self, wait: bool = False) -> None:
        """
        Start the server.

        Args:
            wait: If True, wait for termination.

        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the server."""
        pass

    @abstractmethod
    def wait_for_termination(self) -> None:
        """Wait for termination."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the server is running.

        Returns:
            True if the server is running, False otherwise.

        """
        pass

    ####
    # Service Implementation (server logic on protobuff)
    ####

    def handshake(self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext) -> node_pb2.ResponseMessage:
        """
        Service. It is called when a node connects to another.

        Args:
            request: Request message.
            _: Context.

        """
        if self._neighbors.add(request.addr, non_direct=False, handshake=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(error="Cannot add the node (duplicated or wrong direction)")

    def disconnect(self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext) -> google.protobuf.empty_pb2.Empty:
        """
        Service. It is called when a node disconnects from another.

        Args:
            request: Request message.
            _: Context.

        """
        self._neighbors.remove(request.addr, disconnect_msg=False)
        return google.protobuf.empty_pb2.Empty()

    def send(self, request: node_pb2.RootMessage, _: grpc.ServicerContext) -> node_pb2.ResponseMessage:
        """
        Service. Handles both regular messages and model weights.

        Args:
            request: The RootMessage containing either a Message or Weights payload.
            _: Context.

        """
        # If message already processed, return
        if request.HasField("message") and not self._gossiper.check_and_set_processed(request):
            """
            if request.cmd != "beat" or (not Settings.heartbeat.EXCLUDE_BEAT_LOGS and request.source == "beat"):
                logger.debug(self.addr, f"🙅 Message already processed: {request.cmd} (id {request.message.hash})")
            """
            return node_pb2.ResponseMessage()

        # Process message/model
        if request.cmd != "beat" or (not Settings.heartbeat.EXCLUDE_BEAT_LOGS and request.cmd == "beat"):
            emoji = "📫" if request.HasField("message") else "📦"
            logger.debug(
                self.addr,
                f"{emoji} {request.cmd.upper()} received from {request.source}",
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
            self._gossiper.add_message(request)

        return node_pb2.ResponseMessage()

    ####
    # Commands
    ####

    @allow_no_addr_check
    def add_command(self, cmds: Union[Command, list[Command]]) -> None:
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
