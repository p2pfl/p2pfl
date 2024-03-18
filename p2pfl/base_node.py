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

from typing import Callable, Dict, List, Optional
import grpc
import socket
from concurrent import futures
from p2pfl.proto import node_pb2
from p2pfl.proto import node_pb2_grpc
import google.protobuf.empty_pb2
from p2pfl.neighbors import Neighbors
from p2pfl.messages import NodeMessages
from p2pfl.management.logger import logger


class BaseNode(node_pb2_grpc.NodeServicesServicer):
    """
    This class represents a base node in the network (without **FL**).

    Args:
        host (str): The host of the node.
        port (int): The port of the node.
        simulation (bool): If False, communication will be encrypted.

    Attributes:
        addr (str): The address of the node.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        simulation: bool = False,
    ) -> None:
        # Message handlers
        self.__msg_callbacks: Dict[str, Callable] = {}
        self.add_message_handler(NodeMessages.BEAT, self.__heartbeat_callback)

        # Random port
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
        self.addr = f"{host}:{port}"

        # Neighbors
        self._neighbors = Neighbors(self.addr)
        if simulation:
            raise NotImplementedError("Simulation not implemented yet")

        # Server
        self.__running = False
        self.__server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

    #######################################
    #   Node Management (servicer loop)   #
    #######################################

    def assert_running(self, running: bool) -> None:
        """
        Asserts that the node is running or not running.

        Args:
            running (bool): True if the node must be running, False otherwise.

        Raises:
            Exception: If the node is not running and running is True, or if the node is running and running is False.
        """
        running_state = self.__running
        if running_state != running:
            raise Exception(f"Node is {'not ' if running_state else ''}running.")

    def start(self, wait: bool = False) -> None:
        """
        Starts the node: server and neighbors(gossip and heartbeat).

        Args:
            wait (bool): If True, the function will wait until the server is terminated.

        Raises:
            Exception: If the node is already running.
        """
        # Check not running
        self.assert_running(False)
        # Set running
        self.__running = True
        # Server
        node_pb2_grpc.add_NodeServicesServicer_to_server(self, self.__server)
        try:
            self.__server.add_insecure_port(self.addr)
        except Exception as e:
            raise Exception(f"Cannot bind the address ({self.addr}): {e}")
        self.__server.start()
        logger.info(self.addr, "gRPC started")
        # Heartbeat and Gossip
        self._neighbors.start()
        if wait:
            self.__server.wait_for_termination()
            logger.info(self.addr, "gRPC terminated.")

    def stop(self) -> None:
        """
        Stops the node: server and neighbors(gossip and heartbeat).

        Raises:
            Exception: If the node is not running.
        """
        logger.info(self.addr, "Stopping node...")
        # Check running
        self.assert_running(True)
        # Stop server
        self.__server.stop(0)
        # Stop neighbors
        self._neighbors.stop()
        # Set not running
        self.__running = False

    #############################
    #  Neighborhood management  #
    #############################

    def connect(self, addr: str) -> bool:
        """
        Connects a node to another.

        Args:
            addr (str): The address of the node to connect to.

        Returns:
            bool: True if the node was connected, False otherwise.
        """
        # Check running
        self.assert_running(True)
        # Connect
        logger.info(self.addr, f"Connecting to {addr}...")
        return self._neighbors.add(addr, handshake_msg=True)

    def get_neighbors(self, only_direct: bool = False) -> List[str]:
        """
        Returns the neighbors of the node.

        Args:
            only_direct (bool): If True, only the direct neighbors will be returned.

        Returns:
            list: The list of neighbors.
        """
        return self._neighbors.get_all(only_direct)

    def disconnect_from(self, addr: str) -> None:
        """
        Disconnects a node from another.

        Args:
            addr (str): The address of the node to disconnect from.
        """
        # Check running
        self.assert_running(True)
        # Disconnect
        logger.info(self.addr, f"Removing {addr}...")
        self._neighbors.remove(addr, disconnect_msg=True)

    ############################
    #  GRPC - Remote Services  #
    ############################

    def handshake(
        self, request: node_pb2.HandShakeRequest, _: grpc.ServicerContext
    ) -> node_pb2.ResponseMessage:
        """
        GRPC service. It is called when a node connects to another.
        """
        if self._neighbors.add(request.addr, handshake_msg=False):
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
        self._neighbors.remove(request.addr, disconnect_msg=False)
        return google.protobuf.empty_pb2.Empty()

    def send_message(
        self, request: node_pb2.Message, _: grpc.ServicerContext
    ) -> node_pb2.ResponseMessage:
        """
        GRPC service. It is called when a node sends a message to another.
        """
        # If not processed
        if self._neighbors.add_processed_msg(request.hash):
            logger.debug(
                self.addr,
                f"Received message from {request.source} > {request.cmd} {request.args}",
            )
            # Gossip
            self._neighbors.gossip(request)
            # Process message
            if request.cmd in self.__msg_callbacks.keys():
                try:
                    self.__msg_callbacks[request.cmd](request)
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

    def add_model(
        self, request: node_pb2.Weights, _: grpc.ServicerContext
    ) -> node_pb2.ResponseMessage:
        raise NotImplementedError

    ####
    # Message Handlers
    ####

    def add_message_handler(self, cmd: str, callback: Callable) -> None:
        """
        Adds a function callback to a message.

        Args:
            cmd (str): The command of the message.
            callback (callable): The callback function.
        """
        self.__msg_callbacks[cmd] = callback

    def __heartbeat_callback(self, request: node_pb2.Message) -> None:
        time = float(request.args[0])
        self._neighbors.heartbeat(request.source, time)
