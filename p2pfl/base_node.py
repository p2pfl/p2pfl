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

import logging
import sys
import grpc
import socket
from concurrent import futures
from p2pfl.proto import node_pb2
from p2pfl.proto import node_pb2_grpc
from p2pfl.neighbors import Neighbors
from p2pfl.settings import Settings
from p2pfl.messages import NodeMessages


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

    def __init__(self, host="127.0.0.1", port=None, simulation=False):
        # Message handlers
        self.__msg_callbacks = {}
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
       
        # Logging
        log_level = logging.getLevelName(Settings.LOG_LEVEL)
        logging.basicConfig(stream=sys.stdout, level=log_level)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################

    def assert_running(self, running):
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

    def start(self, wait=False):
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
        # Heartbeat and Gossip
        self._neighbors.start()
        # Server
        node_pb2_grpc.add_NodeServicesServicer_to_server(self, self.__server)
        self.__server.add_insecure_port(self.addr)
        self.__server.start()
        logging.info(f"({self.addr}) Server started.")
        if wait:
            self.__server.wait_for_termination()
            logging.info(f"({self.addr}) Server terminated.")

    def stop(self):
        """
        Stops the node: server and neighbors(gossip and heartbeat).

        Raises:
            Exception: If the node is not running.
        """
        logging.info(f"({self.addr}) Stopping node...")
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

    def connect(self, addr):
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
        logging.info(f"({self.addr}) connecting to {addr}...")
        return self._neighbors.add(addr, handshake_msg=True)

    def get_neighbors(self, only_direct=False):
        """
        Returns the neighbors of the node.

        Args:
            only_direct (bool): If True, only the direct neighbors will be returned.

        Returns:
            list: The list of neighbors.
        """
        return self._neighbors.get_all(only_direct)

    def disconnect_from(self, addr):
        """
        Disconnects a node from another.

        Args:
            addr (str): The address of the node to disconnect from.
        """
        # Check running
        self.assert_running(True)
        # Disconnect
        logging.info(f"({self.addr}) removing {addr}...")
        self._neighbors.remove(addr, disconnect_msg=True)

    ############################
    #  GRPC - Remote Services  #
    ############################

    def handshake(self, request, _):
        """
        GRPC service. It is called when a node connects to another.
        """
        if self._neighbors.add(request.addr, handshake_msg=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(
                error="Cannot add the node (duplicated or wrong direction)"
            )

    def disconnect(self, request, _):
        """
        GRPC service. It is called when a node disconnects from another.
        """
        self._neighbors.remove(request.addr, disconnect_msg=False)
        return node_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def send_message(self, request, _):
        """
        GRPC service. It is called when a node sends a message to another.
        """
        # If not processed
        if self._neighbors.add_processed_msg(request.hash):
            # Gossip
            self._neighbors.gossip(request)
            # Process message
            if request.cmd in self.__msg_callbacks.keys():
                try:
                    self.__msg_callbacks[request.cmd](request)
                except Exception as e:
                    error_text = f"[{self.addr}] Error while processing command: {request.cmd} {request.args}: {e}"
                    logging.error(error_text)
                    return node_pb2.ResponseMessage(error=error_text)
            else:
                # disconnect node
                logging.error(
                    f"[{self.addr}] Unknown command: {request.cmd} from {request.source}"
                )
                return node_pb2.ResponseMessage(error=f"Unknown command: {request.cmd}")
        return node_pb2.ResponseMessage()

    def add_model(self, request, _):
        raise NotImplementedError

    ####
    # Message Handlers
    ####

    def add_message_handler(self, cmd, callback):
        """
        Adds a function callback to a message.
        
        Args:
            cmd (str): The command of the message.
            callback (function): The callback function.
        """
        self.__msg_callbacks[cmd] = callback

    def __heartbeat_callback(self, request):
        time = float(request.args[0])
        self._neighbors.heartbeat(request.source, time)
