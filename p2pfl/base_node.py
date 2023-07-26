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


class BaseNode:
    #####################
    #     Node Init     #
    #####################

    def __init__(self, host="127.0.0.1", port=None, simulation=True):
        # Set message handlers
        self.__msg_callbacks = {}
        self.add_message_handler(NodeMessages.BEAT, self.__heartbeat_callback)
        # Is running
        self.__running = False
        # Random port
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
        self.addr = f"{host}:{port}"
        # Neighbors
        self._neighbors = Neighbors(self.addr)
        # Server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        # Logging
        log_level = logging.getLevelName(Settings.LOG_LEVEL)
        logging.basicConfig(stream=sys.stdout, level=log_level)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################

    def assert_running(self, running):
        running_state = self.__running
        if running_state != running:
            raise Exception(f"Node is {'not ' if running_state else ''}running.")

    def start(self, wait=False):
        # Check not running
        self.assert_running(False)
        # Set running
        self.__running = True
        # Heartbeat and Gossip
        self._neighbors.start()
        # Server
        node_pb2_grpc.add_NodeServicesServicer_to_server(self, self.server)
        self.server.add_insecure_port(self.addr)
        self.server.start()
        logging.info(f"({self.addr}) Server started.")
        if wait:
            self.server.wait_for_termination()
            logging.info(f"({self.addr}) Server terminated.")

    def stop(self):
        logging.info(f"({self.addr}) Stopping node...")
        # Check running
        self.assert_running(True)
        # Stop server
        self.server.stop(0)
        # Stop neighbors
        self._neighbors.stop()
        # Set not running
        self.__running = False

    #############################
    #  Neighborhood management  #
    #############################

    def connect(self, addr):
        # Check running
        self.assert_running(True)
        # Connect
        logging.info(f"({self.addr}) connecting to {addr}...")
        return self._neighbors.add(addr, handshake_msg=True)

    def get_neighbors(self, only_direct=False):
        return self._neighbors.get_all(only_direct)

    def disconnect_from(self, addr):
        # Check running
        self.assert_running(True)
        # Disconnect
        logging.info(f"({self.addr}) removing {addr}...")
        self._neighbors.remove(addr, disconnect_msg=True)

    ############################
    #  GRPC - Remote Services  #
    ############################

    def handshake(self, request, _):
        if self._neighbors.add(request.addr, handshake_msg=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(
                error="Cannot add the node (duplicated or wrong direction)"
            )

    def disconnect(self, request, _):
        self._neighbors.remove(request.addr, disconnect_msg=False)
        return node_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def send_message(self, request, context):
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

    def add_model(self, request, context):
        raise NotImplementedError

    ####
    # Message Handlers
    ####

    def add_message_handler(self, cmd, callback):
        self.__msg_callbacks[cmd] = callback

    def __heartbeat_callback(self, request):
        time = float(request.args[0])
        self._neighbors.heartbeat(request.source, time)
