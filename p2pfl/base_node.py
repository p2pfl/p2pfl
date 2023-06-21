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
from p2pfl.proto import node_pb2
from p2pfl.proto import node_pb2_grpc
from p2pfl.neighbors import Neighbors
from concurrent import futures


"""
- revisar cierre de conexiones directamente conectadas
- nuevos mensajes se añadan a gossip?? o a parte
- NEEDED OBSERVER? -> No xq los eventos eran mensajes en la comunicación entre nodos
"""


class BaseNode:
    #####################
    #     Node Init     #
    #####################

    def __init__(self, host="127.0.0.1", port=None, simulation=True):
        # Set message callbacks
        self.__msg_callbacks = {
            "beat": self.__heartbeat_callback,  # cambiar string x algún tipo de dato más ligero
        }
        # Random port
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
        self.addr = f"{host}:{port}"
        # Neighbors
        self.__neighbors = Neighbors(self.addr)
        # Server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        # Logging
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################

    def start(self, wait=False):
        # Heartbeat
        self.__neighbors.start_heartbeater()
        # Gossping
        self.__neighbors.start_gossiper()
        # Server
        node_pb2_grpc.add_NodeServicesServicer_to_server(self, self.server)
        self.server.add_insecure_port(self.addr)
        self.server.start()
        print(f"Server started at {self.addr} ... ")
        if wait:
            self.server.wait_for_termination()
            print("Server terminated.")

    def stop(self):
        self.server.stop(0)
        # REEMPLAZAR POR STOP DE NEIGHBORS
        self.__neighbors.clear_neis()
        self.__neighbors.stop_heartbeater()
        self.__neighbors.stop_gossiper()

    #############################
    #  Neighborhood management  #
    #############################

    def connect(self, addr):
        print(f"[{self.addr}] connecting to {addr}...")
        self.__neighbors.add(addr, handshake_msg=True)

    def get_neighbors(self, only_direct=False):
        return self.__neighbors.get_all(only_direct)

    def disconnect_from(self, addr):
        print(f"[{self.addr}] removing {addr}...")
        self.__neighbors.remove(addr, disconnect_msg=True)

    ####
    # GRPC - Remote Services
    ####

    def handshake(self, request, _):
        self.__neighbors.add(request.addr, handshake_msg=False)
        return node_pb2.Empty()

    def disconnect(self, request, _):
        self.__neighbors.remove(request.addr, disconnect_msg=False)
        return node_pb2.Empty()

    def send_message(self, request, context):
        # If not processed
        if self.__neighbors.add_processed_msg(request.hash):
            # Gossip
            self.__neighbors.gossip(request)
            # Process message
            if request.cmd in self.__msg_callbacks.keys():
                self.__msg_callbacks[request.cmd](request)
            else:
                raise Exception(f"[{self.addr}] Unknown command: {request.cmd}")
        return node_pb2.Status(status="ok")

    ####
    # Message Handlers
    ####

    # ns si hace falta mensajes o meterlo como callbacks depende de como se haga lo de interpretar mensajes

    #
    # AÑADIR TIMESTAMP COMO ARGUMENTO PARA EVITAR HEARTBEATS GOSSIPEADOS RESIDUALES
    #
    def __heartbeat_callback(self, request):
        self.__neighbors.heartbeat(request.source)
        return node_pb2.Empty()

    ###########################
    #     Observer Events     #
    ###########################


"""
    def notify_heartbeat(self, node):
        self.notify(Events.BEAT_RECEIVED_EVENT, node)

    def notify_conn_to(self, h, p):
        self.notify(Events.CONN_TO_EVENT, (h, p))

    def notify_start_learning(self, r, e):
        self.notify(Events.START_LEARNING_EVENT, (r, e))

    def notify_stop_learning(self, cmd):
        self.notify(Events.STOP_LEARNING_EVENT, None)

    def notify_params(self, params):
        self.notify(Events.PARAMS_RECEIVED_EVENT, (params))

    def notify_metrics(self, node, round, loss, metric):
        self.notify(Events.METRICS_RECEIVED_EVENT, (node, round, loss, metric))

    def notify_train_set_votes(self, node, votes):
        self.notify(Events.TRAIN_SET_VOTE_RECEIVED_EVENT, (node, votes))

    def update(self, event, obj):
        if event == Events.END_CONNECTION_EVENT:
            self.rm_neighbor(obj)

        elif event == Events.NODE_CONNECTED_EVENT:
            n, _ = obj
            n.send(CommunicationProtocol.build_beat_msg(self.get_name()))

        elif event == Events.CONN_TO_EVENT:
            self.connect_to(obj[0], obj[1], full=False)

        elif event == Events.SEND_BEAT_EVENT:
            self.broadcast(CommunicationProtocol.build_beat_msg(self.get_name()))

        elif event == Events.GOSSIP_BROADCAST_EVENT:
            self.broadcast(obj[0], exc=obj[1])

        elif event == Events.PROCESSED_MESSAGES_EVENT:
            node, msgs = obj
            # Comunicate to connections the new messages processed
            for nc in self.__neighbors:
                if nc != node:
                    nc.add_processed_messages(list(msgs.keys()))
            # Gossip the new messages
            self.gossiper.add_messages(list(msgs.values()), node)

        elif event == Events.BEAT_RECEIVED_EVENT:
            self.heartbeater.add_node(obj)
"""
