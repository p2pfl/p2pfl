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

import threading
import time
import random
from datetime import datetime
import grpc
from p2pfl.settings import Settings
from p2pfl.proto import node_pb2, node_pb2_grpc
from p2pfl.messages import NodeMessages
import logging


class Neighbors:
    """
    This class is to manage the neighbors of a node.
        - Add neighbors (check duplicates)
        - Remove neighbors
        - Get neighbors
        - Heartbeat: remove neighbors that not send a heartbeat in a period of time
    """

    def __init__(self, self_addr):
        self.__self_addr = self_addr
        self.__neighbors = {}  # private to avoid concurrency issues
        self.__nei_lock = threading.Lock()

        # Heartbeat
        self.__heartbeat_terminate_flag = threading.Event()

        # Gossip
        self.__pending_msgs = []
        self.__pending_msgs_lock = threading.Lock()
        self.__gossip_terminate_flag = threading.Event()
        self.__processed_messages = []
        self.__processed_messages_lock = threading.Lock()

    def start(self):
        self.start_heartbeater()
        self.start_gossiper()

    def stop(self):
        self.stop_heartbeater()
        self.stop_gossiper()
        self.clear_neis()

    ####
    # Message
    ####

    def build_msg(self, cmd, args=[]):
        hs = hash(
            str(cmd) + str(args) + str(datetime.now()) + str(random.randint(0, 100000))
        )
        args = [str(a) for a in args]
        return node_pb2.Message(
            source=self.__self_addr, ttl=Settings.TTL, hash=hs, cmd=cmd, args=args
        )

    def send_message(self, nei, msg):
        try:
            self.__neighbors[nei][1].send_message(msg, timeout=Settings.GRPC_TIMEOUT)
        except Exception as e:
            """
            MAYBE ADD CUSTOM EXCEPTIONS -> direct disconnection
            """
            # Remove neighbor
            logging.info(
                f"({self.__self_addr}) Cannot send message {msg.cmd} to {nei}. Error: {str(e)}"
            )
            self.remove(nei)

    def broadcast_msg(self, msg, node_list=None):
        # Node list
        if node_list is not None:
            node_list = node_list
        else:
            node_list = self.get_all(only_direct=True)
        # Send
        for n in node_list:
            self.send_message(n, msg)

    def send_model(self, nei, round, serialized_model, contributors=[], weight=0):
        try:
            stub = self.__neighbors[nei][1]
            # if not connected, create a temporal stub to send the message
            if stub is None:
                channel = grpc.insecure_channel(nei)
                stub = node_pb2_grpc.NodeServicesStub(channel)
            else:
                channel = None
            stub.add_model(
                node_pb2.Weights(
                    source=self.__self_addr,
                    round=round,
                    weights=serialized_model,
                    contributors=contributors,
                    weight=weight,
                ),
                timeout=Settings.GRPC_TIMEOUT,
            )
            if not (channel is None):
                channel.close()

        except Exception as e:
            # Remove neighbor
            print(f"Cannot send model to {nei}. Error: {str(e)}")
            self.remove(nei)

    ####
    # Neighbors management
    ####

    def add(self, addr, handshake_msg=True, non_direct=False):  # si h
        # Cannot add itself
        if addr == self.__self_addr:
            print("Cannot add itself")
            return False

        # Cannot add duplicates
        self.__nei_lock.acquire()
        duplicated = addr in self.__neighbors.keys()
        self.__nei_lock.release()
        # Avoid adding if duplicated and not non_direct neighbor (otherwise, connect creating a channel)
        if duplicated and not non_direct:
            print("Cannot add duplicates")
            return False

        # Add non direct connected neighbors
        if non_direct:
            self.__nei_lock.acquire()
            self.__neighbors[addr] = [None, None, time.time()]
            self.__nei_lock.release()
            return True

        try:
            # Create channel and stub
            channel = grpc.insecure_channel(addr)
            stub = node_pb2_grpc.NodeServicesStub(channel)

            # Handshake
            if handshake_msg:
                res = stub.handshake(node_pb2.HandShakeRequest(addr=self.__self_addr))
                if not res.bool:
                    channel.close()
                    return False

            # Add neighbor
            self.__nei_lock.acquire()
            self.__neighbors[addr] = [channel, stub, time.time()]
            self.__nei_lock.release()
            return True

        except Exception as e:
            print("Crash desde:")
            print(e)
            print(self.__neighbors)
            # Try to remove neighbor
            try:
                self.remove(addr)
            except:
                pass
            return False

    def remove(self, nei, disconnect_msg=True):
        print(f"({self.__self_addr}) Removing {nei}")
        self.__nei_lock.acquire()
        try:
            try:
                # If the other node still connected, disconnect
                if disconnect_msg:
                    self.__neighbors[nei][1].disconnect(
                        node_pb2.HandShakeRequest(addr=self.__self_addr)
                    )
                # Close channel
                self.__neighbors[nei][0].close()
            except:
                pass
            # Remove neighbor
            del self.__neighbors[nei]
        except:
            pass
        self.__nei_lock.release()

    def get(self, nei):
        return self.__neighbors[nei][1]

    def get_all(self, only_direct=False):
        neis = self.__neighbors.copy()
        if only_direct:
            return [k for k, v in neis.items() if v[1] is not None]
        return list(neis.keys())

    def clear_neis(self):
        nei_copy = self.__neighbors.copy()
        for nei in nei_copy.keys():
            self.remove(nei)

    ####
    # Heartbeating
    ####

    def heartbeat(self, nei, time):
        self.__nei_lock.acquire()
        if nei not in self.__neighbors.keys():
            self.__nei_lock.release()
            # Add non-direct connected neighbor
            self.add(nei, non_direct=True)

        else:
            # Update time
            if self.__neighbors[nei][2] < time:
                self.__neighbors[nei][2] = time
            self.__nei_lock.release()

    def start_heartbeater(self):
        threading.Thread(target=self.__heartbeater).start()

    def stop_heartbeater(self):
        self.__heartbeat_terminate_flag.set()

    def __heartbeater(
        self, period=Settings.HEARTBEAT_PERIOD, timeout=Settings.HEARTBEAT_TIMEOUT
    ):
        toggle = False
        while not self.__heartbeat_terminate_flag.is_set():
            t = time.time()

            # Check heartbeats (every 2 periods)
            if toggle:
                nei_copy = self.__neighbors.copy()
                for nei in nei_copy.keys():
                    if t - nei_copy[nei][2] > timeout:
                        logging.info(
                            f"({self.__self_addr}) Heartbeat timeout for {nei} ({t - nei_copy[nei][2]}). Removing..."
                        )
                        self.remove(nei)
            else:
                toggle = True

            # Send heartbeat
            nei_copy = self.__neighbors.copy()
            for nei, (_, stub, _) in nei_copy.items():
                if stub is None:
                    continue
                try:
                    stub.send_message(
                        self.build_msg(NodeMessages.BEAT, args=[str(time.time())])
                    )
                except Exception as e:
                    logging.info(
                        f"({self.__self_addr}) Cannot send heartbeat to {nei}. Error: {str(e)}"
                    )
                    self.remove(nei)

            # Sleep to allow the periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)

    ####
    # Gossping
    ####

    def add_processed_msg(self, msg):
        self.__processed_messages_lock.acquire()
        # Check if message was already processed
        if msg in self.__processed_messages:
            self.__processed_messages_lock.release()
            return False
        # If there are more than X messages, remove the oldest one
        if len(self.__processed_messages) > Settings.AMOUNT_LAST_MESSAGES_SAVED:
            self.__processed_messages.pop(0)
        # Add message
        self.__processed_messages.append(msg)
        self.__processed_messages_lock.release()
        return True

    def gossip(self, msg):
        if msg.ttl > 1:
            # Update ttl and broadcast
            msg.ttl -= 1

            # Add to pending messages
            self.__pending_msgs_lock.acquire()
            pending_neis = [n for n in self.__neighbors.keys() if n != msg.source]
            self.__pending_msgs.append((msg, pending_neis))
            self.__pending_msgs_lock.release()

    def start_gossiper(self):
        threading.Thread(target=self.__gossiper).start()

    def stop_gossiper(self):
        self.__gossip_terminate_flag.set()

    def __gossiper(
        self,
        period=Settings.GOSSIP_PERIOD,
        messases_per_period=Settings.GOSSIP_MESSAGES_PER_PERIOD,
    ):
        """
        quizá añadir locks para evitar iteraciones
        """
        while not self.__gossip_terminate_flag.is_set():
            t = time.time()
            messages_to_send = []
            messages_left = messases_per_period

            # Lock
            self.__pending_msgs_lock.acquire()

            # Select the max amount of messages to send
            while messages_left > 0 and len(self.__pending_msgs) > 0:
                head_msg = self.__pending_msgs[0]
                # Select msgs
                if len(head_msg[1]) < messages_left:
                    # Select all
                    messages_to_send.append(head_msg)
                    # Remove from pending
                    self.__pending_msgs.pop(0)
                else:
                    # Select only the first neis
                    messages_to_send.append((head_msg[0], head_msg[1][:messages_left]))
                    # Remove from pending
                    self.__pending_msgs[0][1] = self.__pending_msgs[0][1][
                        messages_left:
                    ]

            # Unlock
            self.__pending_msgs_lock.release()

            # Send messages
            for msg, neis in messages_to_send:
                for nei in neis:
                    # send only if direct connected (also add a try to deal with desconnections)
                    try:
                        if self.__neighbors[nei][1] is not None:
                            self.send_message(nei, msg)
                    except KeyError:
                        pass
            # Sleep to allow periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)

    def __str__(self):
        return str(self.__neighbors.keys())
