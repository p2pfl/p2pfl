#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Protocol agnostic gossiper."""

import random
import threading
import time
from collections.abc import Callable
from typing import Any

from p2pfl.communication.commands.message.pre_send_model_command import PreSendModelCommand
from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent


class Gossiper(threading.Thread, NodeComponent):
    """Gossiper for agnostic communication protocol."""

    def __init__(
        self,
        neighbors: Neighbors,
        build_msg: Callable[..., node_pb2.RootMessage],
        period: float | None = None,
        messages_per_period: int | None = None,
    ) -> None:
        """Initialize the gossiper."""
        if period is None:
            period = Settings.gossip.PERIOD
        if messages_per_period is None:
            messages_per_period = Settings.gossip.MESSAGES_PER_PERIOD
        # Thread
        super().__init__()
        self.name = "gossiper-thread-unknown"

        # Lists, locks and flag
        self.__processed_messages: list[int] = []
        self.__processed_messages_lock = threading.Lock()
        self.__pending_msgs: list[tuple[node_pb2.RootMessage, list[ProtobuffClient]]] = []
        self.__pending_msgs_lock = threading.Lock()
        self.__gossip_terminate_flag = threading.Event()

        # Props
        self.__neighbors = neighbors
        self.period = period
        self.messages_per_period = messages_per_period

        # Build msgs
        self.build_msg_fn = build_msg

    def set_addr(self, addr: str) -> str:
        """Set the address."""
        addr = super().set_addr(addr)
        self.name = f"gossiper-thread-{addr}"
        return addr

    ###
    # Thread control
    ###

    def start(self) -> None:
        """Start the gossiper thread."""
        logger.info(self.addr, "🏁 Starting gossiper...")
        return super().start()

    def stop(self) -> None:
        """Stop the gossiper thread."""
        logger.info(self.addr, "🛑 Stopping gossiper...")
        self.__gossip_terminate_flag.set()

    ###
    # Gossip
    ###

    def add_message(self, msg: node_pb2.RootMessage) -> None:
        """
        Add message to pending.

        Args:
            msg: Message to send.
            pending_neis: Neighbors to send the message.

        """
        self.__pending_msgs_lock.acquire()
        pending_neis = [v[0] for addr, v in self.__neighbors.get_all(only_direct=True).items() if addr != self.addr and addr != msg.source]
        self.__pending_msgs.append((msg, pending_neis))
        self.__pending_msgs_lock.release()

    def check_and_set_processed(self, msg: node_pb2.RootMessage) -> bool:
        """
        Check if message was already processed and set it as processed.

        Args:
            msg: Message to check.

        """
        # If self address, return False
        if msg.source == self.addr:
            return False

        # Check if message was already processed
        with self.__processed_messages_lock:
            if msg.gossip_message.hash in self.__processed_messages:
                return False
            # If there are more than X messages, remove the oldest one
            if len(self.__processed_messages) > Settings.gossip.AMOUNT_LAST_MESSAGES_SAVED:
                self.__processed_messages.pop(0)
            # Add message
            self.__processed_messages.append(msg.gossip_message.hash)
            return True

    def run(self) -> None:
        """Run the gossiper thread."""
        while not self.__gossip_terminate_flag.is_set():
            t = time.time()
            messages_to_send = []
            messages_left = self.messages_per_period

            # Lock
            with self.__pending_msgs_lock:
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
                        self.__pending_msgs[0] = (
                            self.__pending_msgs[0][0],
                            self.__pending_msgs[0][1][messages_left:],
                        )

            # Send messages
            for msg, neis in messages_to_send:
                for nei in neis:
                    nei.send(msg)
            # Sleep to allow periodicity
            sleep_time = max(0, self.period - (t - time.time()))
            time.sleep(sleep_time)

    ###
    # Gossip Model (syncronous gossip not as a thread)
    ###

    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], list[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], tuple[Any, str, int, list[str]]],  # TODO: this can be simplified
        period: float,
        temporal_connection: bool,
    ) -> None:
        """
        Gossip model weights. This is a synchronous gossip. End when there are no more neighbors to gossip.

        Args:
            early_stopping_fn: Function to check if the gossip should stop.
            get_candidates_fn: Function to get the neighbors to gossip.
            status_fn: Function to get the status of the node.
            model_fn: Function to get the model of a neighbor.
            period: Period of gossip.
            temporal_connection: Flag to create a connection if neis not connected directly.

        """
        # Initialize list with status of nodes in the last X iterations
        last_x_status: list[Any] = []
        j = 0

        while True:
            # Get time to calculate frequency
            t = time.time()

            # If the trainning has been interrupted, stop waiting
            if early_stopping_fn():
                logger.info(self.addr, "Stopping model gossip process.")
                return

            # Get nodes wich need models
            neis = get_candidates_fn()

            # Determine end of gossip
            if neis == []:
                logger.info(self.addr, "🤫 Gossip finished.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            logger.debug(self.addr, f"👥 Gossip remaining nodes: {neis}")
            if len(last_x_status) != Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS:
                last_x_status.append(status_fn())
            else:
                last_x_status[j] = str(status_fn())
                j = (j + 1) % Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS

                # Check if las messages are the same
                for i in range(len(last_x_status) - 1):
                    if last_x_status[i] != last_x_status[i + 1]:
                        break
                    logger.info(
                        self.addr,
                        f"⏹️  Gossiping exited for {Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS} equal rounds.",
                    )
                    logger.debug(self.addr, f"Gossip last status: {last_x_status[-1]}")
                    return

            # Select a random subset of neighbors
            samples = min(Settings.gossip.MODELS_PER_ROUND, len(neis))
            neis = random.sample(neis, samples)
            # Getting all nodes and forcing tmp direct message
            neis_clients = [v[0] for k, v in self.__neighbors.get_all(only_direct=False).items() if k in neis]

            # Generate and Send Model Partial Aggregations (model, node_contributors)
            for client in neis_clients:
                # Get Model
                model, command_name, round, model_hashes = model_fn(client.nei_addr)
                if model is None:
                    continue

                # Pre send weights
                presend_msg = self.build_msg_fn(PreSendModelCommand.get_name(), [command_name] + model_hashes, round, direct=True)
                presend_response = client.send(presend_msg, temporal_connection=temporal_connection)

                # Send model
                if presend_response != "true":
                    logger.debug(
                        self.addr, f"Avoiding concurrent model sending to {client.nei_addr}. Msg: {command_name} | Hash: {model_hashes}"
                    )
                    continue

                # Send
                logger.debug(self.addr, f"🗣️ Gossiping model to {client.nei_addr}.")
                client.send(model, temporal_connection=temporal_connection)

            # Sleep to allow periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)
