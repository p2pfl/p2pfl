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
from typing import Any, Callable, List, Optional, Tuple

from p2pfl.communication.protocols.client import Client
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class Gossiper(threading.Thread):
    """
    Gossiper for agnostic communication protocol.

    Args:
        self_addr: Address of the node.
        client: Client to send messages.
        period: Period of gossip.
        messages_per_period: Amount of messages to send per period.

    """

    def __init__(
        self,
        self_addr,
        client: Client,  # can be generalized to any protocol
        period: Optional[float] = None,
        messages_per_period: Optional[int] = None,
    ) -> None:
        """Initialize the gossiper."""
        if period is None:
            period = Settings.GOSSIP_PERIOD
        if messages_per_period is None:
            messages_per_period = Settings.GOSSIP_MESSAGES_PER_PERIOD
        # Thread
        super().__init__()
        self.__self_addr = self_addr
        self.name = f"gossiper-thread-{self.__self_addr}"

        # Lists, locks and flag
        self.__processed_messages: List[int] = []
        self.__processed_messages_lock = threading.Lock()
        self.__pending_msgs: List[Tuple[Any, List[str]]] = []
        self.__pending_msgs_lock = threading.Lock()
        self.__gossip_terminate_flag = threading.Event()

        # Props
        self.__client = client
        self.period = period
        self.messages_per_period = messages_per_period

    ###
    # Thread control
    ###

    def start(self) -> None:
        """Start the gossiper thread."""
        logger.info(self.__self_addr, "🏁 Starting gossiper...")
        return super().start()

    def stop(self) -> None:
        """Stop the gossiper thread."""
        logger.info(self.__self_addr, "🛑 Stopping gossiper...")
        self.__gossip_terminate_flag.set()

    ###
    # Gossip
    ###

    def add_message(self, msg: Any, pending_neis: List[str]) -> None:
        """
        Add message to pending.

        Args:
            msg: Message to send.
            pending_neis: Neighbors to send the message.

        """
        self.__pending_msgs_lock.acquire()
        self.__pending_msgs.append((msg, pending_neis))
        self.__pending_msgs_lock.release()

    def check_and_set_processed(self, msg_hash: int) -> bool:
        """
        Check if message was already processed and set it as processed.

        Args:
            msg_hash: Hash of the message to check.

        """
        self.__processed_messages_lock.acquire()
        # Check if message was already processed
        if msg_hash in self.__processed_messages:
            self.__processed_messages_lock.release()
            return False
        # If there are more than X messages, remove the oldest one
        if len(self.__processed_messages) > Settings.AMOUNT_LAST_MESSAGES_SAVED:
            self.__processed_messages.pop(0)
        # Add message
        self.__processed_messages.append(msg_hash)
        self.__processed_messages_lock.release()
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
                    self.__client.send(nei, msg)
            # Sleep to allow periodicity
            sleep_time = max(0, self.period - (t - time.time()))
            time.sleep(sleep_time)

    ###
    # Gossip Model (syncronous gossip not as a thread)
    ###

    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], List[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], Any],
        period: float,
        create_connection: bool,
    ) -> None:
        """
        Gossip model weights. This is a synchronous gossip. End when there are no more neighbors to gossip.

        Args:
            early_stopping_fn: Function to check if the gossip should stop.
            get_candidates_fn: Function to get the neighbors to gossip.
            status_fn: Function to get the status of the node.
            model_fn: Function to get the model of a neighbor.
            period: Period of gossip.
            create_connection: Flag to create a connection.

        """
        # Initialize list with status of nodes in the last X iterations
        last_x_status: List[Any] = []
        j = 0

        while True:
            # Get time to calculate frequency
            t = time.time()

            # If the trainning has been interrupted, stop waiting
            if early_stopping_fn():
                logger.info(self.__self_addr, "Stopping model gossip process.")
                return

            # Get nodes wich need models
            neis = get_candidates_fn()

            # Determine end of gossip
            if neis == []:
                logger.info(self.__self_addr, "🤫 Gossip finished.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            logger.debug(self.__self_addr, f"👥 Gossip remaining nodes: {neis}")
            if len(last_x_status) != Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS:
                last_x_status.append(status_fn())
            else:
                last_x_status[j] = str(status_fn())
                j = (j + 1) % Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS

                # Check if las messages are the same
                for i in range(len(last_x_status) - 1):
                    if last_x_status[i] != last_x_status[i + 1]:
                        break
                    logger.info(
                        self.__self_addr,
                        f"⏹️  Gossiping exited for {Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS} equal rounds.",
                    )
                    logger.debug(self.__self_addr, f"Gossip last status: {last_x_status[-1]}")
                    return

            # Select a random subset of neighbors
            samples = min(Settings.GOSSIP_MODELS_PER_ROUND, len(neis))
            neis = random.sample(neis, samples)

            # Generate and Send Model Partial Aggregations (model, node_contributors)
            for nei in neis:
                # Send Partial Aggregation
                model = model_fn(nei)
                if model is None:
                    continue
                logger.debug(self.__self_addr, f"🗣️ Gossiping model to {nei}.")
                self.__client.send(nei, model, create_connection=create_connection)

            # Sleep to allow periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)
