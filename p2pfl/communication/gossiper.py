import random
import threading
import time
from typing import Any, Callable, List, Tuple

from p2pfl.communication.grpc.grpc_client import GrpcClient
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

# Define type aliases for clarity
CandidateCondition = Callable[[str], bool]
StatusFunction = Callable[[str], Any]
ModelFunction = Callable[[str], Tuple[Any, List[str], int]]


class Gossiper(threading.Thread):
    """
    Gossiper.
    """

    ###
    # Init
    ###

    def __init__(
        self,
        self_addr,
        client: GrpcClient,  # can be generalized to any protocol
        period: float = Settings.GOSSIP_PERIOD,
        messages_per_period: int = Settings.GOSSIP_MESSAGES_PER_PERIOD,
    ) -> None:
        # Thread
        super().__init__()
        self.__self_addr = self_addr
        self.name = f"gossiper-thread-{self.__self_addr}"

        # Lists, locks and flag
        self.__processed_messages = []
        self.__processed_messages_lock = threading.Lock()
        self.__pending_msgs = []
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
        logger.info(self.__self_addr, "Starting gossiper...")
        return super().start()

    def stop(self) -> None:
        logger.info(self.__self_addr, "Stopping gossiper...")
        self.__gossip_terminate_flag.set()

    ###
    # Gossip
    ###

    def add_message(self, msg: any, pending_neis: List[str]) -> None:
        """
        Add to pending messages
        """
        self.__pending_msgs_lock.acquire()
        self.__pending_msgs.append((msg, pending_neis))
        self.__pending_msgs_lock.release()

    def check_and_set_processed(self, msg_hash: int) -> bool:
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
        while not self.__gossip_terminate_flag.is_set():
            t = time.time()
            messages_to_send = []
            messages_left = self.messages_per_period

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
                    self.__pending_msgs[0] = (
                        self.__pending_msgs[0][0],
                        self.__pending_msgs[0][1][messages_left:],
                    )

            # Unlock
            self.__pending_msgs_lock.release()

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
        get_candidates_fn,
        status_fn: StatusFunction,
        model_fn: ModelFunction,
        period: float,
        create_connection: bool,
    ) -> None:
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
                logger.info(self.__self_addr, "Gossip finished.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            logger.debug(self.__self_addr, f"Gossip remaining nodes: {neis}")
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
                        f"Gossiping exited for {Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS} equal rounds.",
                    )
                    logger.debug(
                        self.__self_addr, f"Gossip last status: {last_x_status[-1]}"
                    )
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
                logger.info(self.__self_addr, f"Gossiping model to {nei}.")
                self.__client.send(nei, model, create_connection=create_connection)

            # Sleep to allow periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)
