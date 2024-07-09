import threading
import time

from p2pfl.communication.client import Client
from p2pfl.communication.neighbors import Neighbors
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

heartbeater_cmd_name = "beat"


class Heartbeater(threading.Thread):
    def __init__(
        self, self_addr: str, neighbors: Neighbors, client: Client
    ) -> None:
        """
        Initialize the heartbeat thread.

        Args:
            self_addr (str): Address of the node.
            neighbors (dict): Dictionary of neighbors.
        """
        super().__init__()
        self.__self_addr = self_addr
        self.__neighbors = neighbors
        self.__client = client
        self.__heartbeat_terminate_flag = threading.Event()
        self.daemon = True
        self.name = f"heartbeater-thread-{self.__self_addr}"

    def run(self) -> None:
        self.__heartbeater()

    def stop(self) -> None:
        self.__heartbeat_terminate_flag.set()

    def beat(self, nei: str, time: float) -> None:
        """
        Update the time of the last heartbeat of a neighbor. If the neighbor is not added, add it.

        Args:
            nei (str): Address of the neighbor.
            time (float): Time of the heartbeat.
        """
        # Check if it is itself
        if nei == self.__self_addr:
            return

        # Check if exists
        self.__neighbors.refresh_or_add(nei, time)

    def __heartbeater(
        self,
        period: float = Settings.HEARTBEAT_PERIOD,
        timeout: float = Settings.HEARTBEAT_TIMEOUT,
    ) -> None:
        toggle = False
        while not self.__heartbeat_terminate_flag.is_set():
            t = time.time()

            # Check heartbeats (every 2 periods)
            if toggle:
                # Get Neis
                neis = self.__neighbors.get_all()
                for nei in neis.keys():
                    if t - neis[nei][2] > timeout:
                        logger.info(
                            self.__self_addr,
                            f"Heartbeat timeout for {nei} ({t - neis[nei][2]}). Removing...",
                        )
                        self.__neighbors.remove(nei)
            else:
                toggle = True

            # Send heartbeat
            beat_msg = self.__client.build_message(
                heartbeater_cmd_name, args=[str(time.time())]
            )
            self.__client.broadcast(beat_msg)

            # Sleep to allow the periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)
