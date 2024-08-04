#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

"""Protocol agnostic heartbeater."""

import threading
import time
from typing import Optional

from p2pfl.communication.client import Client
from p2pfl.communication.neighbors import Neighbors
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

heartbeater_cmd_name = "beat"


class Heartbeater(threading.Thread):
    """
    Heartbeater for agnostic communication protocol. Send and update fresh heartbeats.

    Args:
        self_addr: Address of the node.
        neighbors: Neighbors to update.
        client: Client to send messages.

    """

    def __init__(self, self_addr: str, neighbors: Neighbors, client: Client) -> None:
        """Initialize the heartbeat thread."""
        super().__init__()
        self.__self_addr = self_addr
        self.__neighbors = neighbors
        self.__client = client
        self.__heartbeat_terminate_flag = threading.Event()
        self.daemon = True
        self.name = f"heartbeater-thread-{self.__self_addr}"

    def run(self) -> None:
        """Run the heartbeat thread."""
        self.__heartbeater()

    def stop(self) -> None:
        """Stop the heartbeat thread."""
        self.__heartbeat_terminate_flag.set()

    def beat(self, nei: str, time: float) -> None:
        """
        Update the time of the last heartbeat of a neighbor. If the neighbor is not added, add it.

        Args:
            nei: Address of the neighbor.
            time: Time of the heartbeat.

        """
        # Check if it is itself
        if nei == self.__self_addr:
            return

        # Check if exists
        self.__neighbors.refresh_or_add(nei, time)

    def __heartbeater(
        self,
        period: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if period is None:
            period = Settings.HEARTBEAT_PERIOD
        if timeout is None:
            timeout = Settings.HEARTBEAT_TIMEOUT
        toggle = False
        while not self.__heartbeat_terminate_flag.is_set():
            t = time.time()

            # Check heartbeats (every 2 periods)
            if toggle:
                # Get Neis
                neis = self.__neighbors.get_all()
                for nei in neis:
                    if t - neis[nei][2] > timeout:
                        logger.info(
                            self.__self_addr,
                            f"Heartbeat timeout for {nei} ({t - neis[nei][2]}). Removing...",
                        )
                        self.__neighbors.remove(nei)
            else:
                toggle = True

            # Send heartbeat
            beat_msg = self.__client.build_message(heartbeater_cmd_name, args=[str(time.time())])
            self.__client.broadcast(beat_msg)

            # Sleep to allow the periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)
