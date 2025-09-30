#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

"""Node monitor."""

import datetime
import threading
import time

import psutil  # type: ignore

from p2pfl.settings import Settings


class NodeMonitor(threading.Thread):
    """
    Node monitor thread.

    Args:
        node_addr: Node address.
        metric_report_callback: Metric report callback.

    """

    def __init__(self, report_fn) -> None:
        """Initialize the node monitor."""
        self.report_fn = report_fn
        self.period = Settings.general.RESOURCE_MONITOR_PERIOD
        self.running = True
        # Logs
        self.logs: dict[datetime.datetime, dict[str, float]] = {}
        # Super
        super().__init__()
        self.name = "resource-monitor-thread"
        self.daemon = True

    def set_report_fn(self, report_fn) -> None:
        """Set the report function."""
        self.report_fn = report_fn

    def get_logs(self) -> dict[datetime.datetime, dict[str, float]]:
        """Get the logs."""
        return self.logs

    def stop(self) -> None:
        """Stop the node monitor."""
        self.running = False

    def run(self) -> None:
        """Run the node monitor."""
        while self.running:
            # Sys Resources
            time_now = datetime.datetime.now()
            resources = self.__report_system_resources()
            # Update logs
            self.logs.update({time_now: resources})
            # Report
            if self.report_fn:
                for key, value in resources.items():
                    self.report_fn(key, value, time_now)
            time.sleep(self.period)

    def __report_system_resources(self) -> dict[str, float]:
        """Report the system resources."""
        res = {}
        # CPU
        res["cpu"] = psutil.cpu_percent()
        # RAM
        res["ram"] = psutil.virtual_memory().percent
        return res
