#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

import datetime
import threading
import time
from typing import Dict
import psutil
from p2pfl.settings import Settings


class NodeMonitor(threading.Thread):

    def __init__(self, node_addr, metric_report_callback) -> None:
        self.node_addr = node_addr
        self.metric_report_callback = metric_report_callback
        self.period = Settings.RESOURCE_MONITOR_PERIOD
        self.last_net_in = -1
        self.last_net_out = -1
        # Super
        super().__init__()
        self.name = "resource-monitor-thread-" + self.node_addr
        self.daemon = True

    def run(self) -> None:
        while True:
            # Sys Resources
            time_now = datetime.datetime.now()
            for key, value in self.report_system_resources().items():
                self.metric_report_callback(self.node_addr, key, value, time_now)
            time.sleep(self.period)

    def report_system_resources(self) -> Dict[str, float]:
        res = {}
        # CPU
        res["cpu"] = psutil.cpu_percent()
        # RAM
        res["ram"] = psutil.virtual_memory().percent
        # NetWork
        net_stat = psutil.net_io_counters()
        if self.last_net_in != -1 and self.last_net_out != -1:
            res["net_in"] = (
                (net_stat.bytes_recv - self.last_net_in) / self.period / (2**20)
            )
            res["net_out"] = (
                (net_stat.bytes_sent - self.last_net_out) / self.period / (2**20)
            )
        self.last_net_in = net_stat.bytes_recv
        self.last_net_out = net_stat.bytes_sent

        return res

    def report_status(self):
        # USAR EN UN FUTURO UNA STATUS FUNCTION PROPORCIONADA X EL NODO
        pass
