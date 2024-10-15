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
import logging
from typing import List, Tuple

from p2pfl.management.logger.logger import *
from p2pfl.management.node_monitor import NodeMonitor
from p2pfl.management.p2pfl_web_services import P2pflWebServices

#########################################
#    Logging handler (transmit logs)    #
#########################################

class DictFormatter(logging.Formatter):
    """Formatter (logging) that returns a dictionary with the log record attributes."""

    def format(self, record):
        """
        Format the log record as a dictionary.

        Args:
            record: The log record.

        """
        # Get node
        if not hasattr(record, "node"):
            raise ValueError("The log record must have a 'node' attribute.")
        log_dict = {
            "timestamp": datetime.datetime.fromtimestamp(record.created),
            "level": record.levelname,
            "node": record.node,  # type: ignore
            "message": record.getMessage(),
        }
        return log_dict
    
class P2pflWebLogHandler(logging.Handler):
    """
    Custom logging handler that sends log entries to the API.

    Args:
        p2pfl_web: The P2PFL Web Services.

    """

    def __init__(self, p2pfl_web: P2pflWebServices):
        """Initialize the handler."""
        super().__init__()
        self.p2pfl_web = p2pfl_web
        self.formatter = DictFormatter()  # Instantiate the custom formatter

    def emit(self, record):
        """
        Emit the log record.

        Args:
            record: The log record.

        """
        # Format the log record using the custom formatter
        log_message = self.formatter.format(record)  # type: ignore
        # Send log entry to the API
        self.p2pfl_web.send_log(
            log_message["timestamp"],  # type: ignore
            log_message["node"],  # type: ignore
            log_message["level"],  # type: ignore
            log_message["message"],  # type: ignore
        )


class WebLocalLogger(P2PFLogger):
    _p2pflogger: P2PFLogger = None

    def __init__(self, p2pflogger: P2PFLogger, p2pfl_web_services: P2pflWebServices):
        self._logger = p2pflogger._logger
        self._p2pflogger = p2pflogger
        self.p2pfl_web_services = p2pfl_web_services

        # Setup the web handler for the provided logger instance
        web_handler = P2pflWebLogHandler(self.p2pfl_web_services)
        self._logger.addHandler(web_handler)

    def connect_web(self, url: str, key: str) -> None:
        """
        Connect to the web services.

        Args:
            url: The URL of the web services.
            key: The API key.

        """
        # Create the instance
        p2pfl_web = P2pflWebServices(url, key)

        # P2PFL Web Services
        self.p2pfl_web_services = p2pfl_web
        if p2pfl_web is not None:
            web_handler = P2pflWebLogHandler(p2pfl_web)
            self._logger.addHandler(web_handler)

    def info(self, node: str, message: str) -> None:
        self._p2pflogger.info(node, message)

    def debug(self, node: str, message: str) -> None:
        self._p2pflogger.debug(node, message)

    def warning(self, node: str, message: str) -> None:
        self._p2pflogger.warning(node, message)

    def error(self, node: str, message: str) -> None:
        self._p2pflogger.error(node, message)

    def critical(self, node: str, message: str) -> None:
        self._p2pflogger.critical(node, message)

    def log_metric(self, addr: str, metric: str,
                   value: float, round: int | None = None,
                   step: int | None = None) -> None:
        self._p2pflogger.log_metric(addr, metric, value, round, step)

        # Get Experiment
        try:
            experiment = self._nodes[addr]["Experiment"]
        except KeyError:
            raise NodeNotRegistered(f"Node {addr} not registered.")

        if self.p2pfl_web_services is not None:
            if step is None:
                # Global Metrics
                self.p2pfl_web_services.send_global_metric(experiment.exp_name, experiment.round, metric, addr, value)
            else:
                # Local Metrics
                self.p2pfl_web_services.send_local_metric(experiment.exp_name, experiment.round, metric, addr, value, step)

    def log_system_metric(self, node: str, metric: str, value: float, time: datetime.datetime) -> None:
        """
        Log a system metric. Only on web.

        Args:
            node: The node name.
            metric: The metric to log.
            value: The value.
            time: The time.

        """
        # Web
        if self.p2pfl_web_services is not None:
            self.p2pfl_web_services.send_system_metric(node, metric, value, time)

    def get_local_logs(self) -> Dict[str, Dict[int, Dict[str, Dict[str, List[Tuple[int | float]]]]]]:
        return self._p2pflogger.get_local_logs()

    def get_global_logs(self) -> Dict[str, Dict[str, Dict[str, List[Tuple[int | float]]]]]:
        return self._p2pflogger.get_global_logs()

    def register_node(self, node: str, simulation: bool) -> None:
        self._p2pflogger.register_node(node, simulation)

        # Register the node
        self.p2pfl_web_services.register_node(node, simulation)

        # Start the node status reporter
        node_monitor = NodeMonitor(node, self.log_system_metric)
        node_monitor.start()

        # Dict[str, Dict[str, Any]]
        self._p2pflogger._nodes[node]["NodeMonitor"] = node_monitor

    def unregister_node(self, node: str) -> None:
        # Web
        if self.p2pfl_web_services is not None:
            self.p2pfl_web_services.unregister_node(node)

        # Node state
        n = self._p2pflogger._nodes[node]
        if n is not None:
            # Stop the node status reporter
            if "NodeMonitor" in n:
                n["NodeMonitor"].stop()
        else:
            raise Exception(f"Node {node} not registered.")

        self._p2pflogger.unregister_node(node)

    def cleanup(self) -> None:
        self._p2pflogger.cleanup()

    def get_level_name(self, lvl: int) -> str:
        return self._p2pflogger.get_level_name(lvl)

    def set_level(self, level: int) -> None:
        self._p2pflogger.set_level(level)

    def get_level(self) -> int:
        return self._p2pflogger.get_level()

    def log(self, level: int, node: str, message: str) -> None:
        self._p2pflogger.log(level, node, message)

    def experiment_started(self, node: str, experiment: Experiment) -> None:
        """
        Notify the experiment start.

        Args:
            node: The node address.

        """
        self._p2pflogger.experiment_started(node,experiment)

    def experiment_finished(self, node: str) -> None:
        """
        Notify the experiment end.

        Args:
            node: The node address.

        """
        self._p2pflogger.experiment_finished(node)

    def round_started(self, node: str, experiment: Experiment) -> None:
        """
        Notify the round start.

        Args:
            node: The node address.

        """
        self._p2pflogger.round_started(node,experiment)

    def round_finished(self, node: str) -> None:
        """
        Notify the round end.

        Args:
            node: The node address.

        """
        self._p2pflogger.round_finished(node)

