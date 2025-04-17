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

"""
P2PFL Logger.

.. note:: Not all is typed because the python logger is not typed (yep, is a TODO...).

"""

import copy
import datetime
import logging
from typing import Any, Dict, Optional, Union

from p2pfl.experiment import Experiment
from p2pfl.management.metric_storage import GlobalLogsType, GlobalMetricStorage, LocalLogsType, LocalMetricStorage
from p2pfl.settings import Settings

###################
#    Exception    #
###################


class NodeNotRegistered(Exception):
    """Exception raised when a node is not registered."""

    pass


#########################
#    Colored logging    #
#########################

# COLORS
GRAY = "\033[90m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to the log messages."""

    def format(self, record):
        """
        Format the log record with color.

        Args:
            record: The log record.

        """
        # Copy the original record
        record_copy = copy.copy(record)

        # Warn level color
        if record_copy.levelname == "DEBUG":
            record_copy.levelname = BLUE + record_copy.levelname + RESET
        elif record_copy.levelname == "INFO":
            record_copy.levelname = GREEN + record_copy.levelname + RESET
        elif record_copy.levelname == "WARNING":
            record_copy.levelname = YELLOW + record_copy.levelname + RESET
        elif record_copy.levelname == "ERROR" or record_copy.levelname == "CRITICAL":
            record_copy.levelname = RED + record_copy.levelname + RESET
        return super().format(record_copy)


################
#    Logger    #
################


class P2PFLogger:
    """
    Class that manages the node logging (not a singleton).

    Args:
        p2pfl_web_services: The P2PFL Web Services to log and monitor the nodes remotely.

    """

    def __init__(self, nodes: Optional[Dict[str, Dict[str, Any]]] = None, disable_locks: bool = False) -> None:
        """Initialize the logger."""
        # Node Information
        self._nodes: Dict[str, Dict[Any, Any]] = nodes if nodes else {}

        # Experiment Metrics
        self.disable_locks = disable_locks
        self.local_metrics = LocalMetricStorage(disable_locks=disable_locks)
        self.global_metrics = GlobalMetricStorage(disable_locks=disable_locks)

        # Python logging
        self._logger = logging.getLogger("p2pfl")
        if self._logger.handlers != []:
            print(self._logger.handlers)
            raise Exception("Logger already initialized.")
        self._logger.propagate = False
        self._logger.setLevel(logging.getLevelName(Settings.general.LOG_LEVEL))

        # STDOUT - Handler
        stream_handler = logging.StreamHandler()
        cmd_formatter = ColoredFormatter(
            f"{GRAY}[ {YELLOW}%(asctime)s {GRAY}| {CYAN}%(node)s {GRAY}| %(levelname)s{GRAY} ]{RESET} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(cmd_formatter)
        self._logger.addHandler(stream_handler)  # not async

    def connect_web(self, url: str, key: str) -> None:
        """
        Connect to the web services.

        Args:
            url: The URL of the web services.
            key: The API key.

        """
        raise NotImplementedError("Web Services not implemented.")

    def cleanup(self) -> None:
        """Cleanup the logger."""
        # Unregister nodes
        for node in self._nodes.copy():
            self.unregister_node(node)

        # Remove handlers from the logger
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)

    ######
    # Application logging
    ######

    def set_level(self, level: Union[int, str]) -> None:
        """
        Set the logger level.

        Args:
            level: The logger level.

        """
        if isinstance(level, str):
            self._logger.setLevel(logging.getLevelName(level))
        else:
            self._logger.setLevel(level)

    def get_level(self) -> int:
        """
        Get the logger level.

        Returns
            The logger level.

        """
        return self._logger.getEffectiveLevel()

    def get_level_name(self, lvl: int) -> str:
        """
        Get the logger level name.

        Args:
            lvl: The logger level.

        Returns:
            The logger level name.

        """
        return logging.getLevelName(lvl)

    def info(self, node: str, message: str) -> None:
        """
        Log an info message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.INFO, node, message)

    def debug(self, node: str, message: str) -> None:
        """
        Log a debug message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.DEBUG, node, message)

    def warning(self, node: str, message: str) -> None:
        """
        Log a warning message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.WARNING, node, message)

    def error(self, node: str, message: str) -> None:
        """
        Log an error message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.ERROR, node, message)

    def critical(self, node: str, message: str) -> None:
        """
        Log a critical message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.CRITICAL, node, message)

    def log(self, level: int, node: str, message: str) -> None:
        """
        Log a message.

        Args:
            level: The log level.
            node: The node name.
            message: The message to log.

        """
        # Traditional logging
        if level == logging.DEBUG:
            self._logger.debug(message, extra={"node": node})
        elif level == logging.INFO:
            self._logger.info(message, extra={"node": node})
        elif level == logging.WARNING:
            self._logger.warning(message, extra={"node": node})
        elif level == logging.ERROR:
            self._logger.error(message, extra={"node": node})
        elif level == logging.CRITICAL:
            self._logger.critical(message, extra={"node": node})
        else:
            raise ValueError(f"Invalid level: {level}")

    ######
    # Metrics
    ######

    def log_metric(self, addr: str, metric: str, value: float, step: Optional[int] = None, round: Optional[int] = None) -> None:
        """
        Log a metric.

        Args:
            addr: The node name.
            metric: The metric to log.
            value: The value.
            step: The step.
            round: The round.

        """
        # Get Experiment
        try:
            experiment = self._nodes[addr]["Experiment"]
        except KeyError:
            # print(f"Node {addr} not registered.")
            return
            raise NodeNotRegistered(f"Node {addr} not registered.") from None

        # Get Round
        if round is None:
            round = experiment.round
            if round is None:
                raise Exception("No round provided. Needed for training metrics.")

        # Get Experiment Name
        exp = experiment.exp_name
        if exp is None:
            raise Exception("No experiment name provided. Needed for training metrics.")

        # Local storage
        if step is None:
            # Global Metrics
            self.global_metrics.add_log(exp, round, metric, addr, value)
        else:
            # Local Metrics
            self.local_metrics.add_log(exp, round, metric, addr, value, step)

    def get_local_logs(self) -> LocalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self.local_metrics.get_all_logs()

    def get_global_logs(self) -> GlobalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self.global_metrics.get_all_logs()

    ######
    # Node registration
    ######

    def register_node(self, node: str) -> None:
        """
        Register a node.

        Args:
            node: The node address.

        """
        # Node State
        if self._nodes.get(node) is None:
            # Dict[str, Dict[str,Any]]
            self._nodes[node] = {}
        else:
            raise Exception(f"Node {node} already registered.")

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        # Node state
        n = self._nodes[node]
        if n is not None:
            # Unregister the node
            self._nodes.pop(node)
        else:
            raise Exception(f"Node {node} not registered.")

    ######
    # Node Status
    ######

    def experiment_started(self, node: str, experiment: Experiment) -> None:
        """
        Notify the experiment start.

        Args:
            node: The node address.
            experiment: The experiment.

        """
        self._nodes[node]["Experiment"] = experiment

    def experiment_updated(self, node: str, experiment: Experiment) -> None:
        """
        Notify the round end.

        Args:
            node: The node address.
            experiment: The experiment to update.

        """
        self.warning(node, "Uncatched Round Finished on Logger")
        if self._nodes[node]["Experiment"] is not None:
            self._nodes[node]["Experiment"] = experiment
        else:
            raise Exception(f"Node {node} has no experiment.")

    def experiment_finished(self, node: str) -> None:
        """
        Notify the experiment end.

        Args:
            node: The node address.

        """
        self.warning(node, "Uncatched Experiment Ended on Logger")
        del self._nodes[node]["Experiment"]

    def get_nodes(self) -> Dict[str, Dict[Any, Any]]:
        """
        Get the registered nodes.

        Returns:
            The registered nodes.

        """
        return self._nodes

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add a handler to the logger.

        Args:
            handler: The logger handler.

        """
        self._logger.addHandler(handler)

    def log_system_metric(self, node: str, metric: str, value: float, time: datetime.datetime) -> None:
        """
        Log a system metric.

        Args:
            node: The node name.
            metric: The metric to log.
            value: The value.
            time: The time.

        """
        pass
