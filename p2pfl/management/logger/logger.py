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
from typing import Any

from p2pfl.experiment import Experiment
from p2pfl.management.message_storage import MessageEntryType, MessageStorage
from p2pfl.management.metric_storage import GlobalLogsType, GlobalMetricStorage, LocalLogsType, LocalMetricStorage
from p2pfl.management.node_monitor import NodeMonitor
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

    def __init__(self, nodes: dict[str, dict[str, Any]] | None = None, disable_locks: bool = False) -> None:
        """Initialize the logger."""
        # Node Information
        self._nodes: dict[str, dict[Any, Any]] = nodes if nodes else {}

        # Experiment Metrics and Message Storage
        self.disable_locks = disable_locks
        self.local_metrics = LocalMetricStorage(disable_locks=disable_locks)
        self.global_metrics = GlobalMetricStorage(disable_locks=disable_locks)
        self.message_storage = MessageStorage(disable_locks=disable_locks)
        self.node_monitor = NodeMonitor(report_fn=None)
        self.node_monitor.start()

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

    def connect(self, **kwargs: Any) -> None:
        """
        Establish connection/setup for the logger.

        This method should be overridden by loggers that require connection setup.
        By default, it does nothing.

        Args:
            **kwargs: Connection parameters specific to each logger type.

        """
        pass

    def cleanup(self) -> None:
        """Cleanup the logger."""
        # Unregister nodes
        for node in self._nodes.copy():
            self.unregister_node(node)

        # Remove handlers from the logger
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)

    def finish(self) -> None:
        """
        Finish any logging activities, like closing a W&B run.

        This method is a placeholder and is meant to be implemented by a decorator.
        By default, it does nothing.
        """
        pass

    ######
    # Application logging
    ######

    def set_level(self, level: int | str) -> None:
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

    def log_metric(self, addr: str, metric: str, value: float, step: int | None = None, round: int | None = None) -> None:
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
            # Node not registered, skip logging
            return

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
        if node in self._nodes:
            # Unregister the node
            self._nodes.pop(node)
        else:
            self.warning("SYSTEM", f"Attempted to unregister node {node} that was not registered.")

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

    def get_nodes(self) -> dict[str, dict[Any, Any]]:
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

    ######
    # Communication Logs
    ######

    def log_communication(
        self,
        node: str,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: int | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a communication event.

        Args:
            node: The node address.
            direction: Direction of communication ("sent" or "received").
            cmd: The command or message type.
            source_dest: Source (if receiving) or destination (if sending) node.
            package_type: Type of package ("message" or "weights").
            package_size: Size of the package in bytes (if available).
            round_num: The federated learning round number (if applicable).
            additional_info: Additional information as a dictionary.

        """
        # Determine emoji based on direction and package type
        emoji = ("📫" if package_type == "message" else "📦") if direction == "received" else ("📤" if package_type == "message" else "📬")

        # If round_num is not specified but we're in an experiment, get the current round
        if round_num is None or round_num < 0:
            try:
                # Look for the node in registered nodes
                if node in self._nodes and "Experiment" in self._nodes[node]:
                    experiment = self._nodes[node]["Experiment"]
                    if experiment is not None and hasattr(experiment, "round") and experiment.round is not None:
                        round_num = experiment.round
            except Exception:
                # If we can't get the round, just continue with default round_num (None)
                pass

        # Create base message
        message = f"{emoji} {cmd.upper()} {direction} "
        if direction == "received":
            message += f"from {source_dest}"
        else:
            message += f"to {source_dest}"

        # Add round information if available
        if round_num is not None and round_num >= 0:
            message += f" (round {round_num})"

        # Log the message at debug level
        if cmd != "beat" or (not Settings.heartbeat.EXCLUDE_BEAT_LOGS and cmd == "beat"):
            pass
            # self.debug(node, message)

        # Get actual round number for storage (default to 0 if None)
        storage_round = 0 if round_num is None or round_num < 0 else round_num

        # Store in message storage
        self.message_storage.add_message(
            node=node,
            direction=direction,
            cmd=cmd,
            source_dest=source_dest,
            package_type=package_type,
            package_size=package_size,
            round_num=storage_round,
            additional_info=additional_info,
        )

    def get_messages(
        self,
        direction: str = "all",  # "all", "sent", or "received"
        node: str | None = None,
        cmd: str | None = None,
        round_num: int | None = None,
        limit: int | None = None,
    ) -> list[MessageEntryType]:
        """
        Get communication messages with optional filtering.

        Args:
            direction: Filter by message direction ("all", "sent", or "received").
            node: Filter by node address (optional).
            cmd: Filter by command type (optional).
            round_num: Filter by round number (optional).
            limit: Limit the number of messages returned per node (optional).

        Returns:
            A flat list of message dictionaries. Each message includes a 'direction' field
            indicating whether it was 'sent' or 'received'.

        """
        # Validate direction
        if direction not in ["all", "sent", "received"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'all', 'sent', or 'received'.")

        # Convert "all" to None as expected by message_storage
        storage_direction = None if direction == "all" else direction

        return self.message_storage.get_messages(node=node, direction=storage_direction, cmd=cmd, round_num=round_num, limit=limit)

    def get_system_metrics(self) -> dict[datetime.datetime, dict[str, float]]:
        """
        Get the system metrics.

        Returns:
            The system metrics.

        """
        return self.node_monitor.get_logs()

    def reset(self) -> None:
        """
        Reset the logger state between experiments.

        This clears all stored metrics, messages, and system logs while keeping
        the logger configuration and handlers intact.
        """
        # Recreate storage instances to clear all data
        self.local_metrics = LocalMetricStorage(disable_locks=self.disable_locks)
        self.global_metrics = GlobalMetricStorage(disable_locks=self.disable_locks)
        self.message_storage = MessageStorage(disable_locks=self.disable_locks)

        # Clear system metrics and registered nodes
        self.node_monitor.logs.clear()
        self._nodes.clear()

        self.info("SYSTEM", "Logger state reset for new experiment")
