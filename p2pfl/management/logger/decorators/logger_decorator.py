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

"""Decorator for the logger to be used in the simulation."""

import datetime
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from p2pfl.experiment import Experiment
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.management.message_storage import MessageEntryType
from p2pfl.management.metric_storage import GlobalLogsType, LocalLogsType


class LoggerDecorator(P2PFLogger):
    """
    Decorator class for logging. Works by wrapping the a previous logger, so new funcionalities can be added.

    By default, it does nothing, just delegates the calls to the wrapped logger.
    """

    def __init__(self, logger: P2PFLogger | Callable[[], P2PFLogger]) -> None:
        """
        Initialize the logger.

        Args:
            logger: The logger to wrap.

        """
        self._p2pfl_logger = logger() if callable(logger) else logger

    def connect_web(self, url: str, key: str) -> None:
        """
        Connect to the web services.

        Args:
            url: The URL of the web services.
            key: The API key.

        """
        self._p2pfl_logger.connect_web(url, key)

    def cleanup(self) -> None:
        """Cleanup the logger."""
        self._p2pfl_logger.cleanup()

    def set_level(self, level: Union[int, str]) -> None:
        """
        Set the logger level.

        Args:
            level: The logger level.

        """
        self._p2pfl_logger.set_level(level)

    def get_level(self) -> int:
        """
        Get the logger level.

        Returns
            The logger level.

        """
        return self._p2pfl_logger.get_level()

    def get_level_name(self, lvl: int) -> str:
        """
        Get the logger level name.

        Args:
            lvl: The logger level.

        Returns:
            The logger level name.

        """
        return self._p2pfl_logger.get_level_name(lvl)

    def log(self, level: int, node: str, message: str) -> None:
        """
        Log a message.

        Args:
            level: The log level.
            node: The node name.
            message: The message to log.

        """
        self._p2pfl_logger.log(level, node, message)

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
        self._p2pfl_logger.log_metric(addr=addr, metric=metric, value=value, step=step, round=round)

    def get_local_logs(self) -> LocalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self._p2pfl_logger.get_local_logs()

    def get_global_logs(self) -> GlobalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self._p2pfl_logger.get_global_logs()

    def register_node(self, node: str) -> None:
        """
        Register a node.

        Args:
            node: The node address.

        """
        self._p2pfl_logger.register_node(node)

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        self._p2pfl_logger.unregister_node(node)

    def experiment_started(self, node: str, experiment: Experiment) -> None:
        """
        Notify the experiment start.

        Args:
            node: The node address.
            experiment: The experiment.

        """
        self._p2pfl_logger.experiment_started(node, experiment)

    def experiment_updated(self, node: str, experiment: Experiment) -> None:
        """
        Notify the round end.

        Args:
            node: The node address.
            experiment: The experiment to update.

        """
        self._p2pfl_logger.experiment_updated(node, experiment)

    def experiment_finished(self, node: str) -> None:
        """
        Notify the experiment end.

        Args:
            node: The node address.

        """
        self._p2pfl_logger.experiment_finished(node)

    def get_nodes(self) -> Dict[str, Dict[Any, Any]]:
        """
        Get the registered nodes.

        Returns:
            The registered nodes.

        """
        return self._p2pfl_logger.get_nodes()

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add a handler to the logger.

        Args:
            handler: The handler to add.

        """
        self._p2pfl_logger.add_handler(handler)

    def log_communication(
        self,
        node: str,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None,
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
        self._p2pfl_logger.log_communication(
            node=node,
            direction=direction,
            cmd=cmd,
            source_dest=source_dest,
            package_type=package_type,
            package_size=package_size,
            round_num=round_num,
            additional_info=additional_info,
        )

    def get_messages(
        self,
        direction: str = "all",
        node: Optional[str] = None,
        cmd: Optional[str] = None,
        round_num: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[MessageEntryType]:
        """
        Get communication messages with optional filtering.

        Args:
            direction: Filter by message direction ("all", "sent", or "received").
            node: Filter by node address (optional).
            cmd: Filter by command type (optional).
            round_num: Filter by round number (optional).
            limit: Limit the number of messages returned per node (optional).

        Returns:
            Messages matching the filters.

        """
        return self._p2pfl_logger.get_messages(direction=direction, node=node, cmd=cmd, round_num=round_num, limit=limit)

    def get_system_metrics(self) -> Dict[datetime.datetime, Dict[str, float]]:
        """
        Get the system metrics.

        Returns:
            The system metrics.

        """
        return self._p2pfl_logger.get_system_metrics()
