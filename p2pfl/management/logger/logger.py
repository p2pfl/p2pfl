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

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from p2pfl.experiment import Experiment
from p2pfl.management.metric_storage import GlobalLogsType, LocalLogsType


class P2PFLogger(ABC):
    """Interface for the P2PFL Logger."""

    ######
    # Singleton and instance management
    ######

    _logger: logging.Logger
    _nodes: Dict[str, Dict[Any, Any]]
    _handlers: List[logging.Handler] = []

    ######
    # Getters and setters
    ######

    @abstractmethod
    def get_logger(self) -> logging.Logger:
        """
        Get the logger instance.

        Returns:
            The logger instance.

        """
        pass

    @abstractmethod
    def get_nodes(self) -> Dict[str, Dict[Any, Any]]:
        """
        Get the registered nodes.

        Returns:
            The registered nodes.

        """
        pass

    @abstractmethod
    def get_handlers(self) -> List[logging.Handler]:
        """
        Get the logger handlers.

        Returns:
            The logger handlers.

        """
        pass

    @abstractmethod
    def set_logger(self, logger: logging.Logger) -> None:
        """
        Set the logger instance.

        Args:
            logger: The logger instance.

        """
        pass

    @abstractmethod
    def set_nodes(self, nodes: Dict[str, Dict[Any, Any]]) -> None:
        """
        Set the registered nodes.

        Args:
            nodes: The registered nodes.

        """
        pass

    @abstractmethod
    def set_handlers(self, handler: List[logging.Handler]) -> None:
        """
        Set the logger handlers.

        Args:
            handler: The logger handlers.

        """
        pass

    ######
    # Application logging
    ######

    @abstractmethod
    def set_level(self, level: Union[int, str]) -> None:
        """
        Set the logger level.

        Args:
            level: The logger level.

        """
        pass

    @abstractmethod
    def get_level(self) -> int:
        """
        Get the logger level.

        Returns
            The logger level.

        """
        pass

    @abstractmethod
    def get_level_name(self, lvl: int) -> str:
        """
        Get the logger level name.

        Args:
            lvl: The logger level.

        Returns:
            The logger level name.

        """
        pass

    @abstractmethod
    def info(self, node: str, message: str) -> None:
        """
        Log an info message.

        Args:
            node: The node name.
            message: The message to log.

        """
        pass

    @abstractmethod
    def debug(self, node: str, message: str) -> None:
        """
        Log a debug message.

        Args:
            node: The node name.
            message: The message to log.

        """
        pass

    @abstractmethod
    def warning(self, node: str, message: str) -> None:
        """
        Log a warning message.

        Args:
            node: The node name.
            message: The message to log.

        """
        pass

    @abstractmethod
    def error(self, node: str, message: str) -> None:
        """
        Log an error message.

        Args:
            node: The node name.
            message: The message to log.

        """
        pass

    @abstractmethod
    def critical(self, node: str, message: str) -> None:
        """
        Log a critical message.

        Args:
            node: The node name.
            message: The message to log.

        """
        pass

    @abstractmethod
    def log(self, level: int, node: str, message: str) -> None:
        """
        Log a message.

        Args:
            level: The log level.
            node: The node name.
            message: The message to log.

        """
        pass

    ######
    # Metrics
    ######

    @abstractmethod
    def log_metric(self, addr: str, metric: str, value: float, round: Optional[int] = None, step: Optional[int] = None) -> None:
        """
        Log a metric.

        Args:
            addr: The node name.
            metric: The metric to log.
            value: The value.
            step: The step.
            round: The round.

        """
        pass

    @abstractmethod
    def get_local_logs(self) -> LocalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        pass

    @abstractmethod
    def get_global_logs(self) -> GlobalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        pass

    ######
    # Node registration
    ######

    @abstractmethod
    def register_node(self, node: str, simulation: bool) -> None:
        """
        Register a node.

        Args:
            node: The node address.
            simulation: If the node is a simulation.

        """
        pass

    @abstractmethod
    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        pass

    ######
    # Node Status
    ######

    @abstractmethod
    def experiment_started(self, node: str, experiment: Experiment | None) -> None:
        """
        Notify the experiment start.

        Args:
            node: The node address.
            experiment: The experiment.

        """
        pass

    @abstractmethod
    def experiment_finished(self, node: str) -> None:
        """
        Notify the experiment end.

        Args:
            node: The node address.

        """
        pass

    @abstractmethod
    def round_started(self, node: str, experiment: Experiment | None) -> None:
        """
        Notify the round start.

        Args:
            node: The node address.
            experiment: The experiment.

        """
        pass

    @abstractmethod
    def round_finished(self, node: str) -> None:
        """
        Notify the round end.

        Args:
            node: The node address.

        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the logger."""
        pass

    ######
    # Handlers
    ######
    @abstractmethod
    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add a handler to the logger.

        Args:
            handler: The handler to add.

        """
        pass


###################
#    Exception    #
###################


class NodeNotRegistered(Exception):
    """Exception raised when a node is not registered."""

    pass
