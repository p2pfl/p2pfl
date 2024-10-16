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
from p2pfl.experiment import Experiment
from typing import Any, Dict, List, Optional

from p2pfl.management.metric_storage import GlobalLogsType, LocalLogsType

###################
#    Interface    #
###################

from abc import ABC, abstractmethod

class P2PFLogger(ABC):
    ######
    # Singleton and instance management
    ######

    _logger: logging.Logger = None
    _nodes: Dict[str, Dict[Any,Any]]
    _handlers: List[logging.Handler] = []

    ######
    # Getters and setters
    ######

    @abstractmethod
    def get_logger(self) -> None:
        pass

    @abstractmethod
    def get_nodes(self) -> Dict[str, Dict[Any,Any]]:
        pass

    @abstractmethod
    def get_handlers(self) -> List[logging.Handler]:
        pass

    @abstractmethod
    def set_logger(self, logger: logging.Logger) -> None:
        pass

    @abstractmethod
    def set_nodes(self, nodes: Dict[str, Dict[Any,Any]]) -> None:
        pass

    @abstractmethod
    def set_handlers(self, handler: List[logging.Handler]) -> None:
        pass


    ######
    # Application logging
    ######

    @abstractmethod
    def set_level(self, level: int) -> None:
        pass

    @abstractmethod
    def get_level(self) -> int:
        pass

    @abstractmethod
    def get_level_name(self, lvl: int) -> str:
        pass

    @abstractmethod
    def info(self, node: str, message: str) -> None:
        pass

    @abstractmethod
    def debug(self, node: str, message: str) -> None:
        pass

    @abstractmethod
    def warning(self, node: str, message: str) -> None:
        pass

    @abstractmethod
    def error(self, node: str, message: str) -> None:
        pass

    @abstractmethod
    def critical(self, node: str, message: str) -> None:
        pass

    @abstractmethod
    def log(self, level: int, node: str, message: str) -> None:
        pass

    ######
    # Metrics
    ######

    @abstractmethod
    def log_metric(self, addr: str, metric: str,
                   value: float, round: Optional[int] = None,
                   step: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def get_local_logs(self) -> LocalLogsType:
        pass

    @abstractmethod
    def get_global_logs(self) -> GlobalLogsType:
        pass

    ######
    # Node registration
    ######

    @abstractmethod
    def register_node(self, node: str, simulation: bool) -> None:
        pass

    @abstractmethod
    def unregister_node(self, node: str) -> None:
        pass


    ######
    # Node Status
    ######

    @abstractmethod
    def experiment_started(self, node: str, experiment:Experiment) -> None:
        pass

    @abstractmethod
    def experiment_finished(self, node: str) -> None:
        pass

    @abstractmethod
    def round_started(self, node: str, experiment:Experiment) -> None:
        pass

    @abstractmethod
    def round_finished(self, node: str) -> None:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

    ######
    # Handlers
    ######
    @abstractmethod
    def add_handler(self, handler: logging.Handler) -> None:
        pass


###################
#    Exception    #
###################

class NodeNotRegistered(Exception):
    """Exception raised when a node is not registered."""

    pass