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
"""Ray logger decorator."""

import datetime
import logging
from typing import Any, Dict, Union

import ray

from p2pfl.experiment import Experiment
from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.management.metric_storage import GlobalLogsType, LocalLogsType


@ray.remote
class RayP2PFLoggerActor(LoggerDecorator):
    """Actor to add remote logging capabilities to a logger class."""

    pass


class RayP2PFLogger(P2PFLogger):
    """Wrapper to add remote logging capabilities to a logger class."""

    def __init__(self, p2pflogger: P2PFLogger):
        """
        Initialize the wrapper with a Ray actor instance.

        Args:
            p2pflogger: The logger to be wrapped.

        """
        self.ray_actor = RayP2PFLoggerActor.options(  # type: ignore
            name="p2pfl_logger", lifetime="detached", get_if_exists=True
        ).remote(p2pflogger)

    def connect_web(self, url: str, key: str) -> None:
        """
        Connect to the web services.

        Args:
            url: The URL of the web services.
            key: The API key.

        """
        self.ray_actor.connect_web.remote(url, key)

    def cleanup(self) -> None:
        """Cleanup the logger."""
        self.ray_actor.cleanup.remote()

    def set_level(self, level: Union[int, str]) -> None:
        """
        Set the logger level.

        Args:
            level: The logger level.

        """
        self.ray_actor.set_level.remote(level)

    def get_level(self) -> int:
        """
        Get the logger level.

        Returns
            The logger level.

        """
        return ray.get(self.ray_actor.get_level.remote())

    def get_level_name(self, lvl: int) -> str:
        """
        Get the logger level name.

        Args:
            lvl: The logger level.

        Returns:
            The logger level name.

        """
        return ray.get(self.ray_actor.get_level_name.remote(lvl))

    def log(self, level: int, node: str, message: str) -> None:
        """
        Log a message.

        Args:
            level: The log level.
            node: The node name.
            message: The message to log.

        """
        self.ray_actor.log.remote(level, node, message)

    def log_metric(self, addr: str, metric: str, value: float, round: int | None = None, step: int | None = None) -> None:
        """
        Log a metric.

        Args:
            addr: The node name.
            metric: The metric to log.
            value: The value.
            step: The step.
            round: The round.

        """
        self.ray_actor.log_metric.remote(addr, metric, value, round, step)

    def get_local_logs(self) -> LocalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return ray.get(self.ray_actor.get_local_logs.remote())

    def get_global_logs(self) -> GlobalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return ray.get(self.ray_actor.get_global_logs.remote())

    def register_node(self, node: str, simulation: bool) -> None:
        """
        Register a node.

        Args:
            node: The node address.
            simulation: If the node is a simulation.

        """
        self.ray_actor.register_node.remote(node, simulation)

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        self.ray_actor.unregister_node.remote(node)

    def experiment_started(self, node: str, experiment: Experiment | None) -> None:
        """
        Notify the experiment start.

        Args:
            node: The node address.
            experiment: The experiment.

        """
        self.ray_actor.experiment_started.remote(node, experiment)

    def experiment_finished(self, node: str) -> None:
        """
        Notify the experiment end.

        Args:
            node: The node address.

        """
        self.ray_actor.experiment_finished.remote(node)

    def round_started(self, node: str, experiment: Experiment | None) -> None:
        """
        Notify the round start.

        Args:
            node: The node address.
            experiment: The experiment.

        """
        self.ray_actor.round_started.remote(node, experiment)

    def round_finished(self, node: str) -> None:
        """
        Notify the round end.

        Args:
            node: The node address.

        """
        self.ray_actor.round_finished.remote(node)

    def get_nodes(self) -> Dict[str, Dict[Any, Any]]:
        """
        Get the registered nodes.

        Returns:
            The registered nodes.

        """
        return ray.get(self.ray_actor.get_nodes.remote())

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add a handler to the logger.

        Args:
            handler: The handler to add.

        """
        self.ray_actor.add_handler.remote(handler)

    def log_system_metric(self, node: str, metric: str, value: float, time: datetime.datetime) -> None:
        """
        Log a system metric.

        Args:
            node: The node name.
            metric: The metric to log.
            value: The value.
            time: The time.

        """
        self.ray_actor.log_system_metric.remote(node, metric, value, time)
