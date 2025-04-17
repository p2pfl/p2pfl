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
from typing import Any, Callable, Dict, Optional, Union

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

    def __init__(self, logger: P2PFLogger | Callable[[], P2PFLogger]) -> None:
        """
        Initialize the wrapper with a Ray actor instance.

        Args:
            logger: The logger to be wrapped.

        """
        self.ray_actor = RayP2PFLoggerActor.options(  # type: ignore
            name="p2pfl_ray_logger", lifetime="detached", get_if_exists=True
        ).remote(logger)

    @staticmethod
    def from_actor(actor: RayP2PFLoggerActor) -> "RayP2PFLogger":
        """
        Initialize the wrapper with an existing Ray actor instance.

        Args:
            actor: The RayP2PFLoggerActor instance.

        Returns:
            A RayP2PFLogger instance wrapping the given actor.

        """
        # Create instance without calling __init__
        instance = RayP2PFLogger.__new__(RayP2PFLogger)
        instance.ray_actor = actor
        return instance

    def connect_web(self, url: str, key: str) -> None:
        """
        Connect to the web services.

        Args:
            url: The URL of the web services.
            key: The API key.

        """
        ray.get(self.ray_actor.connect_web.remote(url, key))

    def cleanup(self) -> None:
        """Cleanup the logger."""
        ray.get(self.ray_actor.cleanup.remote())

    def set_level(self, level: Union[int, str]) -> None:
        """
        Set the logger level.

        Args:
            level: The logger level.

        """
        # Set ray log level
        ray.get(self.ray_actor.set_level.remote(level))

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
        ray.get(self.ray_actor.log.remote(level, node, message))

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
        ray.get(self.ray_actor.log_metric.remote(addr=addr, metric=metric, value=value, step=step, round=round))

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

    def register_node(self, node: str) -> None:
        """
        Register a node.

        Args:
            node: The node address.

        """
        ray.get(self.ray_actor.register_node.remote(node))

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        ray.get(self.ray_actor.unregister_node.remote(node))

    def experiment_started(self, node: str, experiment: Experiment | None) -> None:
        """
        Notify the experiment start.

        Args:
            node: The node address.
            experiment: The experiment.

        """
        ray.get(self.ray_actor.experiment_started.remote(node, experiment))

    def experiment_finished(self, node: str) -> None:
        """
        Notify the experiment end.

        Args:
            node: The node address.

        """
        ray.get(self.ray_actor.experiment_finished.remote(node))

    def experiment_updated(self, node: str, experiment: Experiment) -> None:
        """
        Notify the round end.

        Args:
            node: The node address.
            experiment: The experiment to update.

        """
        ray.get(self.ray_actor.experiment_updated.remote(node, experiment))

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
        ray.get(self.ray_actor.add_handler.remote(handler))

    def log_system_metric(self, node: str, metric: str, value: float, time: datetime.datetime) -> None:
        """
        Log a system metric.

        Args:
            node: The node name.
            metric: The metric to log.
            value: The value.
            time: The time.

        """
        ray.get(self.ray_actor.log_system_metric.remote(node, metric, value, time))
