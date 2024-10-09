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

from typing import Optional
import ray

from p2pfl.experiment import Experiment
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.management.metric_storage import GlobalLogsType, LocalLogsType

@ray.remote
class RayP2PFLoggerActor(P2PFLogger):
    """Actor to add remote logging capabilities to a logger class."""
    
    _p2pflogger: P2PFLogger = None

    def __init__(self, p2pflogger: P2PFLogger) -> None:
        self._p2pflogger = p2pflogger

    # Methods that simply wrap the P2PFLogger's functionality
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

    def log_metric(self, addr: str, experiment: Experiment, metric: str, value: float,
                   round: Optional[int] = None, step: Optional[int] = None) -> None:
        self._p2pflogger.log_metric(addr, experiment, metric, value, round, step)

    def get_local_logs(self) -> LocalLogsType:
        return self._p2pflogger.get_local_logs()

    def get_global_logs(self) -> GlobalLogsType:
        return self._p2pflogger.get_global_logs()

    def register_node(self, node: str, simulation: bool) -> None:
        self._p2pflogger.register_node(node, simulation)

    def unregister_node(self, node: str) -> None:
        self._p2pflogger.unregister_node(node)
    
    def cleanup(self) -> None:
        self._p2pflogger.cleanup()

    def get_level_name(self, lvl: int) -> str:
        raise self._p2pflogger.get_level_name(lvl)
    
    def set_level(self, level: int) -> None:
        self._p2pflogger.set_level(level)

    def get_level(self) -> int:
        return self._p2pflogger.get_level()

    def log(self, level: int, node: str, message: str) -> None:
        self._p2pflogger.log(level, node, message)

    def experiment_started(self, node: str) -> None:
        self._p2pflogger.experiment_started(node)

    def experiment_finished(self, node: str) -> None:
        self._p2pflogger.experiment_finished(node)

    def round_finished(self, node: str) -> None:
        self._p2pflogger.round_finished(node)


class RayP2PFLogger:
    def __init__(self, p2pflogger: P2PFLogger):
        """
        Initialize the wrapper with a Ray actor instance.

        Args:
            logger: The logger to be wrapped.
        """
        self.ray_actor = RayP2PFLoggerActor.options(name="p2pfl_logger", lifetime="detached", get_if_exists=True).remote(p2pflogger)

    def __getattr__(self, name):
        """
        Intercept method calls and automatically convert them to remote calls.
        
        Args:
            name: The name of the method being called.
        
        Returns:
            A function that invokes the corresponding remote method.
        """
        # Get the actual method from the Ray actor
        method = getattr(self.ray_actor, name)
        
        # Return a wrapper that automatically calls .remote() on the method
        def remote_method(*args, **kwargs):
            return method.remote(*args, **kwargs)

        return remote_method
