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

from logging.handlers import QueueHandler, QueueListener
import multiprocessing
from typing import Dict, List, Tuple
from p2pfl.experiment import Experiment
from p2pfl.management.logger.logger import P2PFLogger

class AsyncLocalLogger(P2PFLogger):
    _p2pflogger: P2PFLogger = None

    def __init__(self, p2pflogger: P2PFLogger) -> None:
        self._logger = p2pflogger._logger
        self._p2pflogger = p2pflogger

        # Set up asynchronous logging
        self.log_queue = multiprocessing.Queue()
        queue_handler = QueueHandler(self.log_queue)
        self._logger.addHandler(queue_handler)

        # Set up a listener for the queue
        self.queue_listener = QueueListener(self.log_queue,
                                            *self._logger.handlers)
        self.queue_listener.start()

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

    def get_local_logs(self) -> Dict[str, Dict[int, Dict[str, Dict[str, List[Tuple[int | float]]]]]]:
        return self._p2pflogger.get_local_logs()

    def get_global_logs(self) -> Dict[str, Dict[str, Dict[str, List[Tuple[int | float]]]]]:
        return self._p2pflogger.get_global_logs()

    def register_node(self, node: str, simulation: bool) -> None:
        self._p2pflogger.register_node(node, simulation)

    def unregister_node(self, node: str) -> None:
        self._p2pflogger.unregister_node(node)
    
    def cleanup(self) -> None:
        """Cleanup the logger."""
        if self.queue_listener:
            self.queue_listener.stop()
        self.log_queue.close()

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
