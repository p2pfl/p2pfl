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

import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Dict, List, Tuple

from p2pfl.experiment import Experiment
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.settings import Settings

class FileP2PFLogger(P2PFLogger):
    _p2pflogger: P2PFLogger = None

    def __init__(self, p2pfllogger: P2PFLogger, log_dir: str):
        self._p2pflogger = p2pfllogger
        self.log_dir = log_dir

        # Setup the file handler for logging
        self.setup_file_handler(self.log_dir)
    
    def setup_file_handler(self) -> None:
        """Set up the file handler for logging."""
        if not os.path.exists(Settings.LOG_DIR):
            os.makedirs(Settings.LOG_DIR)

        file_handler = RotatingFileHandler(
            f"{Settings.LOG_DIR}/p2pfl.log", maxBytes=1000000, backupCount=3
        ) # TODO: ADD DIFFERENT LOG FILES FOR DIFFERENT NODES / EXPERIMENTS
        file_formatter = logging.Formatter(
            "[ %(asctime)s | %(node)s | %(levelname)s ]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)

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

    def log_metric(self, addr: str, experiment: Experiment, metric: str,
                   value: float, round: int | None = None,
                   step: int | None = None) -> None:
        self._p2pflogger.log_metric(addr, experiment, metric, value, round, step)

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