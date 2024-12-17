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

"""File Logger Decorator."""

import logging
import os
from logging.handlers import RotatingFileHandler

from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.settings import Settings


class FileLogger(LoggerDecorator):
    """File logger decorator."""

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the logger."""
        super().__init__(p2pflogger)

        # Setup the file handler for logging
        self.setup_file_handler()

    def setup_file_handler(self) -> None:
        """Set up the file handler for logging."""
        if not os.path.exists(Settings.LOG_DIR):
            os.makedirs(Settings.LOG_DIR)

        file_handler = RotatingFileHandler(
            f"{Settings.LOG_DIR}/p2pfl.log", maxBytes=1000000, backupCount=3
        )  # TODO: ADD DIFFERENT LOG FILES FOR DIFFERENT NODES / EXPERIMENTS
        file_formatter = logging.Formatter(
            "[ %(asctime)s | %(node)s | %(levelname)s ]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.add_handler(file_handler)
