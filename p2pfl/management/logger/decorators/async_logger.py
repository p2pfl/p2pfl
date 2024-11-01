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
"""Async logger."""

import atexit
import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener

from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger


class AsyncLogger(LoggerDecorator):
    """Async logger decorator."""

    def __init__(self, p2pflogger: P2PFLogger) -> None:
        """Initialize the logger."""
        super().__init__(p2pflogger)

        # Set up asynchronous logging
        self.log_queue: multiprocessing.Queue[logging.LogRecord] = multiprocessing.Queue()
        queue_handler = QueueHandler(self.log_queue)
        self._p2pfl_logger.add_handler(queue_handler)

        # Set up a listener for the queue
        self.queue_listener = QueueListener(self.log_queue)
        self.queue_listener.start()

        # Register cleanup function to close the queue on exit
        atexit.register(self.cleanup)

    def add_handler(self, handler):
        """Add a handler to the logger."""
        # add handler to tuple of handlers
        self.queue_listener.handlers = self.queue_listener.handlers + (handler,)

    def cleanup(self) -> None:
        """Cleanup the logger."""
        if self.queue_listener:
            self.queue_listener.stop()
        self.log_queue.close()
        super().cleanup()
