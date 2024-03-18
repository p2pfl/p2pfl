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
import os
from p2pfl.settings import Settings
from logging.handlers import QueueHandler, QueueListener
import queue

# COLORS
GRAY = "\033[90m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Warn level color
        if record.levelname == "DEBUG":
            record.levelname = BLUE + record.levelname + RESET
        elif record.levelname == "INFO":
            record.levelname = GREEN + record.levelname + RESET
        elif record.levelname == "WARNING":
            record.levelname = YELLOW + record.levelname + RESET
        elif record.levelname == "ERROR" or record.levelname == "CRITICAL":
            record.levelname = RED + record.levelname + RESET
        return super().format(record)


# METER UI LOGGING COMO HANDLER
"""
class RemoteLogger(HANDLER):
"""


class Logger:
    """
    Class that contains node logging.

    Singleton class.
    """

    __instance = None

    def __init__(self) -> None:
        # Remote logging
        self.remote_logging = None

        # Python logging
        self.logger = logging.getLogger("p2pfl")
        self.logger.propagate = False
        self.logger.setLevel(logging.getLevelName(Settings.LOG_LEVEL))

        # STDOUT
        self.stream_handler = logging.StreamHandler()
        cmd_formatter = ColoredFormatter(
            f"{GRAY}[ {YELLOW}%(asctime)s {GRAY}| {CYAN}%(node)s {GRAY}| %(levelname)s{GRAY} ]:{RESET} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.stream_handler.setFormatter(cmd_formatter)

        # FILE
        if not os.path.exists(Settings.LOG_DIR):
            os.makedirs(Settings.LOG_DIR)
        file_handler = logging.handlers.RotatingFileHandler(
            f"{Settings.LOG_DIR}/p2pfl.log", maxBytes=1000000, backupCount=3
        )  # TODO: ADD DIFFERENT LOG FILES FOR DIFFERENT NODES / EXPERIMENTS
        file_formatter = logging.Formatter(
            "[ %(asctime)s | %(node)s | %(levelname)s ]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        # Asynchronous logging (queue handler)
        log_queue: queue.Queue[logging.LogRecord] = queue.Queue()
        queue_handler = QueueHandler(log_queue)
        self.logger.addHandler(queue_handler)
        self.queue_listener = QueueListener(
            log_queue, file_handler, self.stream_handler
        )
        self.queue_listener.start()

    @staticmethod
    def get_instance() -> "Logger":
        """
        Return the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        if Logger.__instance is None:
            Logger.__instance = Logger()
        return Logger.__instance

    @staticmethod
    def configure_remote_logging(url: str, key: str) -> None:
        """
        Configure remote logging.

        Args:
            url (str): The remote logging URL.
            key (str): The remote logging key.
        """
        raise NotImplementedError("Remote logging not implemented yet")
        # Logger.get_instance().remote_logging = RemoteLogger(url, key)

    @staticmethod
    def set_level(level: int) -> None:
        """
        Set the logger level.

        Args:
            level (int): The logger level.
        """
        Logger.get_instance().logger.setLevel(level)

    @staticmethod
    def info(node: str, message: str) -> None:
        """
        Log an info message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.get_instance().log(logging.INFO, node, message)

    @staticmethod
    def debug(node: str, message: str) -> None:
        """
        Log a debug message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.get_instance().log(logging.DEBUG, node, message)

    @staticmethod
    def warning(node: str, message: str) -> None:
        """
        Log a warning message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.get_instance().log(logging.WARNING, node, message)

    @staticmethod
    def error(node: str, message: str) -> None:
        """
        Log an error message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.get_instance().log(logging.ERROR, node, message)

    @staticmethod
    def critical(node: str, message: str) -> None:
        """
        Log a critical message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.get_instance().log(logging.CRITICAL, node, message)

    def log(self, level: int, node: str, message: str) -> None:
        """
        Log a message.

        Args:
            level (int): The message level.
            message (str): The message to log.
            level (int): The logger level.
        """
        # Traditional logging
        if level == logging.DEBUG:
            self.logger.debug(message, extra={"node": node})
        elif level == logging.INFO:
            self.logger.info(message, extra={"node": node})
        elif level == logging.WARNING:
            self.logger.warning(message, extra={"node": node})
        elif level == logging.ERROR:
            self.logger.error(message, extra={"node": node})
        elif level == logging.CRITICAL:
            self.logger.critical(message, extra={"node": node})
        else:
            raise ValueError(f"Invalid level: {level}")

        # Remote logging
        if self.remote_logging is not None:
            raise NotImplementedError("Remote logging not implemented yet")

    @staticmethod
    def log_metric(node: str, metric: str, value: float) -> None:
        """
        Log a metric.

        Args:
            metric (str): The metric name.
            value (float): The metric value.
        """
        if Logger.get_instance().remote_logging is not None:
            raise NotImplementedError("Remote logging not implemented yet")

    @staticmethod
    def stop():
        """
        Stop the logger.
        """
        # Stop the queue listener
        Logger.get_instance().queue_listener.stop()


logger = Logger
