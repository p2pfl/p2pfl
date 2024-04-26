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
from typing import List, Optional
from p2pfl.settings import Settings
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
import datetime
from p2pfl.management.p2pfl_web_services import P2pflWebServices
from p2pfl.management.node_monitor import NodeMonitor

#########################################
#    Logging handler (transmit logs)    #
#########################################


class DictFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, record):
        log_dict = {
            "timestamp": datetime.datetime.fromtimestamp(record.created),
            "level": record.levelname,
            "node": record.node,
            "message": record.getMessage(),
        }
        return log_dict


class P2pflWebLogHandler(logging.Handler):
    def __init__(self, p2pfl_web: P2pflWebServices):
        super().__init__()
        self.p2pfl_web = p2pfl_web
        self.formatter = DictFormatter()  # Instantiate the custom formatter

    def emit(self, record):
        # Format the log record using the custom formatter
        log_message = self.formatter.format(record)
        # Send log entry to the API
        self.p2pfl_web.send_log(
            log_message["timestamp"],
            log_message["node"],
            log_message["level"],
            log_message["message"],
        )


#########################
#    Colored logging    #
#########################

# COLORS
GRAY = "\033[90m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
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


################
#    Logger    #
################


# AL SER ARINCRONO, AL FINALIZAR EL PROGRAMA NO SE TERMINAN DE EJECTUAR LOS LOGS


class Logger:
    """
    Class that contains node logging.

    Singleton class.
    """

    ######
    # Singleton and instance management
    ######

    __instance = None

    @staticmethod
    def connect_web(url: str, key: str) -> None:
        # Remove the instance if it already exists
        if Logger.__instance is not None:
            Logger.__instance.queue_listener.stop()
            Logger.__instance = None

        # Create the instance
        p2pfl_web = P2pflWebServices(url, key)
        Logger.__instance = Logger(p2pfl_web_services=p2pfl_web)

        print("al setear el web services, se debe iniciar el node status reporter")

    def __init__(self, p2pfl_web_services: Optional[P2pflWebServices] = None) -> None:

        # Python logging
        self.logger = logging.getLogger("p2pfl")
        self.logger.propagate = False
        self.logger.setLevel(logging.getLevelName(Settings.LOG_LEVEL))
        handlers: List[logging.Handler] = []

        # P2PFL Web Services
        self.p2pfl_web_services = p2pfl_web_services
        if p2pfl_web_services is not None:
            web_handler = P2pflWebLogHandler(p2pfl_web_services)
            handlers.append(web_handler)

        # STDOUT - Handler
        stream_handler = logging.StreamHandler()
        cmd_formatter = ColoredFormatter(
            f"{GRAY}[ {YELLOW}%(asctime)s {GRAY}| {CYAN}%(node)s {GRAY}| %(levelname)s{GRAY} ]:{RESET} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(cmd_formatter)
        self.logger.addHandler(stream_handler)  # not async

        # FILE - Handler
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
        handlers.append(file_handler)

        # Asynchronous logging (queue handler)
        log_queue: multiprocessing.Queue[logging.LogRecord] = multiprocessing.Queue()
        queue_handler = QueueHandler(log_queue)
        self.logger.addHandler(queue_handler)
        self.queue_listener = QueueListener(log_queue, *handlers)
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

    ######
    # Node registration
    ######

    @staticmethod
    def register_node(node: str, simulation: bool) -> None:
        """
        Register a node.

        Args:
            node (str): The node address.
            simulation (bool): If the node is simulated.
        """
        if Logger.__instance.p2pfl_web_services is not None:
            # Register the node
            Logger.__instance.p2pfl_web_services.register_node(node, simulation)
            # Start the node status reporter
            NodeMonitor(node, Logger.__instance.log_metric).start()

    @staticmethod
    def unregister_node(node: str) -> None:
        """
        Unregister a node.

        Args:
            node (str): The node address.
        """
        if Logger.__instance.p2pfl_web_services is not None:
            Logger.__instance.p2pfl_web_services.unregister_node(node)
        print("not implemented")
        # NO ESTA SIENDO LLAMADO, NECESARIO PARA SABER EL ESTADO DEL NODO

    ######
    # Application logging
    ######

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
        Logger.__instance.log(logging.INFO, node, message)

    @staticmethod
    def debug(node: str, message: str) -> None:
        """
        Log a debug message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.__instance.log(logging.DEBUG, node, message)

    @staticmethod
    def warning(node: str, message: str) -> None:
        """
        Log a warning message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.__instance.log(logging.WARNING, node, message)

    @staticmethod
    def error(node: str, message: str) -> None:
        """
        Log an error message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.__instance.log(logging.ERROR, node, message)

    @staticmethod
    def critical(node: str, message: str) -> None:
        """
        Log a critical message.

        Args:
            node (str): The node name.
            message (str): The message to log.
        """
        Logger.__instance.log(logging.CRITICAL, node, message)

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

    ######
    # Metrics
    ######

    @staticmethod
    def log_metric(node: str, metric: str, value: float, time: str) -> None:
        """
        Log a metric.

        Args:
            node (str): The node name.
            metric (str): The metric to log.
            value (float): The value.
        """
        if Logger.__instance.p2pfl_web_services is not None:
            Logger.__instance.p2pfl_web_services.send_metric(node, metric, time, value)

    @staticmethod
    def wait_stop():
        # Stop the queue listener
        Logger.__instance.queue_listener.stop()


logger = Logger
