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

"""Web Logger."""

import datetime
import logging
from typing import Any, Dict, Optional

from p2pfl.experiment import Experiment
from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import NodeNotRegistered, P2PFLogger
from p2pfl.management.p2pfl_web_services import P2pflWebServices

#########################################
#    Logging handler (transmit logs)    #
#########################################


class DictFormatter(logging.Formatter):
    """Formatter (logging) that returns a dictionary with the log record attributes."""

    def format(self, record):
        """
        Format the log record as a dictionary.

        Args:
            record: The log record.

        """
        # Get node
        if not hasattr(record, "node"):
            raise ValueError("The log record must have a 'node' attribute.")
        log_dict = {
            "timestamp": datetime.datetime.fromtimestamp(record.created),
            "level": record.levelname,
            "node": record.node,  # type: ignore
            "message": record.getMessage(),
        }
        return log_dict


class P2pflWebLogHandler(logging.Handler):
    """
    Custom logging handler that sends log entries to the API.

    Args:
        p2pfl_web: The P2PFL Web Services.

    """

    def __init__(self, p2pfl_web: P2pflWebServices):
        """Initialize the handler."""
        super().__init__()
        self.p2pfl_web = p2pfl_web
        self.formatter = DictFormatter()  # Instantiate the custom formatter

    def emit(self, record):
        """
        Emit the log record.

        Args:
            record: The log record.

        """
        # Format the log record using the custom formatter
        log_message = self.formatter.format(record)  # type: ignore
        # Send log entry to the API
        self.p2pfl_web.send_log(
            log_message["timestamp"],  # type: ignore
            log_message["node"],  # type: ignore
            log_message["level"],  # type: ignore
            log_message["message"],  # type: ignore
        )


class WebP2PFLogger(LoggerDecorator):
    """Web logger decorator."""

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the logger."""
        super().__init__(p2pflogger)
        self._p2pfl_web_services: Optional[P2pflWebServices] = None

    def connect_web(self, url: str, key: str) -> None:
        """
        Connect to the web services.

        Args:
            url: The URL of the web services.
            key: The API key.

        """
        if self._p2pfl_web_services is not None:
            raise Exception("Web services already connected.")
        self._p2pfl_web_services = P2pflWebServices(url, key)
        self.add_handler(P2pflWebLogHandler(self._p2pfl_web_services))

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
        super().log_metric(addr=addr, metric=metric, value=value, step=step, round=round)
        if self._p2pfl_web_services is not None:
            # Get Experiment
            try:
                experiment: Experiment = self._nodes[addr]["Experiment"]
            except KeyError:
                raise NodeNotRegistered(f"Node {addr} not registered.") from None

            if step is None:
                # Global Metrics
                self._p2pfl_web_services.send_global_metric(experiment.exp_name, experiment.round, metric, addr, value)
            else:
                # Local Metrics
                self._p2pfl_web_services.send_local_metric(experiment.exp_name, experiment.round, metric, addr, value, step)

    def log_communication(
        self,
        node: str,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a communication event and send it to web services if connected.

        Args:
            node: The node address.
            direction: Direction of communication ("sent" or "received").
            cmd: The command or message type.
            source_dest: Source (if receiving) or destination (if sending) node.
            package_type: Type of package ("message" or "weights").
            package_size: Size of the package in bytes (if available).
            round_num: The federated learning round number (if applicable).
            additional_info: Additional information as a dictionary.

        """
        # Call parent's method first
        super().log_communication(
            node=node,
            direction=direction,
            cmd=cmd,
            source_dest=source_dest,
            package_type=package_type,
            package_size=package_size,
            round_num=round_num,
            additional_info=additional_info,
        )

        # Send to web services if connected
        if self._p2pfl_web_services is not None:
            # Create timestamp
            now = datetime.datetime.now()

            # Send as a structured communication log
            try:
                self._p2pfl_web_services.send_communication_log(
                    node=node,
                    timestamp=now,
                    direction=direction,
                    cmd=cmd,
                    source_dest=source_dest,
                    package_type=package_type,
                    package_size=package_size,
                    round_num=round_num,
                    additional_info=additional_info,
                )
            except Exception as e:
                # Error handling
                print(f"Error sending communication log to web services: {e}")
                raise e

    def register_node(self, node: str) -> None:
        """
        Register a node.

        Args:
            node: The node address.

        """
        super().register_node(node)
        if self._p2pfl_web_services is not None:
            self._p2pfl_web_services.register_node(node)

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        super().unregister_node(node)
        if self._p2pfl_web_services is not None:
            self._p2pfl_web_services.unregister_node(node)
