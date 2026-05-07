#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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

"""
P2PFL Logger.

.. note:: Not all is typed because the python logger is not typed (yep, is a TODO...).

"""

from __future__ import annotations

import copy
import datetime
import logging
from typing import TYPE_CHECKING, Any

from p2pfl.management.message_storage import MessageEntryType, MessageStorage
from p2pfl.management.metric_storage import GlobalLogsType, GlobalMetricStorage, LocalLogsType, LocalMetricStorage
from p2pfl.management.node_monitor import NodeMonitor
from p2pfl.settings import Settings

if TYPE_CHECKING:
    from p2pfl.workflow.engine.experiment import Experiment

###################
#    Exception    #
###################


class NodeNotRegistered(Exception):
    """Exception raised when a node is not registered."""

    pass


#########################
#    Colored logging    #
#########################

# COLORS
GRAY = "\033[90m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


def _color_for(value: str) -> str:
    """Generate a pretty pastel color from a string using golden-ratio hue spacing."""
    # FNV-1a hash for good distribution on short strings
    h = 0x811C9DC5
    for c in value:
        h ^= ord(c)
        h = (h * 0x01000193) & 0xFFFFFFFF
    # Golden ratio spacing ensures consecutive hashes land far apart on the hue wheel
    hue = ((h * 137.508) % 360) / 60.0
    x = 1.0 - abs(hue % 2 - 1.0)
    r, g, b = [(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][int(hue) % 6]
    # Pastel: blend toward white (0.55 color + 0.45 white), then scale to 255
    ri, gi, bi = int((r * 0.55 + 0.45) * 255), int((g * 0.55 + 0.45) * 255), int((b * 0.55 + 0.45) * 255)
    return f"\033[38;2;{ri};{gi};{bi}m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to the log messages."""

    def format(self, record):
        """
        Format the log record with color.

        Args:
            record: The log record.

        """
        # Copy the original record
        record_copy = copy.copy(record)

        # Warn level color
        if record_copy.levelname == "DEBUG":
            record_copy.levelname = BLUE + record_copy.levelname + RESET
        elif record_copy.levelname == "INFO":
            record_copy.levelname = GREEN + record_copy.levelname + RESET
        elif record_copy.levelname == "WARNING":
            record_copy.levelname = YELLOW + record_copy.levelname + RESET
        elif record_copy.levelname == "ERROR" or record_copy.levelname == "CRITICAL":
            record_copy.levelname = RED + record_copy.levelname + RESET
        return super().format(record_copy)


################
#    Logger    #
################


class P2PFLogger:
    """
    Class that manages the node logging (not a singleton).

    Args:
        p2pfl_web_services: The P2PFL Web Services to log and monitor the nodes remotely.

    """

    def __init__(self, nodes: dict[str, dict[str, Any]] | None = None, disable_locks: bool = False) -> None:
        """Initialize the logger."""
        # Node Information
        self._nodes: dict[str, dict[Any, Any]] = nodes if nodes else {}

        # Experiment Metrics and Message Storage
        self.disable_locks = disable_locks
        self.local_metrics = LocalMetricStorage(disable_locks=disable_locks)
        self.global_metrics = GlobalMetricStorage(disable_locks=disable_locks)
        self.message_storage = MessageStorage(disable_locks=disable_locks)
        self.node_monitor = NodeMonitor()

        # Image artifacts (for log_image)
        self._run_dir: str | None = None
        self._image_records: list[dict[str, Any]] = []

        # Python logging
        self._logger = logging.getLogger("p2pfl")
        if self._logger.handlers != []:
            print(self._logger.handlers)
            raise Exception("Logger already initialized.")
        self._logger.propagate = False
        self._logger.setLevel(logging.getLevelName(Settings.general.LOG_LEVEL))

        # STDOUT - Handler
        stream_handler = logging.StreamHandler()
        datefmt = "%Y-%m-%d %H:%M:%S" if Settings.general.LOG_FULL_TIMESTAMP else "%H:%M:%S"
        cmd_formatter = ColoredFormatter(
            f"{GRAY}[ \033[38;2;140;140;160m%(asctime)s {GRAY}| %(node)s {GRAY}| %(context)s%(levelname)s{GRAY} ]{RESET} %(message)s",
            datefmt=datefmt,
        )
        stream_handler.setFormatter(cmd_formatter)
        self._logger.addHandler(stream_handler)  # not async

    def connect(self, **kwargs: Any) -> None:
        """
        Establish connection/setup for the logger.

        This method should be overridden by loggers that require connection setup.
        By default, it does nothing.

        Args:
            **kwargs: Connection parameters specific to each logger type.

        """
        pass

    def cleanup(self) -> None:
        """Cleanup the logger."""
        # Stop resource monitoring
        self.node_monitor.stop()

        # Unregister nodes
        for node in self._nodes.copy():
            self.unregister_node(node)

        # Remove handlers from the logger
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)

    def finish(self) -> None:
        """
        Finish any logging activities, like closing a W&B run.

        This method is a placeholder and is meant to be implemented by a decorator.
        By default, it does nothing.
        """
        pass

    ######
    # Application logging
    ######

    def set_level(self, level: int | str) -> None:
        """
        Set the logger level.

        Args:
            level: The logger level.

        """
        if isinstance(level, str):
            self._logger.setLevel(logging.getLevelName(level))
        else:
            self._logger.setLevel(level)

    def get_level(self) -> int:
        """
        Get the logger level.

        Returns
            The logger level.

        """
        return self._logger.getEffectiveLevel()

    def get_level_name(self, lvl: int) -> str:
        """
        Get the logger level name.

        Args:
            lvl: The logger level.

        Returns:
            The logger level name.

        """
        return logging.getLevelName(lvl)

    def info(self, node: str, message: str) -> None:
        """
        Log an info message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.INFO, node, message)

    def debug(self, node: str, message: str) -> None:
        """
        Log a debug message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.DEBUG, node, message)

    def warning(self, node: str, message: str) -> None:
        """
        Log a warning message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.WARNING, node, message)

    def error(self, node: str, message: str) -> None:
        """
        Log an error message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.ERROR, node, message)

    def critical(self, node: str, message: str) -> None:
        """
        Log a critical message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self.log(logging.CRITICAL, node, message)

    def log(self, level: int, node: str, message: str) -> None:
        """
        Log a message.

        Args:
            level: The log level.
            node: The node name.
            message: The message to log.

        """
        # Build context string (round + stage) with independent colors
        context = ""
        node_data = self._nodes.get(node)
        colored_node = f"{_color_for(node)}{node}{RESET}"
        if node_data:
            parts = []
            r = node_data.get("round")
            if r is not None:
                r_str = f"R{r}"
                parts.append(f"{_color_for(r_str)}{r_str}{RESET}")
            s = node_data.get("stage")
            if s:
                parts.append(f"{_color_for(s)}{s}{RESET}")
            if parts:
                context = f"{f' {GRAY}| '.join(parts)} {GRAY}| "

        extra = {"node": colored_node, "context": context}
        if level == logging.DEBUG:
            self._logger.debug(message, extra=extra)
        elif level == logging.INFO:
            self._logger.info(message, extra=extra)
        elif level == logging.WARNING:
            self._logger.warning(message, extra=extra)
        elif level == logging.ERROR:
            self._logger.error(message, extra=extra)
        elif level == logging.CRITICAL:
            self._logger.critical(message, extra=extra)
        else:
            raise ValueError(f"Invalid level: {level}")

    ######
    # Metrics
    ######

    def log_metric(self, addr: str, metric: str, value: float, step: int | None = None, round: int | None = None) -> None:
        """
        Log a metric.

        Args:
            addr: The node name.
            metric: The metric to log.
            value: The value.
            step: The step.
            round: The round.

        """
        # Get Experiment
        try:
            experiment = self._nodes[addr]["Experiment"]
        except KeyError:
            # Node not registered, skip logging
            return

        # Get Round
        if round is None:
            round = self._nodes[addr].get("round")
            if round is None:
                raise Exception("No round provided. Needed for training metrics.")

        # Get Experiment Name
        exp = experiment.exp_name
        if exp is None:
            raise Exception("No experiment name provided. Needed for training metrics.")

        # Local storage
        if step is None:
            # Global Metrics
            self.global_metrics.add_log(exp, round, metric, addr, value)
        else:
            # Local Metrics
            self.local_metrics.add_log(exp, round, metric, addr, value, step)

    def set_run_dir(self, run_dir: str | None) -> None:
        """Set the run directory where image artifacts are persisted by log_image."""
        self._run_dir = run_dir

    def log_image(
        self,
        addr: str,
        name: str,
        image: Any,
        step: int | None = None,
        round: int | None = None,
        caption: str | None = None,
        fmt: str = "png",
    ) -> None:
        """
        Log an image artifact for a node at a given round/step.

        Accepts: matplotlib.figure.Figure, numpy.ndarray (HxW or HxWxC), PIL.Image, or str/Path
        to an existing image file. Encodes to ``fmt`` and writes to
        ``<run_dir>/images/<name>/round_NNNN_<addr>.<fmt>`` plus a ``latest.<fmt>``
        symlink for live preview.
        """
        if self._run_dir is None:
            self._logger.debug(f"log_image skipped (no run_dir set): name={name} addr={addr}")
            return

        import io
        import os
        from pathlib import Path

        out_dir = os.path.join(self._run_dir, "images", name)
        os.makedirs(out_dir, exist_ok=True)
        round_tag = "step" if round is None else "round"
        round_val = step if round is None else round
        if round_val is None:
            round_val = 0
        safe_addr = addr.replace("/", "_")
        filename = f"{round_tag}_{round_val:04d}_{safe_addr}.{fmt}"
        out_path = os.path.join(out_dir, filename)

        # Normalise input -> bytes
        try:
            if isinstance(image, (str, Path)):
                with open(image, "rb") as f:
                    data = f.read()
            elif hasattr(image, "savefig"):
                # matplotlib.figure.Figure
                buf = io.BytesIO()
                image.savefig(buf, format=fmt, dpi=100)
                data = buf.getvalue()
            elif hasattr(image, "save"):
                # PIL.Image.Image
                buf = io.BytesIO()
                image.save(buf, format=fmt.upper())
                data = buf.getvalue()
            else:
                # numpy array
                from PIL import Image as _PILImage

                arr = image
                if arr.dtype != "uint8":
                    import numpy as _np

                    arr = (_np.clip(arr, 0.0, 1.0) * 255).astype("uint8")
                buf = io.BytesIO()
                _PILImage.fromarray(arr).save(buf, format=fmt.upper())
                data = buf.getvalue()
        except Exception as exc:
            self._logger.warning(f"log_image: could not encode image: {exc}")
            return

        with open(out_path, "wb") as f:
            f.write(data)

        # Maintain a 'latest' symlink per name for live preview
        latest_link = os.path.join(out_dir, f"latest.{fmt}")
        try:
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(filename, latest_link)
        except OSError:
            pass

        self._image_records.append(
            {
                "addr": addr,
                "name": name,
                "round": round,
                "step": step,
                "path": out_path,
                "caption": caption,
                "size": len(data),
            }
        )

    def get_image_records(self) -> list[dict[str, Any]]:
        """Return the list of image records logged via log_image."""
        return list(self._image_records)

    def get_local_logs(self) -> LocalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self.local_metrics.get_all_logs()

    def get_global_logs(self) -> GlobalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self.global_metrics.get_all_logs()

    ######
    # Node registration
    ######

    def register_node(self, address: str) -> None:
        """
        Register a node.

        Args:
            address: The node address.

        """
        # Node State
        if self._nodes.get(address) is None:
            self._nodes[address] = {}
        else:
            raise Exception(f"Node {address} already registered.")

    def unregister_node(self, address: str) -> None:
        """
        Unregister a node.

        Args:
            address: The node address.

        """
        # Node state
        if address in self._nodes:
            self._nodes.pop(address)
        else:
            self.warning("SYSTEM", f"Attempted to unregister node {address} that was not registered.")

    ######
    # Node Status
    ######

    def experiment_started(self, address: str, experiment: Experiment) -> None:
        """
        Notify the experiment start.

        Args:
            address: The node address.
            experiment: The experiment.

        """
        if address not in self._nodes:
            raise NodeNotRegistered(f"Cannot start experiment: node '{address}' is not registered.")
        self._nodes[address]["Experiment"] = experiment
        self._nodes[address]["round"] = 0
        if not self.node_monitor.running:
            self.node_monitor.start()

    def experiment_ended(self, address: str, experiment: Experiment, status: str) -> None:
        """
        Notify that an experiment has ended.

        Args:
            address: The node address.
            experiment: The experiment.
            status: Final status (e.g. "finished", "cancelled", "failed").

        """
        if address in self._nodes:
            self._nodes[address].pop("Experiment", None)
            self._nodes[address].pop("round", None)
        # Stop monitor only when no nodes have active experiments
        if not any("Experiment" in data for data in self._nodes.values()):
            self.node_monitor.stop()

    def on_experiment_change(self, address: str, field_name: str, value: Any) -> None:
        """
        Handle an experiment attribute change.

        Called by the ``ExperimentLoggerObserver`` whenever an observed
        experiment attribute is set. Updates internal round and stage tracking.

        Args:
            address: The node address.
            field_name: The name of the changed attribute.
            value: The new value.

        """
        if address not in self._nodes:
            return
        if field_name == "round":
            self._nodes[address]["round"] = value
        elif field_name == "current_stage":
            self._nodes[address]["stage"] = value

    def get_nodes(self) -> dict[str, dict[Any, Any]]:
        """
        Get the registered nodes.

        Returns:
            The registered nodes.

        """
        return self._nodes

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add a handler to the logger.

        Args:
            handler: The logger handler.

        """
        self._logger.addHandler(handler)

    ######
    # Communication Logs
    ######

    def log_communication(
        self,
        node: str,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: int | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a communication event.

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
        if not Settings.general.LOG_COMMUNICATION:
            return

        # Determine emoji based on direction and package type
        emoji = ("📫" if package_type == "message" else "📦") if direction == "received" else ("📤" if package_type == "message" else "📬")

        # If round_num is not specified but we're in an experiment, get the current round
        if round_num is None or round_num < 0:
            try:
                if node in self._nodes and "round" in self._nodes[node]:
                    stored_round = self._nodes[node]["round"]
                    if stored_round is not None:
                        round_num = stored_round
            except Exception:
                # If we can't get the round, just continue with default round_num (None)
                pass

        # Create base message
        message = f"{emoji} {cmd.upper()} {direction} "
        if direction == "received":
            message += f"from {source_dest}"
        else:
            message += f"to {source_dest}"

        # Add round information if available
        if round_num is not None and round_num >= 0:
            message += f" (round {round_num})"

        # Log the message at debug level
        if cmd != "beat" or (not Settings.heartbeat.EXCLUDE_BEAT_LOGS and cmd == "beat"):
            self.debug(node, message)

        # Get actual round number for storage (default to 0 if None)
        storage_round = 0 if round_num is None or round_num < 0 else round_num

        # Store in message storage
        self.message_storage.add_message(
            node=node,
            direction=direction,
            cmd=cmd,
            source_dest=source_dest,
            package_type=package_type,
            package_size=package_size,
            round_num=storage_round,
            additional_info=additional_info,
        )

    def get_messages(
        self,
        direction: str = "all",  # "all", "sent", or "received"
        node: str | None = None,
        cmd: str | None = None,
        round_num: int | None = None,
        limit: int | None = None,
    ) -> list[MessageEntryType]:
        """
        Get communication messages with optional filtering.

        Args:
            direction: Filter by message direction ("all", "sent", or "received").
            node: Filter by node address (optional).
            cmd: Filter by command type (optional).
            round_num: Filter by round number (optional).
            limit: Limit the number of messages returned per node (optional).

        Returns:
            A flat list of message dictionaries. Each message includes a 'direction' field
            indicating whether it was 'sent' or 'received'.

        """
        # Validate direction
        if direction not in ["all", "sent", "received"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'all', 'sent', or 'received'.")

        # Convert "all" to None as expected by message_storage
        storage_direction = None if direction == "all" else direction

        return self.message_storage.get_messages(node=node, direction=storage_direction, cmd=cmd, round_num=round_num, limit=limit)

    def get_system_metrics(self) -> dict[datetime.datetime, dict[str, float]]:
        """
        Get the system metrics.

        Returns:
            The system metrics.

        """
        return self.node_monitor.get_logs()

    def reset(self) -> None:
        """
        Reset the logger state between experiments.

        This clears all stored metrics, messages, and system logs while keeping
        the logger configuration and handlers intact.
        """
        # Recreate storage instances to clear all data
        self.local_metrics = LocalMetricStorage(disable_locks=self.disable_locks)
        self.global_metrics = GlobalMetricStorage(disable_locks=self.disable_locks)
        self.message_storage = MessageStorage(disable_locks=self.disable_locks)

        # Clear system metrics and registered nodes
        self.node_monitor.logs.clear()
        self._nodes.clear()

        self.info("SYSTEM", "Logger state reset for new experiment")
