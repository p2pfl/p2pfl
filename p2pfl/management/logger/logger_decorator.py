import logging
from typing import Any, Dict, List, Optional, Tuple

from p2pfl.experiment import Experiment
from p2pfl.management.logger.logger import P2PFLogger

from p2pfl.management.metric_storage import GlobalLogsType, LocalLogsType

class P2PFLoggerDecorator(P2PFLogger):
    _p2pflogger: P2PFLogger

    def info(self, node: str, message: str) -> None:
        """
        Log an info message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self._p2pflogger.info(node, message)

    def debug(self, node: str, message: str) -> None:
        """
        Log a debug message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self._p2pflogger.debug(node, message)

    def warning(self, node: str, message: str) -> None:
        """
        Log a warning message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self._p2pflogger.warning(node, message)

    def error(self, node: str, message: str) -> None:
        """
        Log an error message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self._p2pflogger.error(node, message)

    def critical(self, node: str, message: str) -> None:
        """
        Log a critical message.

        Args:
            node: The node name.
            message: The message to log.

        """
        self._p2pflogger.critical(node, message)

    def log_metric(self, addr: str, metric: str,
                   value: float, round: int | None = None,
                   step: int | None = None) -> None:
        """
        Log a metric.

        Args:
            node: The node name.
            metric: The metric to log.
            value: The value.
            step: The step.
            round: The round.

        """
        self._p2pflogger.log_metric(addr, metric, value, round, step)

    def get_local_logs(self) -> LocalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self._p2pflogger.get_local_logs()

    def get_global_logs(self) -> GlobalLogsType:
        """
        Get the logs.

        Args:
            node: The node name.
            exp: The experiment name.

        Returns:
            The logs.

        """
        return self._p2pflogger.get_global_logs()

    def register_node(self, node: str, simulation: bool) -> None:
        """
        Register a node.

        Args:
            node: The node address.
            simulation: If the node is a simulation.

        """
        self._p2pflogger.register_node(node, simulation)

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        self._p2pflogger.unregister_node(node)
    
    def cleanup(self) -> None:
        """Cleanup the logger."""
        self._p2pflogger.cleanup()

    def get_level_name(self, lvl: int) -> str:
        """
        Get the logger level name.

        Args:
            lvl: The logger level.

        Returns:
            The logger level name.

        """
        return self._p2pflogger.get_level_name(lvl)
    
    def set_level(self, level: int) -> None:
        """
        Set the logger level.

        Args:
            level: The logger level.

        """
        self._p2pflogger.set_level(level)

    def get_level(self) -> int:
        """
        Get the logger level.

        Returns
            The logger level.

        """
        return self._p2pflogger.get_level()

    def log(self, level: int, node: str, message: str) -> None:
        """
        Log a message.

        Args:
            level: The log level.
            node: The node name.
            message: The message to log.

        """
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

    def get_logger(self) -> logging.Logger:
        """
        Get the logger instance.

        Returns:
            The logger instance.
        """
        return self._p2pflogger.get_logger()

    def get_nodes(self) -> Dict[str, Dict[Any, Any]]:
        """
        Get the registered nodes.

        Returns:
            The registered nodes.
        """
        return self._p2pflogger.get_nodes()

    def set_logger(self, logger: logging.Logger) -> None:
        """
        Set the logger instance.

        Args:
            logger: The logger instance.
        """
        self._p2pflogger.set_logger(logger)

    def set_nodes(self, nodes: Dict[str, Dict[Any, Any]]) -> None:
        """
        Set the registered nodes.

        Args:
            nodes: The registered nodes.
        """
        self._p2pflogger.set_nodes(nodes)

    def get_handlers(self) -> List[logging.Handler]:
        """
        Get the logger handlers.

        Returns:
            The logger handlers.
        """
        return self._p2pflogger.get_handlers()
    
    def set_handlers(self, handler: List[logging.Handler]) -> None:
        """
        Set the logger handlers.

        Args:
            handler: The logger handlers.
        """
        self._p2pflogger.set_handlers(handler)

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Add a handler to the logger.

        Args:
            handler: The handler to add.
        """
        self._p2pflogger.add_handler(handler)