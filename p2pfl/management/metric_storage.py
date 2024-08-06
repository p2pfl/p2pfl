#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Metric storage."""

from threading import Lock
from typing import Dict, List, Tuple, Union

MetricsType = Dict[str, List[Tuple[int, float]]]  # Metric name -> [(step, value)...]
NodeLogsType = Dict[str, MetricsType]  # Node name -> MetricsType
RoundLogsType = Dict[int, NodeLogsType]  # Round -> NodeLogsType
LocalLogsType = Dict[str, RoundLogsType]  # Experiment -> RoundLogsType


class LocalMetricStorage:
    """
    Local metric storage. It stores the metrics for each node in each round of each experiment.

    Format:

    .. code-block:: python

        {
            "experiment": {
                "round": {
                    "node_name": {
                        "metric": [(step, value), ...]
                    }
                }
            }
        }

    """

    def __init__(self) -> None:
        """Initialize the local metric storage."""
        self.exp_dicts: LocalLogsType = {}
        self.lock = Lock()

    def add_log(
        self,
        exp_name: str,
        round: int,
        metric: str,
        node: str,
        val: Union[int, float],
        step: int,
    ) -> None:
        """
        Add a log entry.

        Args:
            exp_name: Experiment name.
            round: Round number.
            metric: Metric name.
            node: Node name.
            val: Value of the metric.
            step: Step number.

        """
        # Lock
        self.lock.acquire()

        # Create Experiment if needed
        if exp_name not in self.exp_dicts:
            self.exp_dicts[exp_name] = {}

        # Create round entry if needed
        if round not in self.exp_dicts[exp_name]:
            self.exp_dicts[exp_name][round] = {}

        # Create node entry if needed
        if node not in self.exp_dicts[exp_name][round]:
            self.exp_dicts[exp_name][round][node] = {}

        # Create metric entry if needed
        if metric not in self.exp_dicts[exp_name][round][node]:
            self.exp_dicts[exp_name][round][node][metric] = [(step, val)]
        else:
            self.exp_dicts[exp_name][round][node][metric].append((step, val))

        # Release Lock
        self.lock.release()

    def get_all_logs(self) -> LocalLogsType:
        """
        Obtain all logs.

        Returns:
            All logs

        """
        return self.exp_dicts

    def get_experiment_logs(self, exp: str) -> RoundLogsType:
        """
        Obtain logs for an experiment.

        Args:
            exp: Experiment number

        Returns:
            Experiment logs

        """
        return self.exp_dicts[exp]

    def get_experiment_round_logs(self, exp: str, round: int) -> NodeLogsType:
        """
        Obtain logs for a round in an experiment.

        Args:
            exp: Experiment number
            round: Round number

        Returns:
            Round logs

        """
        return self.exp_dicts[exp][round]

    def get_experiment_round_node_logs(self, exp: str, round: int, node: str) -> MetricsType:
        """
        Obtain logs for a node in an experiment.

        Args:
            exp: Experiment number
            round: Round number
            node: Node name

        Returns:
            Node logs

        """
        return self.exp_dicts[exp][round][node]


GlobalLogsType = Dict[str, NodeLogsType]


class GlobalMetricStorage:
    """
    Global metric storage. It stores the metrics for each node in each experiment.

    Format:

    .. code-block:: python

        {
            "experiment":{
                "node_name": {
                    "metric": [(round, value), ...]
            }
        }

    """

    def __init__(self) -> None:
        """Initialize the global metric storage."""
        self.exp_dicts: GlobalLogsType = {}
        self.lock = Lock()

    def add_log(self, exp_name: str, round: int, metric: str, node: str, val: Union[int, float]) -> None:
        """
        Add a log entry.

        Args:
            exp_name: Experiment name.
            round: Round number.
            metric: Metric name.
            node: Node name.
            val: Value of the metric.

        """
        # Lock
        self.lock.acquire()

        # Create Experiment if needed
        if exp_name not in self.exp_dicts:
            self.exp_dicts[exp_name] = {}

        # Create node entry if needed
        if node not in self.exp_dicts[exp_name]:
            self.exp_dicts[exp_name][node] = {}

        # Create metric entry if needed
        if metric not in self.exp_dicts[exp_name][node]:
            self.exp_dicts[exp_name][node][metric] = [(round, val)]
        else:
            # Log if not already logged
            if round not in [r for r, _ in self.exp_dicts[exp_name][node][metric]]:
                self.exp_dicts[exp_name][node][metric].append((round, val))

        # Release Lock
        self.lock.release()

    def get_all_logs(self) -> GlobalLogsType:
        """
        Obtain all logs.

        Returns:
            All logs

        """
        return self.exp_dicts

    def get_experiment_logs(self, exp: str) -> NodeLogsType:
        """
        Obtain logs for an experiment.

        Args:
            exp: Experiment number

        Returns:
            Experiment logs

        """
        return self.exp_dicts[exp]

    def get_experiment_node_logs(self, exp: str, node: str) -> MetricsType:
        """
        Obtain logs for a node in an experiment.

        Args:
            exp: Experiment number
            node: Node name

        Returns:
            Node logs

        """
        return self.exp_dicts[exp][node]
