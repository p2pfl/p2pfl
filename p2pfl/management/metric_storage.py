from threading import Lock
from typing import Dict, List, Tuple, Union

MetricsType = Dict[str, List[Tuple[int, float]]]  # Metric name -> [(step, value)...]
NodeLogsType = Dict[str, MetricsType]  # Node name -> MetricsType
RoundLogsType = Dict[str, NodeLogsType]  # Round -> NodeLogsType
LocalLogsType = Dict[str, RoundLogsType]  # Experiment -> RoundLogsType


class LocalMetricStorage:
    """
    "experiment":{
        "round": {
            "node_name": {
                "metric": [(step, value), ...]
            }
        }
    }
    """

    def __init__(self):
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
        # Lock
        self.lock.acquire()

        # Create Experiment if needed
        if exp_name not in self.exp_dicts.keys():
            self.exp_dicts[exp_name] = {}

        # Create round entry if needed
        if round not in self.exp_dicts[exp_name].keys():
            self.exp_dicts[exp_name][round] = {}

        # Create node entry if needed
        if node not in self.exp_dicts[exp_name][round].keys():
            self.exp_dicts[exp_name][round][node] = {}

        # Create metric entry if needed
        if metric not in self.exp_dicts[exp_name][round][node].keys():
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
            exp (str): Experiment number

        Returns:
            Experiment logs
        """
        return self.exp_dicts[exp]

    def get_experiment_round_logs(self, exp: str, round: int) -> NodeLogsType:
        """
        Obtain logs for a round in an experiment.

        Args:
            exp (str): Experiment number
            round (int): Round number

        Returns:
            Round logs
        """
        return self.exp_dicts[exp][round]

    def get_experiment_round_node_logs(self, exp: str, round: int, node: str) -> NodeLogsType:
        """
        Obtain logs for a node in an experiment.

        Args:
            exp (str): Experiment number
            node (str): Node name

        Returns:
            Node logs
        """
        return self.exp_dicts[exp][round][node]


GlobalLogsType = Dict[str, NodeLogsType]


class GlobalMetricStorage:
    """
    "experiment":{
        "node_name": {
            "metric": [(round, value), ...]
        }
    }
    """

    def __init__(self):
        self.exp_dicts: GlobalLogsType = {}
        self.lock = Lock()

    def add_log(
        self, exp_name: str, round: int, metric: str, node: str, val: Union[int, float]
    ) -> None:
        # Lock
        self.lock.acquire()

        # Create Experiment if needed
        if exp_name not in self.exp_dicts.keys():
            self.exp_dicts[exp_name] = {}

        # Create node entry if needed
        if node not in self.exp_dicts[exp_name].keys():
            self.exp_dicts[exp_name][node] = {}

        # Create metric entry if needed
        if metric not in self.exp_dicts[exp_name][node].keys():
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
            exp (str): Experiment number

        Returns:
            Experiment logs
        """
        return self.exp_dicts[exp]

    def get_experiment_node_logs(self, exp: str, node: str) -> MetricsType:
        """
        Obtain logs for a node in an experiment.

        Args:
            exp (str): Experiment number
            node (str): Node name

        Returns:
            Node logs
        """
        return self.exp_dicts[exp][node]
