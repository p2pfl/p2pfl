from typing import Dict, List, Optional, Tuple, Union
from pytorch_lightning.loggers.logger import Logger

# exp logs type
# - Experiments: List (AllLogsType)
#   - Experiment: Dict Nodes (ExpLogsType)
#     - Metrics: Dict Metric (list)  (NodeLogsType)
MetricsType = Dict[str, List[Tuple[int, float]]]
ExpLogsType = Dict[str, MetricsType]
AllLogsType = Dict[str, ExpLogsType]

# Predefined types + Logs for all experiments in a node
NodeLogsType = Dict[str, MetricsType]
LogsUnionType = Union[AllLogsType, ExpLogsType, MetricsType, NodeLogsType]


class FederatedLogger(Logger):
    """
    Logger for Federated Learning

    Args:
        node_name (str): Name of the node

    Attributes:
        self_name (str): Name of the node
        exp_dicts (list): List of experiments
        actual_exp (int): Actual experiment
        round (int): Actual round
        local_step (int): Actual local step
        global_step (int): Actual global step
    """

    def __init__(self, node_name: str) -> None:
        super().__init__()
        self.self_name = node_name

        # Create a dict
        self.exp_dicts: AllLogsType = {}
        self.actual_exp = -1
        self.actual_exp_name = ""

        # FL information
        self.round = 0
        self.local_step = 0
        self.global_step = 0

    def create_new_exp(self, name: Optional[str] = None) -> None:
        """
        Create a new experiment
        """
        # Create a new experiment
        self.actual_exp += 1
        if name is None:
            # Create a new experiment name
            name = f"exp_{self.actual_exp}"
        else:
            # Check if the name is already in use
            if name in self.exp_dicts.keys():
                raise ValueError(f"Experiment name {name} already in use.")
        self.actual_exp_name = name
        # Add a new experiment to the dict
        self.exp_dicts[self.actual_exp_name] = {self.self_name: {}}

    @property
    def name(self) -> None:
        pass

    @property
    def version(self) -> None:
        pass

    def log_hyperparams(self, params: dict) -> None:
        pass

    def get_logs(
        self, node: Optional[str] = None, exp: Optional[str] = None
    ) -> LogsUnionType:
        """
        Obtain logs.

        Args:
            node (str): Node name
            exp (int): Experiment number

        Returns:
            Logs
        """
        if exp is None:
            if node is None:
                return self.exp_dicts
            return {
                exp: self.exp_dicts[exp][node]
                for exp in self.exp_dicts
                if node in self.exp_dicts[exp].keys()
            }
        else:
            if node is None:
                return self.exp_dicts[exp]
            return self.exp_dicts[exp][node]

    def __add_log(
        self, metric: str, node: str, val: Union[int, float], step: int
    ) -> None:
        # Create node entry if needed
        if node not in self.exp_dicts[self.actual_exp_name].keys():
            self.exp_dicts[self.actual_exp_name][node] = {}
        # Create metric entry if needed
        if metric not in self.exp_dicts[self.actual_exp_name][node].keys():
            self.exp_dicts[self.actual_exp_name][node][metric] = [(step, val)]
        else:
            self.exp_dicts[self.actual_exp_name][node][metric].append((step, val))

    def log_metrics(self, metrics: dict, step: int, name: Optional[str] = None) -> None:
        """
        Log metrics (in a pytorch format).
        """
        if name is None:
            name = self.self_name

        # FL round information
        self.local_step = step
        __step = self.global_step + self.local_step

        # Log FL-Round by Step
        self.__add_log("fl_round", name, self.round, __step)

        # metrics -> dictionary of metric names and values
        for k, v in metrics.items():
            self.__add_log(k, name, v, __step)

    def log_round_metric(
        self,
        metric: str,
        value: Union[int, float],
        round: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Log a metric for a round.
        """
        if name is None:
            name = self.self_name

        if round is None:
            round = self.round

        self.__add_log(metric, name, value, round)

    def save(self) -> None:
        # Optional. Any code necessary to save logger data goes here
        pass

    def finalize(self, status: str) -> None:
        pass

    def finalize_round(self) -> None:
        """
        Finalize a round: update global step, local step and round.
        """
        # Finish Round
        self.global_step = self.global_step + self.local_step
        self.local_step = 0
        self.round = self.round + 1
