from pytorch_lightning.loggers.base import LightningLoggerBase

class FederatedLogger(LightningLoggerBase):
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
    def __init__(self, node_name):
        super().__init__()
        self.self_name = node_name

        # Create a dict
        self.exp_dicts = []
        self.actual_exp = -1

        # FL information
        self.round = 0
        self.local_step = 0
        self.global_step = 0

    def create_new_exp(self):
        """
        Create a new experiment
        """
        self.exp_dicts.append({self.self_name: {}})
        self.actual_exp += 1

    @property
    def name(self):
        pass

    @property
    def version(self):
        pass

    def log_hyperparams(self, params):
        pass

    def get_logs(self, node=None, exp=None):
        """
        Obtain logs.

        Args:
            node (str): Node name
            exp (int): Experiment number

        Returns:
            Logs
        """
        if exp is None and node is None:
            return self.exp_dicts
        if node is None:
            return self.exp_dicts[exp]
        if exp is None:
            return self.exp_dicts[:][node]

        return self.exp_dicts[exp][node]

    def __add_log(self, metric, node, val, step):
        # Create node entry if needed
        if node not in self.exp_dicts[self.actual_exp].keys():
            self.exp_dicts[self.actual_exp][node] = {}
        # Create metric entry if needed
        if metric not in self.exp_dicts[self.actual_exp][node].keys():
            self.exp_dicts[self.actual_exp][node][metric] = [(step, val)]
        else:
            self.exp_dicts[self.actual_exp][node][metric].append((step, val))

    def log_metrics(self, metrics, step, name=None):
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

    def log_round_metric(self, metric, value, round=None, name=None):
        """
        Log a metric for a round.
        """
        if name is None:
            name = self.self_name

        if round is None:
            round = self.round

        self.__add_log(metric, name, value, round)

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    def finalize(self, status):
        pass

    def finalize_round(self):
        """
        Finalize a round: update global step, local step and round.
        """
        # Finish Round
        self.global_step = self.global_step + self.local_step
        self.local_step = 0
        self.round = self.round + 1
