import os
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.tensorboard import SummaryWriter
import torch

class FederatedTensorboardLogger(LightningLoggerBase):
    """
    Logger for PyTorch Lightning in federated learning. 
    It is made to save the information of the different trainings (in the different rounds) in the same graph.

    End of round determined with `finalize_round`

    Args:
        dir (str): Directory where the logs will be saved.
        name (str): Name of the node.
        version (int): Version of the experiment.
    """

    def __init__(self, dir, name = None, version=0, **kwargs):
        super().__init__()
        self._name = "unknown_node"
        self._version = version
        if name is not None:
            self._name = name

        # Create log directory
        dir = os.path.join(dir, self._name)
        # If exist the experiment, increment the version
        while os.path.exists(os.path.join(dir, "experiment_" + str(version))):
            version += 1
        # Create the writer
        self.writer = SummaryWriter(os.path.join(dir, "experiment_" + str(version)))

        # FL information
        self.round = 0
        self.local_step = 0
        self.global_step = 0
        
        self.writer.add_scalar("fl_round", self.round, self.global_step)

        
    @property
    def name(self):
        """
        """
        return self._name

    @property
    def version(self):
        """
        """
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params):
        """
        """
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """
        """

        # FL round information
        self.local_step = step
        __step = self.global_step + self.local_step

        # Log Round
        self.writer.add_scalar("fl_round", self.round, __step)
       
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.writer.add_scalars(k, v, __step)
            else:
                try:
                    self.writer.add_scalar(k, v, __step)
                # todo: specify the possible exception
                except Exception as ex:
                    m = f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex

    def log_scalar(self, key, value, round=None, name=None):
        """
        """
        if round is None:
            round = self.round

        if name is None:
            self.writer.add_scalar(key, value, round)
        else:
            self.writer.add_scalars(key, {name: value}, round)


    @rank_zero_only
    def save(self):
        """
        """
        # Optional. Any code necessary to save logger data goes here
        pass

    def finalize(self, status):
        """
        """
        pass

    def finalize_round(self):
        # Finish Round
        self.global_step = self.global_step + self.local_step
        self.local_step = 0
        self.round = self.round + 1

    def close(self):
        try:
            self.writer.close()
        except:
            pass