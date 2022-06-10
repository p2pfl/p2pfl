import os
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.tensorboard import SummaryWriter
import torch

class FederatedTensorboardLogger(LightningLoggerBase):
    """
    Logger for PyTorch Lightning in federated learning. Training information persists in a local directory the diferent train rounds.
    """

    def __init__(self, dir, name = None , version = 0, **kwargs):
        super().__init__()
        self._name = "unknown_node"
        if name is not None:
            self._name = name

        self._version = version    
        self.writer = SummaryWriter(os.path.join(dir, self._name))

        # FL information
        self.round = 0
        self.step = 0
        self.actual_step = 0
        self.actual_round = 0
        
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
        self.actual_step = step
        step = step + self.step
        self.writer.add_scalar("fl_round", self.round, step)

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.writer.add_scalars(k, v, step)
            else:
                try:
                    self.writer.add_scalar(k, v, step)
                # todo: specify the possible exception
                except Exception as ex:
                    m = f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex

    @rank_zero_only
    def save(self):
        """
        """
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        """
        """
        # Finish Round
        self.round = self.round + 1
        self.log_metrics({"fl_round": self.round}, self.actual_step)
        # Update Steps
        self.step = self.actual_step