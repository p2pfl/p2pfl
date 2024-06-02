from pytorch_lightning.loggers.logger import Logger
from p2pfl.management.logger import Logger as P2PLogger


class FederatedLogger(Logger):
    """
    Pytorch Lightning Logger for Federated Learning. Handles local training loggin.
    """

    def __init__(self, node_name: str) -> None:
        super().__init__()
        self.self_name = node_name

    @property
    def name(self) -> None:
        pass

    @property
    def version(self) -> None:
        pass

    def log_hyperparams(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int) -> None:
        """
        Log metrics (in a pytorch format).
        """
        for k, v in metrics.items():
            P2PLogger.log_metric(self.self_name, k, v, step)

    def save(self) -> None:
        # Optional. Any code necessary to save logger data goes here
        pass

    def finalize(self, status: str) -> None:
        pass
