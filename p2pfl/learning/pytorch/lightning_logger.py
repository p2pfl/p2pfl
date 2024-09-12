#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

"""Lightning Logger for P2PFL."""

from pytorch_lightning.loggers.logger import Logger

from p2pfl.management.logger import Logger as P2PLogger


class FederatedLogger(Logger):
    """
    Pytorch Lightning Logger for Federated Learning. Handles local training loggin.

    Args:
        node_name: Name of the node.

    """

    def __init__(self, node_name: str) -> None:
        """Initialize the logger."""
        super().__init__()
        self.self_name = node_name

    @property
    def name(self) -> None:
        """Name of the logger."""
        pass

    @property
    def version(self) -> None:
        """Version of the logger."""
        pass

    def log_hyperparams(self, params: dict) -> None:
        """Log hyperparameters."""
        pass

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics (in a pytorch format)."""
        for k, v in metrics.items():
            P2PLogger.log_metric(self.self_name, k, v, step)

    def save(self) -> None:
        """Save the logger."""
        pass

    def finalize(self, status: str) -> None:
        """Finalize the logger."""
        pass
