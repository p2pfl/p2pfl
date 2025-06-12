#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

"""WandB Logger Decorator."""

from typing import Any, Dict, Optional

import wandb

from p2pfl.experiment import Experiment
from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger


class WandbLogger(LoggerDecorator):
    """WandB Logger Decorator that can be configured at runtime."""

    def __init__(self, p2pflogger: P2PFLogger):
        """Initializes the WandbLogger decorator."""
        super().__init__(p2pflogger)
        self._project: str = "p2pfl"  # Default project
        self._entity: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._run_name: Optional[str] = None
        self._wandb_initialized = False
        self._experiment_configured = False

        self._init_wandb()
    def setup_wandb(
        self,
        project: str = "p2pfl",
        config: Optional[Dict[str, Any]] = None,
        experiment: Optional[Experiment] = None,
        run_name: Optional[str] = None,
    ):
        """
        Configure and initialize WandB. This should be called before any logging.

        Args:
            project (str): The name of the project where you're sending the new run.
            config (Dict, optional): A dictionary of hyperparameters for your run.
            entity (str, optional): An entity is a username or team name where you're sending runs.
            experiment (Experiment, optional): The p2pfl experiment object.
            run_name (str, optional): A short display name for this run.
        """
        self._project = project
        if config:
            self._config.update(config)
        self._run_name = run_name

        self._init_wandb(experiment=experiment)

    def _init_wandb(self, experiment: Optional[Experiment] = None):
        """Initializes a wandb run."""
        if self._wandb_initialized:
            return

        wandb.init(
            project=self._project,
            name=self._run_name,
            config=self._config,
            reinit=True,
        )
        self._wandb_initialized = True

    def log_metric(self, addr: str, metric: str, value: float, step: Optional[int] = None, round: Optional[int] = None):
        """Logs a metric to wandb."""
        # Ensure W&B is initialized
        if not self._wandb_initialized:
            return
            
        if self._wandb_initialized:
            try:
                log_dict = {f"{addr}/{metric}": value}
                if round is not None:
                    log_dict["round"] = round
                if step is not None:
                    wandb.log(log_dict, step=step)
                else:
                    wandb.log(log_dict)
            except Exception as e:
                print(f"[WARNING] Failed to log to W&B: {e}")

        return super().log_metric(addr, metric, value, step=step, round=round)

    def finish(self):
        """Finishes the wandb run."""
        if self._wandb_initialized:
            wandb.finish()
            self._wandb_initialized = False
            self._experiment_configured = False