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
from wandb.sdk.wandb_run import Run


class WandbLogger(LoggerDecorator):
    """WandB Logger Decorator that can be configured at runtime."""

    _run: Optional[Run] = None

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the WandbLogger decorator."""
        super().__init__(p2pflogger)
        self._project: str = "p2pfl"  # Default project
        self._entity: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._run_name: Optional[str] = None

        self._init_wandb()

    def setup_wandb(
        self,
        project: str = "p2pfl",
        experiment: Optional[Experiment] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Configure and initialize WandB. This should be called before any logging.

        Args:
            project (str): The name of the project where you're sending the new run.
            experiment (Experiment, optional): The p2pfl experiment object.
            run_name (str, optional): A short display name for this run.

        """
        self._project = project

        if experiment and WandbLogger._run is not None:
            self._run_name = experiment.exp_name
            experiment_config = {
                "exp_name": experiment.exp_name,
                "total_rounds": experiment.total_rounds,
                "dataset_name": experiment.dataset_name,
                "model_name": experiment.model_name,
                "aggregator_name": experiment.aggregator_name,
                "framework_name": experiment.framework_name,
            }
            experiment_config = {k: v for k, v in experiment_config.items() if v is not None}
            self._config.update(experiment_config)

            WandbLogger._run.config.update(experiment_config)  # type: ignore
            WandbLogger._run.name = experiment.exp_name

            super().debug("WandbLogger", f"W&B run initialized with config: {experiment_config}")

        elif run_name:
            self._run_name = run_name

    def _init_wandb(self) -> None:
        """Initialize a wandb run."""
        if WandbLogger._run is not None:
            return

        WandbLogger._run = wandb.init(
            project=self._project,
            name=self._run_name,
            config=self._config,
        )

    def log_metric(self, addr: str, metric: str, value: float, step: Optional[int] = None, round: Optional[int] = None) -> None:
        """Log a metric to wandb."""
        # Ensure W&B is initialized
        if WandbLogger._run is None:
            return super().log_metric(addr, metric, value, step=step, round=round)

        try:
            log_dict = {f"{addr}/{metric}": value}

            if step is not None:
                wandb.log(log_dict, step=round)
            else:
                wandb.log(log_dict)
        except Exception as e:
            super().warning(addr, f"Failed to log to W&B: {e}")

        super().log_metric(addr, metric, value, step=step, round=round)

    def finish(self) -> None:
        """Finish the wandb run."""
        if WandbLogger._run is not None:
            wandb.finish()
            WandbLogger._run = None

        super().finish()
