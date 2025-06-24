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

import os
from typing import Any, Dict, Optional, Literal

try:
    import wandb
    from wandb.sdk.wandb_run import Run

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # type: ignore
    Run = None # type: ignore
    WANDB_AVAILABLE = False

from p2pfl.experiment import Experiment
from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger


class WandbLogger(LoggerDecorator):
    """WandB Logger Decorator that can be configured at runtime."""

    _run: Optional["Run"] = None

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the WandbLogger decorator."""
        super().__init__(p2pflogger)
        self._project: str = "p2pfl"  # Default project
        self._entity: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._run_name: Optional[str] = None
        self._wandb_enabled: bool = WANDB_AVAILABLE

        if WANDB_AVAILABLE:
            self._init_wandb()
        else:
            super().debug("WandbLogger", "WandB not available. Logging will be disabled for WandB.")

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
        if not self._wandb_enabled:
            super().debug("WandbLogger", "WandB not available or disabled. Skipping WandB setup.")
            return

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
        if not WANDB_AVAILABLE:
            return

        if WandbLogger._run is not None:
            return

        try:
            # Determine wandb mode based on environment and configuration
            wandb_mode = self._determine_wandb_mode()

            WandbLogger._run = wandb.init(
                project=self._project,
                name=self._run_name,
                config=self._config,
                mode=wandb_mode,
            )

            if wandb_mode == "disabled":
                self._wandb_enabled = False
                super().debug("WandbLogger", "WandB initialized in disabled mode")
            elif wandb_mode == "offline":
                super().debug("WandbLogger", "WandB initialized in offline mode")
            else:
                super().debug("WandbLogger", "WandB initialized in online mode")

        except Exception as e:
            super().warning("WandbLogger", f"Failed to initialize WandB: {e}. Disabling WandB logging.")
            WandbLogger._run = None
            self._wandb_enabled = False

    def _determine_wandb_mode(self) -> Literal["online", "offline", "disabled"]:
        """
        Determine the appropriate wandb mode based on environment and configuration.

        Returns:
            Literal["online", "offline", "disabled"]: The wandb mode

        """
        # Check if API key is configured
        try:
            api_key = wandb.api.api_key
            if api_key is None or api_key == "":
                # No API key configured, use offline mode in CI/non-interactive environments
                if os.getenv("CI") or not os.isatty(0):
                    return "offline"
                else:
                    return "disabled"
        except Exception:
            if os.getenv("CI") or not os.isatty(0):
                return "offline"
            else:
                return "disabled"

        return "online"

    def log_metric(self, addr: str, metric: str, value: float, step: Optional[int] = None, round: Optional[int] = None) -> None:
        """Log a metric to wandb."""
        # Ensure W&B is initialized and enabled
        if not self._wandb_enabled or WandbLogger._run is None:
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
        if self._wandb_enabled and WandbLogger._run is not None:
            try:
                wandb.finish()
            except Exception as e:
                super().warning("WandbLogger", f"Failed to finish WandB run: {e}")
            WandbLogger._run = None

        super().finish()
