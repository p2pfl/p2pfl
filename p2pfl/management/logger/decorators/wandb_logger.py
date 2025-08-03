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

import contextlib
import os
from typing import Any, Literal

try:
    import wandb
    from wandb.sdk.wandb_run import Run  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    Run = None  # type: ignore
    WANDB_AVAILABLE = False

from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger


class WandbLogger(LoggerDecorator):
    """WandB Logger Decorator that can be configured at runtime."""

    _run: "Run" | None = None

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the WandbLogger decorator."""
        super().__init__(p2pflogger)
        self._project: str = "p2pfl"  # Default project
        self._entity: str | None = None
        self._config: dict[str, Any] = {}
        self._run_name: str | None = None
        self._wandb_enabled: bool = WANDB_AVAILABLE

        if WANDB_AVAILABLE:
            self._init_wandb()
        else:
            super().debug("WandbLogger", "WandB not available. Logging will be disabled for WandB.")

    def connect(self, **kwargs: Any) -> None:
        """
        Connect and setup WandB logging.

        Args:
            **kwargs: Connection parameters. Expected keys:
                - project: The project name (or WANDB_PROJECT env var, default: "p2pfl")
                - entity: The entity/username (or WANDB_ENTITY env var)
                - experiment: The p2pfl experiment object
                - run_name: A short display name for this run
                - mode: WandB mode ("online", "offline", "disabled")

        """
        if not self._wandb_enabled:
            super().debug("WandbLogger", "WandB not available or disabled. Skipping WandB setup.")
            return

        # Get parameters from kwargs or environment variables
        project = kwargs.get("project") or os.environ.get("WANDB_PROJECT") or "p2pfl"
        entity = kwargs.get("entity") or os.environ.get("WANDB_ENTITY")
        experiment = kwargs.get("experiment")
        run_name = kwargs.get("run_name")
        mode = kwargs.get("mode")

        self._project = project
        if entity:
            self._entity = entity

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

        # Initialize wandb if not already done
        if WandbLogger._run is None:
            self._init_wandb(mode=mode)

    def _init_wandb(self, mode: Literal["online", "offline", "disabled"] | None = None) -> None:
        """Initialize a wandb run."""
        if not WANDB_AVAILABLE:
            return

        if WandbLogger._run is not None:
            return

        try:
            # Determine wandb mode based on environment and configuration
            if mode is None:
                mode = self._determine_wandb_mode()

            WandbLogger._run = wandb.init(  # type: ignore
                project=self._project,
                name=self._run_name,
                config=self._config,
                mode=mode,
            )

            if mode == "disabled":
                self._wandb_enabled = False
                super().debug("WandbLogger", "WandB initialized in disabled mode")
            elif mode == "offline":
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
            # Try to get API key from environment or wandb settings
            api_key = os.environ.get("WANDB_API_KEY")
            if not api_key:
                # Try to get from wandb settings if available
                with contextlib.suppress(AttributeError, Exception):
                    api_key = wandb.api.api_key  # type: ignore

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

    def log_metric(self, addr: str, metric: str, value: float, step: int | None = None, round: int | None = None) -> None:
        """Log a metric to wandb."""
        # Ensure W&B is initialized and enabled
        if not self._wandb_enabled or WandbLogger._run is None:
            return super().log_metric(addr, metric, value, step=step, round=round)

        try:
            log_dict = {f"{addr}/{metric}": value}

            if step is not None:
                wandb.log(log_dict, step=round)  # type: ignore
            else:
                wandb.log(log_dict)  # type: ignore
        except Exception as e:
            super().warning(addr, f"Failed to log to W&B: {e}")

        super().log_metric(addr, metric, value, step=step, round=round)

    def finish(self) -> None:
        """Finish the wandb run."""
        if self._wandb_enabled and WandbLogger._run is not None:
            try:
                wandb.finish()  # type: ignore
            except Exception as e:
                super().warning("WandbLogger", f"Failed to finish WandB run: {e}")
            WandbLogger._run = None

        super().finish()
