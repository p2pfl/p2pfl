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
from typing import Any

try:
    import wandb
    from wandb.sdk.wandb_run import Run  # type: ignore

except ImportError:
    wandb = None  # type: ignore
    Run = None  # type: ignore

from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger


class WandbLogger(LoggerDecorator):
    """WandB Logger Decorator that can be configured at runtime."""

    _run: Run | None = None

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the WandbLogger decorator."""
        super().__init__(p2pflogger)
        self._wandb_enabled: bool = wandb is not None
        self.connect()

    def connect(
        self,
        wandb_api_key: str | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_tags: list[str] | str | None = None,
        wandb_notes: str | None = None,
        wandb_group: str | None = None,
        experiment: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Connect and setup WandB logging.

        Args:
            wandb_api_key: The API key (or WANDB_API_KEY env var)
            wandb_project: The project name (or WANDB_PROJECT env var, default: "p2pfl")
            wandb_entity: The entity/username (or WANDB_ENTITY env var)
            wandb_run_name: A short display name for this run (or WANDB_RUN_NAME env var)
            wandb_tags: List of tags for this run (or WANDB_TAGS env var as comma-separated)
            wandb_notes: Notes about this run (or WANDB_NOTES env var)
            wandb_group: Group for this run (or WANDB_RUN_GROUP env var)
            experiment: The p2pfl experiment object
            **kwargs: Additional parameters (for compatibility)

        """
        # Skip if already connected
        if WandbLogger._run is not None:
            super().debug("WandbLogger", "WandB already connected, skipping initialization")
            return

        # Get parameters from function args or environment variables
        api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
        project = wandb_project or os.environ.get("WANDB_PROJECT")
        entity = wandb_entity or os.environ.get("WANDB_ENTITY")
        run_name = wandb_run_name or os.environ.get("WANDB_RUN_NAME")
        notes = wandb_notes or os.environ.get("WANDB_NOTES")
        group = wandb_group or os.environ.get("WANDB_RUN_GROUP")

        # Handle tags (can be list or comma-separated string)
        tags = wandb_tags or os.environ.get("WANDB_TAGS")
        if isinstance(tags, str):
            tags_list: list[str] | None = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif isinstance(tags, list):
            tags_list = tags
        else:
            tags_list = None

        # Extract experiment config if provided
        config = {}
        if experiment:
            if not run_name:
                run_name = experiment.exp_name
            experiment_config = {
                "exp_name": experiment.exp_name,
                "total_rounds": experiment.total_rounds,
                "dataset_name": experiment.dataset_name,
                "model_name": experiment.model_name,
                "aggregator_name": experiment.aggregator_name,
                "framework_name": experiment.framework_name,
                "batch_size": experiment.batch_size,
                "learning_rate": experiment.learning_rate,
                "epochs_per_round": experiment.epochs_per_round,
            }
            experiment_config = {k: v for k, v in experiment_config.items() if v is not None}
            config.update(experiment_config)

        # Check if wandb is available (imported successfully)
        if not self._wandb_enabled:
            if project is not None or entity is not None:
                super().warning("WandbLogger", "WandB library not installed but project or entity provided. Install wandb to enable logging.")
            return

        # Check if API key is available
        if api_key is None and "WANDB_API_KEY" not in os.environ:
            if project is not None or entity is not None:
                super().warning("WandbLogger", "WandB project or entity provided but no API key found. Disabling WandB logging.")
            return

        # Only create a new run if none exists
        if WandbLogger._run is not None:
            return

        try:
            # Set API key if provided
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key

            # Prepare init parameters
            init_params: dict[str, Any] = {
                "project": project or "p2pfl",  # Default project name
                "config": config or {},
            }

            # Add optional parameters if provided
            if entity:
                init_params["entity"] = entity
            if run_name:
                init_params["name"] = run_name
            if tags_list:
                init_params["tags"] = tags_list
            if notes:
                init_params["notes"] = notes
            if group:
                init_params["group"] = group

            WandbLogger._run = wandb.init(**init_params)  # type: ignore
            super().debug("WandbLogger", f"WandB initialized successfully with project '{init_params['project']}'")

        except Exception as e:
            super().warning("WandbLogger", f"Failed to initialize WandB: {e}. Disabling WandB logging.")
            WandbLogger._run = None
            self._wandb_enabled = False

    def log_metric(self, addr: str, metric: str, value: float, step: int | None = None, round: int | None = None) -> None:
        """Log a metric to wandb."""
        # Ensure W&B is initialized
        if WandbLogger._run is None:
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
        if WandbLogger._run is not None:
            try:
                wandb.finish()  # type: ignore
            except Exception as e:
                super().warning("WandbLogger", f"Failed to finish WandB run: {e}")
            WandbLogger._run = None

        super().finish()
