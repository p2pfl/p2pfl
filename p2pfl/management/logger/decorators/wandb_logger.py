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

from p2pfl.experiment import Experiment
from p2pfl.management.logger.decorators.logger_decorator import LoggerDecorator
from p2pfl.management.logger.logger import P2PFLogger


class WandbLogger(LoggerDecorator):
    """WandB Logger Decorator that can be configured at runtime."""

    def __init__(self, p2pflogger: P2PFLogger):
        """Initialize the WandbLogger decorator."""
        super().__init__(p2pflogger)
        self._wandb_enabled: bool = wandb is not None
        self._run: Run | None = None
        self._connected: bool = False

        # Connection parameters (stored for later use)
        self._api_key: str | None = None
        self._project: str | None = None
        self._entity: str | None = None
        self._tags: list[str] | None = None
        self._notes: str | None = None
        self._group: str | None = None

        # Try to auto-connect using environment variables
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
        **kwargs: Any,
    ) -> None:
        """
        Store WandB connection parameters for later use.

        Args:
            wandb_api_key: The API key (or WANDB_API_KEY env var)
            wandb_project: The project name (or WANDB_PROJECT env var, default: "p2pfl")
            wandb_entity: The entity/username (or WANDB_ENTITY env var)
            wandb_run_name: A short display name for this run (or WANDB_RUN_NAME env var)
            wandb_tags: List of tags for this run (or WANDB_TAGS env var as comma-separated)
            wandb_notes: Notes about this run (or WANDB_NOTES env var)
            wandb_group: Group for this run (or WANDB_RUN_GROUP env var)
            **kwargs: Additional parameters (for compatibility)

        """
        # Check if wandb is available
        if not self._wandb_enabled:
            if wandb_project is not None or wandb_entity is not None:
                super().debug("WandbLogger", "WandB library not installed but project or entity provided. Install wandb to enable logging.")
            return

        # Get parameters from function args or environment variables
        self._api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
        self._project = wandb_project or os.environ.get("WANDB_PROJECT") or "p2pfl"
        self._entity = wandb_entity or os.environ.get("WANDB_ENTITY")
        self._group = wandb_group or os.environ.get("WANDB_RUN_GROUP")
        self._notes = wandb_notes or os.environ.get("WANDB_NOTES")

        # Store run name separately (might be overridden by experiment)
        self._run_name = wandb_run_name or os.environ.get("WANDB_RUN_NAME")

        # Handle tags (can be list or comma-separated string)
        tags = wandb_tags or os.environ.get("WANDB_TAGS")
        if isinstance(tags, str):
            self._tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif isinstance(tags, list):
            self._tags = tags
        else:
            self._tags = None

        # Check if API key is available
        if self._api_key is None and "WANDB_API_KEY" not in os.environ:
            if wandb_project is not None or wandb_entity is not None:
                super().warning("WandbLogger", "WandB project or entity provided but no API key found.")
            return

        # Mark as connected (credentials stored)
        self._connected = True
        super().debug("WandbLogger", "WandB credentials stored successfully")

    def experiment_started(self, node: str, experiment: Experiment) -> None:
        """
        Initialize a WandB run when experiment starts.

        Args:
            node: The node address.
            experiment: The experiment object containing metadata.

        """
        # Check if we can initialize
        if not self._wandb_enabled:
            super().debug("WandbLogger", "WandB not available, skipping experiment initialization")
            return super().experiment_started(node, experiment)

        if not self._connected:
            super().debug("WandbLogger", "WandB not connected. Call connect() first or set environment variables.")
            return super().experiment_started(node, experiment)

        # Skip if already running (handles round increases)
        if self._run is not None:
            super().debug("WandbLogger", "WandB run already active, updating round")
            return super().experiment_started(node, experiment)

        try:
            # Set API key if provided
            if self._api_key:
                os.environ["WANDB_API_KEY"] = self._api_key

            # Extract experiment config using the to_dict method
            config = experiment.to_dict(exclude_none=True)

            # Prepare init parameters
            init_params: dict[str, Any] = {
                "project": self._project,
                "config": config,
                "name": self._run_name or experiment.exp_name,  # Use experiment name if no run name provided
            }

            # Add optional parameters if provided
            if self._entity:
                init_params["entity"] = self._entity
            if self._tags:
                init_params["tags"] = self._tags
            if self._notes:
                init_params["notes"] = self._notes
            if self._group:
                init_params["group"] = self._group

            self._run = wandb.init(**init_params)  # type: ignore
            super().debug("WandbLogger", f"WandB run initialized successfully for experiment '{experiment.exp_name}'")

        except Exception as e:
            super().warning("WandbLogger", f"Failed to initialize WandB run: {e}")
            self._run = None

        # Call parent's experiment_started
        super().experiment_started(node, experiment)

    def log_metric(self, addr: str, metric: str, value: float, step: int | None = None, round: int | None = None) -> None:
        """Log a metric to wandb."""
        # Ensure W&B run is initialized
        if self._run is None:
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
        if self._run is not None:
            try:
                wandb.finish()  # type: ignore
                super().debug("WandbLogger", "WandB run finished successfully")
            except Exception as e:
                super().warning("WandbLogger", f"Failed to finish WandB run: {e}")
            self._run = None

        super().finish()
