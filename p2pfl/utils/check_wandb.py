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

"""Check if wandb is available and properly configured."""

import contextlib
import importlib.util
import os


def check_wandb() -> bool:
    """
    Check if wandb is installed and available.

    Returns:
        True if wandb is available, False otherwise.

    """
    return importlib.util.find_spec("wandb") is not None


def is_wandb_configured() -> bool:
    """
    Check if wandb is properly configured with an API key.

    Returns:
        True if wandb is configured, False otherwise.

    """
    if not check_wandb():
        return False

    try:
        import wandb

        # Check if API key is available through environment variable
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key and api_key != "":
            return True

        # Try to get from wandb settings if available
        with contextlib.suppress(AttributeError, Exception):
            api_key = wandb.api.api_key  # type: ignore
            return api_key is not None and api_key != ""
        return False
    except Exception:
        return False


def should_use_wandb() -> bool:
    """
    Determine if wandb should be used based on availability and configuration.

    Returns:
        True if wandb should be used, False otherwise.

    """
    if os.getenv("WANDB_DISABLED", "false").lower() in ("true", "1", "yes"):
        return False

    if os.getenv("WANDB_MODE", "").lower() == "offline":
        return check_wandb()

    # For online mode, check if properly configured
    return check_wandb() and is_wandb_configured()
