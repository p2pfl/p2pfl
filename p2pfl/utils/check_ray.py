#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
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

"""Check if ray is installed."""

import importlib
import os

from p2pfl.settings import Settings


def ray_installed() -> bool:
    """Check if ray is installed."""
    if os.environ.get("P2PFL_DISABLE_RAY") or Settings.general.DISABLE_RAY:
        return False

    os.environ["RAY_DEDUP_LOGS"] = "0"

    if importlib.util.find_spec("ray") is not None:
        try:
            import ray

            # If ray not initialized, initialize it
            if not ray.is_initialized():
                init_kwargs = {
                    "namespace": "p2pfl",
                    "include_dashboard": False,
                    "logging_level": Settings.general.LOG_LEVEL,
                    "logging_config": ray.LoggingConfig(encoding="TEXT", log_level=Settings.general.LOG_LEVEL),
                }
                ray.init(**init_kwargs)
            return True
        except (AttributeError, TypeError, RuntimeError) as e:
            print(f"[p2pfl] Ray found but not usable: {e}")
            return False
    return False
