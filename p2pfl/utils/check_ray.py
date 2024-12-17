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

"""Check if ray is installed."""

import importlib

from p2pfl.settings import Settings


def ray_installed() -> bool:
    """Check if ray is installed."""
    if Settings.DISABLE_RAY:
        return False

    if importlib.util.find_spec("ray") is not None:
        # Try to initialize ray
        import ray

        # If ray not initialized, initialize it
        if not ray.is_initialized():
            ray.init(
                namespace="p2pfl",
                # include_dashboard=False
            )
        return True
    return False
