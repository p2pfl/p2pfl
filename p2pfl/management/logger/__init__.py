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

"""Provides a logger singleton that can be used to log messages from different parts of the codebase."""

import importlib.util
from typing import Any

from p2pfl.management.logger.main_logger import MainP2PFLogger

# Check if 'ray' is installed in the Python environment
ray_installed = importlib.util.find_spec("ray") is not None

logger: Any

# Create the logger depending on the availability of 'ray'
if ray_installed:
    from p2pfl.management.logger.ray_logger import RayP2PFLogger

    # Logger actor singleton
    logger = RayP2PFLogger(MainP2PFLogger(disable_locks = True))
else:
    from p2pfl.management.logger.loggers.async_logger import AsyncLocalLogger

    # Logger actor singleton
    logger = AsyncLocalLogger(MainP2PFLogger(disable_locks = False))
