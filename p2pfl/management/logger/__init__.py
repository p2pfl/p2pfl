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

from p2pfl.management.logger.decorators.async_logger import AsyncLogger
from p2pfl.management.logger.decorators.file_logger import FileLogger
from p2pfl.management.logger.decorators.singleton_logger import SingletonLogger
from p2pfl.management.logger.decorators.web_logger import WebP2PFLogger
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.utils.check_ray import ray_installed

# Check if 'ray' is installed in the Python environment
if ray_installed():
    from p2pfl.management.logger.decorators.ray_logger import RayP2PFLogger

    # Logger actor singleton
    logger = SingletonLogger(RayP2PFLogger(WebP2PFLogger(FileLogger(P2PFLogger(disable_locks=True)))))
else:
    logger = SingletonLogger(WebP2PFLogger(FileLogger(AsyncLogger(P2PFLogger(disable_locks=False)))))
