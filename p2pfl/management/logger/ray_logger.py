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

import ray

from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.management.logger.logger_decorator import P2PFLoggerDecorator

@ray.remote
class RayP2PFLoggerActor(P2PFLoggerDecorator):
    """Actor to add remote logging capabilities to a logger class."""
    
    _p2pflogger: P2PFLogger = None

    def __init__(self, p2pflogger: P2PFLogger) -> None:
        self._p2pflogger = p2pflogger



class RayP2PFLogger:
    def __init__(self, p2pflogger: P2PFLogger):
        """
        Initialize the wrapper with a Ray actor instance.

        Args:
            logger: The logger to be wrapped.
        """
        self.ray_actor = RayP2PFLoggerActor.options(name="p2pfl_logger", lifetime="detached", get_if_exists=True).remote(p2pflogger)

    def __getattr__(self, name):
        """
        Intercept method calls and automatically convert them to remote calls.
        
        Args:
            name: The name of the method being called.
        
        Returns:
            A function that invokes the corresponding remote method.
        """
        # Check if name is "ray_actor" or any other known attribute before delegating to __getattr__
        if name == "ray_actor":
            return object.__getattribute__(self, "ray_actor")

        # Get the actual method from the Ray actor
        method = getattr(self.ray_actor, name)
        
        # Return a wrapper that automatically calls .remote() on the method
        def remote_method(*args, **kwargs):
            # Try to retrieve the result if it's a getter method
            if method._method_name.startswith("get_"):
                return ray.get(method.remote(*args, **kwargs))
            return method.remote(*args, **kwargs)

        return remote_method
