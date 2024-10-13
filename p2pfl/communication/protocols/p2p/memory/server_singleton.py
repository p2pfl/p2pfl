# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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

"""Server singleton."""

import threading


class ServerSingleton(dict):
    """Server singleton class."""

    def __new__(cls) -> "ServerSingleton":
        """Get or create an instance of the server singleton."""
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.termination_event = threading.Event()  # Initialize the threading event
        return cls.instance

    @classmethod
    def reset_instance(cls) -> "ServerSingleton":
        """Reset the instance of the server singleton."""
        if hasattr(cls, "instance"):
            cls.termination_event.set()
            del cls.instance
        return cls()

    @classmethod
    def wait_for_termination(cls) -> None:
        """Blocks until the server is signaled to terminate."""
        cls.termination_event.wait()
