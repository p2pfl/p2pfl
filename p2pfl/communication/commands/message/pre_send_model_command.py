#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""PRE_SEND_MODEL command."""

import threading
import time
from typing import Optional

from p2pfl.communication.commands.command import Command
from p2pfl.settings import Settings


class PreSendModelCommand(Command):
    """Command used to notify a recipient node before a model is actually sent."""

    def __init__(self, addr: str) -> None:
        """Initialize the command."""
        self.addr = addr
        super().__init__()
        # Dict of sending models {cmd: [{model_hash: timestamp}]}
        self.sending_models: dict[str, dict[str, float]] = {}
        # Lock
        self.sending_models_lock = threading.Lock()

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "PRE_SEND_MODEL"

    def execute(self, source: str, round: int, *args, **kwargs) -> Optional[str]:
        """Execute the command."""
        if len(args) < 2:
            raise ValueError("Expected at least 2 args: original_cmd_intended and model_hash")

        # Get args
        cmd = args[0]
        hashes = self.sending_models.get(cmd, {})
        if hashes == {}:
            self.sending_models[cmd] = {}
        model_hash = [f"{str(hs)}-{round}" for hs in args[1:]]

        with self.sending_models_lock:
            # Clear outdated models
            hashes_to_delete = []
            for saved_hash, timestamp in hashes.items():
                if (time.time() - timestamp) > Settings.gossip.MODE_EXPECTATION_TIMEOUT:
                    hashes_to_delete.append(saved_hash)
            for saved_hash in hashes_to_delete:
                del self.sending_models[cmd][saved_hash]

            # Check if hash is already in sending_models
            compatible_hashes = True
            for new_hash in model_hash:
                if new_hash in hashes:
                    compatible_hashes = False
                    break

            if compatible_hashes:
                for new_hash in model_hash:
                    self.sending_models[cmd].update({new_hash: time.time()})
                return "true"
            else:
                return "false"
