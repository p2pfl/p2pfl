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

import time

from p2pfl.communication.commands.command import Command
from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings


class PreSendModelCommand(Command):
    """Command used to notify a recipient node before a model is actually sent."""

    def __init__(self, node_state: NodeState) -> None:
        """Initialize the command."""
        super().__init__()
        self.node_state = node_state

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "PRE_SEND_MODEL"

    @staticmethod
    def remove_hashed(node_state: NodeState, cmd: str, hashes: list[str], round: int) -> None:
        """Remove hashes from sending_models."""
        with node_state.sending_models_lock:
            for hashed in [f"{str(hs)}-{round}" for hs in hashes]:
                del node_state.sending_models[cmd][hashed]

    def execute(self, source: str, round: int, *args, **kwargs) -> str | None:
        """Execute the command."""
        if len(args) < 2:
            raise ValueError("Expected at least 2 args: original_cmd_intended and model_hash")

        # Get args
        cmd = args[0]
        hashes = self.node_state.sending_models.get(cmd, {})
        if hashes == {}:
            self.node_state.sending_models[cmd] = {}
        model_hash = [f"{str(hs)}-{round}" for hs in args[1:]]

        with self.node_state.sending_models_lock:
            # Clear outdated models
            hashes_to_delete = []
            for saved_hash, timestamp in hashes.items():
                if (time.time() - timestamp) > Settings.gossip.MODE_EXPECTATION_TIMEOUT:
                    hashes_to_delete.append(saved_hash)
            for saved_hash in hashes_to_delete:
                del self.node_state.sending_models[cmd][saved_hash]

            # Check if hash is already in sending_models
            compatible_hashes = True
            for new_hash in model_hash:
                if new_hash in hashes:
                    compatible_hashes = False
                    break

            # ----- Bandwidth optimization -------
            # If add_model and node in trainset has full local state, refuse redundant weights
            if cmd == FullModelCommand.get_name() and self.node_state.addr in self.node_state.train_set:
                # Collect all unique models aggregated locally
                all_aggregated_models = set()
                for models_list in self.node_state.models_aggregated.values():
                    all_aggregated_models.update(models_list)

                # If this node has aggregated all trainset models, it has full weights locally
                if set(self.node_state.train_set) == all_aggregated_models:
                    return "false"

            if compatible_hashes:
                for new_hash in model_hash:
                    self.node_state.sending_models[cmd].update({new_hash: time.time()})
                return "true"
            else:
                return "false"
