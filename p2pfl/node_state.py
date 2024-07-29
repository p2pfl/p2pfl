#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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
"""Node state."""

import threading
from typing import Dict, List, Optional

from p2pfl.learning.learner import NodeLearner


class NodeState:
    """Class to store the main state of a learning node."""

    def __init__(self, addr: str) -> None:
        """Initialize the node state."""
        self.addr = addr
        self.status = "Idle"
        self.actual_exp_name: Optional[str] = None
        self.round: Optional[int] = None
        self.total_rounds: Optional[int] = None

        # Simulation
        self.simulation = False

        # Learning
        self.learner: Optional[NodeLearner] = None

        # Aggregator (TRATAR DE MOVERLO A LA CLASE AGGREGATOR)
        self.models_aggregated: Dict[str, List[str]] = {}

        # Other neis state (only round)
        self.nei_status: Dict[str, int] = {}

        # Train Set
        self.train_set: List[str] = []
        self.train_set_votes: Dict[str, Dict[str, int]] = {}

        # Locks
        self.train_set_votes_lock = threading.Lock()
        self.start_thread_lock = threading.Lock()
        self.wait_votes_ready_lock = threading.Lock()
        self.model_initialized_lock = threading.Lock()
        self.model_initialized_lock.acquire()

    def set_experiment(self, exp_name: str, total_rounds: int) -> None:
        """Set the experiment name."""
        self.status = "Learning"
        self.actual_exp_name = exp_name
        self.total_rounds = total_rounds
        self.round = 0

    def increase_round(self) -> None:
        """Increase the round number."""
        if self.round is None:
            raise ValueError("Round not initialized")
        self.round += 1
        self.models_aggregated = {}

    def clear(self) -> None:
        """Clear the state."""
        self.status = "Idle"
        self.actual_exp_name = None
        self.round = None
        self.total_rounds = None
