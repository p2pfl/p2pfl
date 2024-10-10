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
"""Node state."""

import threading
from typing import Dict, List, Optional


class NodeState:
    """
    Class to store the main state of a learning node.

    Attributes:
        addr(str): The address of the node.
        status(str): The status of the node.
        actual_exp_name(str): The name of the experiment.
        round(int): The current round.
        total_rounds(int): The total rounds of the experiment.
        simulation(bool): If the node is a simulation.
        learner(NodeLearner): The learner of the node.
        models_aggregated(Dict[str, List[str]]): The models aggregated by the node.
        nei_status(Dict[str, int]): The status of the neighbors.
        train_set(List[str]): The train set of the node.
        train_set_votes(Dict[str, Dict[str, int]]): The votes of the train set.
        train_set_votes_lock(threading.Lock): The lock for the train set votes.
        start_thread_lock(threading.Lock): The lock for the start thread.
        wait_votes_ready_lock(threading.Lock): The lock for the wait votes ready.
        model_initialized_lock(threading.Lock): The lock for the model initialized.

    Args:
        addr: The address of the node.

    """

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
        self.experiment_config = None  # NOT IMPLEMENTED YET

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
        # initally set the event, gets clearend in wait_agg_models_stage, wait is called after
        self.wait_aggregated_model_event = threading.Event()
        self.wait_aggregated_model_event.set()

        # puede quedar guay el privatizar todos los locks y meter métodos que al mismo tiempo seteen un estado (string)

    def set_experiment(self, exp_name: str, total_rounds: int) -> None:
        """
        Set the experiment name.

        Args:
            exp_name: The name of the experiment.
            total_rounds: The total rounds of the experiment.

        """
        self.status = "Learning"
        self.actual_exp_name = exp_name
        self.total_rounds = total_rounds
        self.round = 0

    def increase_round(self) -> None:
        """
        Increase the round number.

        Raises:
            ValueError: If the round is not initialized.

        """
        if self.round is None:
            raise ValueError("Round not initialized")
        self.round += 1
        self.models_aggregated = {}

    def clear(self) -> None:
        """Clear the state."""
        type(self).__init__(self, self.addr)
