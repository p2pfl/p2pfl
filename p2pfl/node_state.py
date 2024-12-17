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

from p2pfl.experiment import Experiment


class NodeState:
    """
    Class to store the main state of a learning node.

    Attributes:
        addr(str): The address of the node.
        status(str): The status of the node.
        simulation(bool): If the node is a simulation.
        learner(Learner): The learner of the node.
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

    def __init__(self, addr: str, simulation: bool = False) -> None:
        """Initialize the node state."""
        self.addr = addr
        self.status = "Idle"

        # Simulation
        self.simulation = simulation

        # Learning
        self.experiment_config = None  # NOT IMPLEMENTED YET

        # Aggregator (TRATAR DE MOVERLO A LA CLASE AGGREGATOR)
        self.models_aggregated: Dict[str, List[str]] = {}

        # Other neis state (only round)
        self.nei_status: Dict[str, int] = {}

        # Train Set
        self.train_set: List[str] = []
        self.train_set_votes: Dict[str, Dict[str, int]] = {}

        # Actual experiment
        self.experiment: Optional[Experiment] = None

        # Locks
        self.train_set_votes_lock = threading.Lock()
        self.start_thread_lock = threading.Lock()
        self.wait_votes_ready_lock = threading.Lock()
        self.model_initialized_lock = threading.Lock()
        self.model_initialized_lock.acquire()
        self.aggregated_model_event = threading.Event()
        self.aggregated_model_event.set()

        # puede quedar guay el privatizar todos los locks y meter métodos que al mismo tiempo seteen un estado (string)

    @property
    def round(self) -> Optional[int]:
        """Get the round."""
        return self.experiment.round if self.experiment is not None else None

    @property
    def total_rounds(self) -> Optional[int]:
        """Get the total rounds."""
        return self.experiment.total_rounds if self.experiment is not None else None

    @property
    def exp_name(self) -> Optional[str]:
        """Get the actual experiment name."""
        return self.experiment.exp_name if self.experiment is not None else None

    def set_experiment(self, exp_name: str, total_rounds: int) -> None:
        """
        Start a new experiment.

        Attributes:
            exp_name: The name of the experiment.
            total_rounds: The total rounds of the experiment.

        """
        self.status = "Learning"
        self.experiment = Experiment(exp_name, total_rounds)

    def increase_round(self) -> None:
        """
        Increase the round number.

        Raises:
            ValueError: If the experiment is not initialized.

        """
        if self.experiment is None:
            raise ValueError("Experiment not initialized")

        self.experiment.increase_round()
        self.models_aggregated = {}

    def clear(self) -> None:
        """Clear the state."""
        type(self).__init__(self, self.addr)

    def __str__(self) -> str:
        """Return a String representation of the node state."""
        return (
            f"NodeState(addr={self.addr}, status={self.status}, exp_name={self.exp_name}, "
            f"round={self.round}, total_rounds={self.total_rounds}, simulation={self.simulation}, "
            f"models_aggregated={self.models_aggregated}, nei_status={self.nei_status}, "
            f"train_set={self.train_set}, train_set_votes={self.train_set_votes})"
        )
