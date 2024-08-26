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

import ray
from typing import Dict, List, Optional

from p2pfl.learning.learner import NodeLearner

@ray.remote
class NodeStateActor:
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

    def __init__(self, addr: str, simulation: bool=False) -> None:
        """Initialize the node state."""
        self.addr = addr
        self.status = "Idle"
        self.actual_exp_name: Optional[str] = None
        self.round: Optional[int] = None
        self.total_rounds: Optional[int] = None

        # Simulation
        self.simulation = simulation

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
        #self.train_set_votes_lock = threading.Lock()
        #self.start_thread_lock = threading.Lock()
        #self.wait_votes_ready_lock = threading.Lock()
        #self.model_initialized_lock = threading.Lock()
        #self.model_initialized_lock.acquire()

    def get_addr(self):
        return self.addr
    
    def get_status(self):
        return self.status
    
    def get_actual_exp_name(self):
        return self.actual_exp_name
    
    def get_total_rounds(self):
        return self.total_rounds
    
    def get_simulation(self):
        return self.simulation
    
    def get_learner(self):
        return self.learner
    
    def get_models_aggregated(self):
        return self.models_aggregated
    
    def get_nei_status(self):
        return self.nei_status
    
    def get_round(self):
        return self.round
    
    def get_train_set(self):
        return self.train_set
    
    def get_train_set_votes(self):
        return self.train_set_votes
    
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
        self.status = "Idle"
        self.actual_exp_name = None
        self.round = None
        self.total_rounds = None

    def __str__(self) -> str:
        """String representation of the node state."""
        return f"NodeState(addr={self.addr}, status={self.status}, actual_exp_name={self.actual_exp_name}, round={self.round}, total_rounds={self.total_rounds}, simulation={self.simulation}, learner={self.learner}, models_aggregated={self.models_aggregated}, nei_status={self.nei_status}, train_set={self.train_set}, train_set_votes={self.train_set_votes})"

class NodeState():
    def __init__(self, addr: str, simulation: bool=False) -> None:
        self.state = ray.put(NodeStateActor.remote(
            addr,
            simulation=simulation
        ))

    @property 
    def addr(self):
        state = ray.get(self.state)
        return state.get_addr.remote()

    @property 
    def status(self):
        state = ray.get(self.state)
        return state.get_status.remote()

    @property 
    def actual_exp_name(self):
        state = ray.get(self.state)
        return state.get_actual_exp_name.remote()

    @property 
    def total_rounds(self):
        state = ray.get(self.state)
        return state.get_total_rounds.remote()

    @property 
    def simulation(self):
        state = ray.get(self.state)
        return state.get_simulation.remote()

    @property 
    def learner(self):
        state = ray.get(self.state)
        return state.get_learner.remote()

    @property 
    def models_aggregated(self):
        state = ray.get(self.state)
        return state.get_models_aggregated.remote()

    @property 
    def nei_status(self):
        state = ray.get(self.state)
        return state.get_nei_status.remote()

    @property
    def round(self):
        state = ray.get(self.state)
        print(ray.get(state.get_round.remote()))
        return state.get_round.remote()
    
    @property 
    def train_set(self):
        state = ray.get(self.state)
        return state.get_train_set.remote()

    @property 
    def train_set_votes(self):
        state = ray.get(self.state)
        return state.get_train_set_votes.remote()

    def set_experiment(self, exp_name: str, total_rounds: int) -> None:
        """
        Set the experiment name.

        Args:
            exp_name: The name of the experiment.
            total_rounds: The total rounds of the experiment.

        """
        state = ray.get(self.state)
        return state.increase_round.remote(exp_name, total_rounds)

    def increase_round(self) -> None:
        """
        Increase the round number.

        Raises:
            ValueError: If the round is not initialized.

        """
        state = ray.get(self.state)
        return state.increase_round.remote()

    def clear(self) -> None:
        """Clear the state."""
        state = ray.get(self.state)
        return state.clear.remote()

    def __str__(self) -> str:
        """String representation of the node state."""
        state = ray.get(self.state)
        return state.__str__.remote()