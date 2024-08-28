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
    
    def set_addr(self, addr: str):
        self.addr = addr

    def set_status(self, status: str):
        self.status = status

    def set_actual_exp_name(self, exp_name: str):
        self.actual_exp_name = exp_name

    def set_total_rounds(self, total_rounds: int):
        self.total_rounds = total_rounds

    def set_simulation(self, simulation: bool):
        self.simulation = simulation

    def set_learner(self, learner: NodeLearner):
        self.learner = learner

    def add_models_aggregated(self, addr: str, models: List[str]):
        self.models_aggregated[addr] = models

    def set_models_aggregated(self, models_aggregated: Dict[str, List[str]]):
        self.models_aggregated = models_aggregated

    def add_nei_status(self, addr: str, value: int):
        self.nei_status[addr] = value

    def set_nei_status(self, nei_status: Dict[str, int]):
        self.nei_status = nei_status

    def set_round(self, round: int):
        self.round = round

    def set_train_set(self, train_set: List[str]):
        self.train_set = train_set

    def set_train_set_votes(self, train_set_votes: Dict[str, Dict[str, int]]):
        self.train_set_votes = train_set_votes

    def add_train_set_votes(self, addr: str, train_set_votes: Dict[str, int]):
        self.train_set_votes[addr] = train_set_votes

    
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
        self.state_addr = ray.put(NodeStateActor.remote(
            addr,
            simulation=simulation
        ))

        self.state = None

    def refresh_state(self) -> None:
        self.state = ray.get(self.state_addr)

    @property 
    def addr(self) -> str:
        """
        Get the address of the node.

        Returns:
            The address of the node.
        """
        return ray.get(self.state.get_addr.remote())
    
    @addr.setter
    def addr(self, addr: str) -> None:
        """
        Set the address of the node.

        Args:
            addr: The address to set.
        """
        return self.state.set_addr.remote(addr)

    @property 
    def status(self) -> str:
        """
        Get the status of the node.

        Returns:
            str: The status of the node.
        """
        return ray.get(self.state.get_status.remote())
    
    @status.setter
    def status(self, status: str) -> None:
        """
        Set the status of the node.

        Args:
            status: The status to set.
        """
        return self.state.set_status.remote(status)

    @property 
    def actual_exp_name(self) -> str:
        """
        Returns the actual experiment name.

        Returns:
            str: The actual experiment name.
        """
        return ray.get(self.state.get_actual_exp_name.remote())
    
    @actual_exp_name.setter
    def actual_exp_name(self, exp_name: str) -> None:
        """
        Set the actual experiment name.

        Args:
            exp_name: The experiment name to set.
        """
        return self.state.set_actual_exp_name.remote(exp_name)

    @property 
    def total_rounds(self) -> int:
        """
        Returns the total number of rounds in the current state.

        Returns:
            int: The total number of rounds.
        """
        return ray.get(self.state.get_total_rounds.remote())
    
    @total_rounds.setter
    def total_rounds(self, total_rounds: int) -> None:      
        """
        Set the total number of rounds in the current state.

        Args:
            total_rounds: The total number of rounds to set.
        """
        return self.state.set_total_rounds.remote(total_rounds)

    @property 
    def simulation(self) -> bool:
        """
        Returns the simulation state of the node.

        Returns:
            bool: The simulation state of the node.
        """
        return ray.get(self.state.get_simulation.remote())
    
    @simulation.setter
    def simulation(self, simulation: bool) -> None:
        """
        Set the simulation state of the node.

        Args:
            simulation: The simulation state to set.
        """
        return self.state.set_simulation.remote(simulation)

    @property 
    def learner(self) -> NodeLearner:
        """
        Returns the learner object associated with the node.

        Returns:
            NodeLearner: The learner object associated with the node.
        """
        return ray.get(self.state.get_learner.remote())
    
    @learner.setter
    def learner(self, learner: NodeLearner) -> None:
        """
        Set the learner object associated with the node.

        Args:
            learner: The learner object to set.
        """
        return self.state.set_learner.remote(learner)

    @property 
    def models_aggregated(self) -> Dict[str, List[str]]:
        """
        Retrieves the aggregated models from the state.

        Returns:
            A dictionary containing the aggregated models, where the keys are strings and the values are lists of strings.
        """
        return ray.get(self.state.get_models_aggregated.remote())
    
    @models_aggregated.setter
    def models_aggregated(self, models_aggregated: Dict[str, List[str]]) -> None:
        """
        Set the aggregated models in the state.

        Args:
            models_aggregated: The aggregated models to set.
        """
        return self.state.set_models_aggregated.remote(models_aggregated)
    
    def add_models_aggregated(self, addr: str, models: List[str]) -> None:
        """
        Add aggregated models to the state.

        Args:
            addr: The address of the node.
            models: The models to add.
        """
        return self.state.add_models_aggregated.remote(addr, models)

    @property 
    def nei_status(self) -> Dict[str, int]:
        """
        Retrieves the neighbor status.

        Returns:
            A dictionary containing the neighbor status, where the keys are the neighbor names and the values are the status codes.
        """
        return ray.get(self.state.get_nei_status.remote())
    
    @nei_status.setter
    def nei_status(self, nei_status: Dict[str, int]) -> None:
        """
        Set the neighbor status.

        Args:
            nei_status: The neighbor status to set.
        """
        return self.state.set_nei_status.remote(nei_status)
    
    def add_nei_status(self, addr: str, value: int) -> None:
        """
        Add a neighbor status.

        Args:
            nei_status: The neighbor status to set.
        """
        return self.state.add_nei_status.remote(addr, value)

    @property
    def round(self) -> int | None:
        """
        Get the current round number.

        Returns:
            int: The current round number.
        """
        return ray.get(self.state.get_round.remote())
    
    @round.setter
    def round(self, round: int) -> None:
        """
        Set the current round number.

        Args:
            round: The round number to set.
        """
        return self.state.set_round.remote(round)
    
    @property 
    def train_set(self) -> List[str]:
        """
        Retrieves the training set from the node's state.

        Returns:
            A list of strings representing the training set.
        """
        return ray.get(self.state.get_train_set.remote())
    
    @train_set.setter
    def train_set(self, train_set: List[str]) -> None:
        """
        Set the training set in the node's state.

        Args:
            train_set: The training set to set.
        """
        return self.state.set_train_set.remote(train_set)

    @property 
    def train_set_votes(self) -> Dict[str, Dict[str, int]]:
        """
        Retrieves the train set votes from the node's state.

        Returns:
            A dictionary containing the train set votes, where the keys are strings representing the train set names,
            and the values are dictionaries mapping the vote names to the corresponding vote counts.
        """
        return ray.get(self.state.get_train_set_votes.remote())
    
    @train_set_votes.setter
    def train_set_votes(self, train_set_votes: Dict[str, Dict[str, int]]) -> None:
        """
        Set the train set votes in the node's state.

        Args:
            train_set_votes: The train set votes to set.
        """
        return self.state.set_train_set_votes.remote(train_set_votes)
    
    def add_train_set_votes(self, addr: str, train_set_votes: Dict[str, int]) -> None:
        """
        Add train set votes to the node's state.

        Args:
            addr: The address of the node.
            train_set_votes: The train set votes to add.
        """
        return self.state.add_train_set_votes.remote(addr, train_set_votes)

    def set_experiment(self, exp_name: str, total_rounds: int) -> None:
        """
        Set the experiment name.

        Args:
            exp_name: The name of the experiment.
            total_rounds: The total rounds of the experiment.

        """
        return self.state.set_experiment.remote(exp_name, total_rounds)

    def increase_round(self) -> None:
        """
        Increase the round number.

        Raises:
            ValueError: If the round is not initialized.

        """
        return self.state.increase_round.remote()

    def clear(self) -> None:
        """Clear the state."""
        return self.state.clear.remote()

    def __str__(self) -> str:
        """String representation of the node state."""
        return ray.get(self.state.__str__.remote())