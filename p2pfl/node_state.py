import threading
from typing import Dict, List


class NodeState:
    """
    Class to store the main state of a learning node.
    """

    def __init__(self, addr):
        self.addr = addr
        self.status = "Idle"
        self.actual_exp_name = None
        self.round = None
        self.total_rounds = None

        """
        A piñón temporalmente :/
        """

        # Simulation
        self.simulation = False

        # Learning
        self.learner = None

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
        """
        Set the experiment name.
        """
        self.status = "Learning"
        self.actual_exp_name = exp_name
        self.total_rounds = total_rounds
        self.round = 0

    def increase_round(self) -> None:
        """
        Increase the round number.
        """
        self.round += 1
        self.models_aggregated = {}

    def clear(self) -> None:
        """
        Clear the state.
        """
        self.status = "Idle"
        self.actual_exp_name = None
        self.round = None
        self.total_rounds = None
