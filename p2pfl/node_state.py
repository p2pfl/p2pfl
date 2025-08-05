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

from p2pfl.experiment import Experiment
from p2pfl.management.logger import logger


class NodeState:
    """
    Class to store the main state of a learning node.

    Attributes:
        addr: The address of the node.
        status: The status of the node.
        learner: The learner of the node.
        models_aggregated: The models aggregated by the node.
        nei_status: The status of the neighbors.
        train_set: The train set of the node.
        train_set_votes: The votes of the train set.
        train_set_votes_lock: The lock for the train set votes.
        start_thread_lock: The lock for the start thread.
        wait_votes_ready_lock: The lock for the wait votes ready.
        model_initialized_lock: The lock for the model initialized.

    Args:
        addr: The address of the node.

    """

    def __init__(self, addr: str) -> None:
        """Initialize the node state."""
        self.addr = addr
        self.status = "Idle"

        # Aggregator (move to the aggregator?)
        self.models_aggregated_lock = threading.Lock()
        self.models_aggregated: dict[str, list[str]] = {}

        # Other neis state (only round)
        self.nei_status: dict[str, int] = {}

        # Train Set
        self.train_set: list[str] = []
        self.train_set_votes: dict[str, dict[str, int]] = {}

        # Actual experiment
        self.experiment: Experiment | None = None

        # For PreSendModelCommand state
        self.sending_models: dict[str, dict[str, float]] = {}
        self.sending_models_lock = threading.Lock()

        # Locks
        self.train_set_votes_lock = threading.Lock()
        self.start_thread_lock = threading.Lock()
        self.wait_votes_ready_lock = threading.Lock()
        self.model_initialized_lock = threading.Lock()
        self.model_initialized_lock.acquire()
        self.aggregated_model_event = threading.Event()
        self.aggregated_model_event.set()

    @property
    def round(self) -> int | None:
        """Get the round."""
        return self.experiment.round if self.experiment is not None else None

    @property
    def total_rounds(self) -> int | None:
        """Get the total rounds."""
        return self.experiment.total_rounds if self.experiment is not None else None

    @property
    def exp_name(self) -> str | None:
        """Get the actual experiment name."""
        return self.experiment.exp_name if self.experiment is not None else None

    def set_experiment(
        self,
        exp_name: str,
        total_rounds: int,
        dataset_name: str | None = None,
        model_name: str | None = None,
        aggregator_name: str | None = None,
        framework_name: str | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        epochs_per_round: int | None = None,
    ) -> None:
        """
        Start a new experiment.

        Args:
            exp_name: The name of the experiment.
            total_rounds: The total rounds of the experiment.
            dataset_name: The name of the dataset.
            model_name: The name of the model.
            aggregator_name: The name of the aggregator.
            framework_name: The name of the framework.
            learning_rate: The learning rate.
            batch_size: The batch size.
            epochs_per_round: The number of epochs per round.

        """
        self.status = "Learning"
        if self.experiment is None:
            self.experiment = Experiment(
                exp_name,
                total_rounds,
                dataset_name=dataset_name,
                model_name=model_name,
                aggregator_name=aggregator_name,
                framework_name=framework_name,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs_per_round=epochs_per_round,
            )
        logger.experiment_started(self.addr, self.experiment)  # TODO: Improve changes on the experiment

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
        logger.experiment_started(self.addr, self.experiment)  # TODO: Improve changes on the experiment

    def clear(self) -> None:
        """Clear the state."""
        type(self).__init__(self, self.addr)

    def __str__(self) -> str:
        """Return a String representation of the node state."""
        return (
            f"NodeState(addr={self.addr}, status={self.status}, exp_name={self.exp_name}, "
            f"round={self.round}, total_rounds={self.total_rounds}, "
            f"models_aggregated={self.models_aggregated}, nei_status={self.nei_status}, "
            f"train_set={self.train_set}, train_set_votes={self.train_set_votes}, "
            f"sending_models={self.sending_models})"
        )
