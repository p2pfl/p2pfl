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
"""Experiment class."""

from dataclasses import dataclass


@dataclass
class Experiment:
    """
    Class to represent an experiment.

    Attributes:
        exp_name: The name of the experiment.
        total_rounds: The total rounds of the experiment.
        round: The current round.
        dataset_name: The name of the dataset.
        model_name: The name of the model.
        aggregator_name: The name of the aggregator.
        framework_name: The name of the framework.
        learning_rate: The learning rate.
        batch_size: The batch size.
        epochs_per_round: The number of epochs per round.

    """

    exp_name: str
    total_rounds: int
    round: int = 0
    dataset_name: str | None = None
    model_name: str | None = None
    aggregator_name: str | None = None
    framework_name: str | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    epochs_per_round: int | None = None

    def increase_round(self) -> None:
        """
        Increase the round number.

        Raises:
            ValueError: If the round is not initialized.

        """
        if self.round is None:
            raise ValueError("Round not initialized")
        self.round += 1

    def self(self, param_name, param_val=None):
        """
        Getter and setter for the experiment parameters.

        Args:
            param_name: The parameter name.
            param_val: The parameter value.

        Returns:
            The parameter value if param_val is None, or the parameter value if param_val is not None.

        """
        if param_val is None:
            return getattr(self, param_name)
        else:
            setattr(self, param_name, param_val)

    def to_dict(self, exclude_none: bool = True) -> dict:
        """
        Convert the experiment to a dictionary.

        Args:
            exclude_none: If True, exclude fields with None values.

        Returns:
            Dictionary representation of the experiment.

        """
        config = {
            "exp_name": self.exp_name,
            "total_rounds": self.total_rounds,
            "round": self.round,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "aggregator_name": self.aggregator_name,
            "framework_name": self.framework_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs_per_round": self.epochs_per_round,
        }

        if exclude_none:
            return {k: v for k, v in config.items() if v is not None}
        return config

    def __str__(self):
        """Return the string representation of the experiment."""
        metadata_str = ""
        if self.dataset_name:
            metadata_str += f", dataset_name={self.dataset_name}"
        if self.model_name:
            metadata_str += f", model_name={self.model_name}"
        if self.aggregator_name:
            metadata_str += f", aggregator_name={self.aggregator_name}"
        if self.framework_name:
            metadata_str += f", framework_name={self.framework_name}"
        if self.learning_rate:
            metadata_str += f", learning_rate={self.learning_rate}"
        if self.batch_size:
            metadata_str += f", batch_size={self.batch_size}"
        if self.epochs_per_round:
            metadata_str += f", epochs_per_round={self.epochs_per_round}"

        return f"Experiment(exp_name={self.exp_name}, total_rounds={self.total_rounds}, round={self.round}{metadata_str})"
