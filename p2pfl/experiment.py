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


class Experiment:
    """
    Class to represent an experiment.

    Attributes:
        exp_name(str): The name of the experiment.
        total_rounds(int): The total rounds of the experiment.
        round(int): The current round.

    Args:
        exp_name: The name of the experiment.
        total_rounds: The total rounds of the experiment.

    """

    def __init__(self, exp_name: str, total_rounds: int):
        """Initialize the experiment."""
        self.exp_name = exp_name
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

    def __str__(self):
        """Return the string representation of the experiment."""
        return f"Experiment(exp_name={self.exp_name}, total_rounds={self.total_rounds}, " f"round={self.round})"
