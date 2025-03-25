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

"""Context Information Updating command."""

from typing import Optional

import numpy as np

from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class ContextInformationUpdatingCommand(Command):
    """ContextInformationUpdatingCommand."""

    def __init__(self, state: NodeState) -> None:
        """Initialize the command."""
        super().__init__()
        self.__state = state

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "context information updating"

    def execute(self,
                source: str,
                round: int,
                loss: Optional[list[np.ndarray]],
                weights: Optional[list[np.ndarray]],
                index: Optional[int],
                push_sum_weight: Optional[float],
                **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            loss: The loss of the command.
            model: The model of the command.
            index: The index of the command.
            **kwargs: The command keyword arguments.

        """
        # Save loss
        if loss is not None:
            self.__state.loss[source] = (loss, round)

        # Save index of local iteration about model updating
        if index is not None:
            self.__state.reception_times[source] = index

        # Save model
        if weights is not None:
            model = self.learner.get_model().build_copy(params=weights)
            #self._node.state.models_aggregated[source] = model
            self.aggregator.add_model(model)

        # Save push-sum weight
        if push_sum_weight is not None:
            self.__state.push_sum_weights[source] = push_sum_weight
