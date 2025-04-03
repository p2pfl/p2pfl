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

"""Context Information Updating commands."""

from typing import Optional
from p2pfl.communication.commands.command import Command
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.node import Node


class LossInformationUpdatingCommand(Command):
    """LossInformationUpdatingCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        super().__init__()
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "loss_information_updating"

    def execute(self, source: str, round: int, loss: str, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            loss: Training loss of the source.
            **kwargs: The command keyword arguments.

        """
        # Save loss
        if loss is not None:
            logger.info(self.__node.state.addr, "📉 Neighbour training loss received.")
            self.__node.state.losses[source] = (float(loss), round)


class IndexInformationUpdatingCommand(Command):
    """IndexInformationUpdatingCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        super().__init__()
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "index_information_updating"

    def execute(self, source: str, round: int, index: str, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            index: Index of local iteration of the source.
            **kwargs: The command keyword arguments.

        """
        # Save index of local iteration about model updating
        if index is not None:
            logger.info(self.__node.state.addr, "🔄 Index of local iteration received.")
            self.__node.state.reception_times[source] = int(index)


class ModelInformationUpdatingCommand(Command):
    """ModelInformationUpdatingCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        super().__init__()
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "model_information_updating"

    def execute(self, source: str, round: int,
                weights: Optional[bytes] = None,
                contributors: Optional[list[str]] = None,  # TIPO ESTA MAL (NECESARIO CASTEARLO AL LLAMAR)
                num_samples: Optional[int] = None,
                **kwargs,
        ) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            *args: Vote values (pairs of key and values).
            **kwargs: The command keyword arguments.

        """
        if weights is None or contributors is None or num_samples is None:
            raise ValueError("Weights, contributors and weight are required")

        logger.info(self.__node.state.addr, "Model received.")

        # Check if Learning is running
        if self.__node.state.round is not None:
            try:
                # Save model
                model = self.__node.learner.get_model().build_copy(params=weights, num_samples=num_samples, contributors=list(contributors))
                #self._node.state.models_aggregated[source] = model
                self.__node.aggregator.add_model(model)
            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.__node.state.addr, "Error decoding parameters.")
                self.__node.stop()

            except ModelNotMatchingError:
                logger.error(self.__node.state.addr, "Models not matching.")
                self.__node.stop()

            except Exception as e:
                logger.error(self.__node.state.addr, f"Unknown error adding model: {e}")
                self.__node.stop()

        else:
            logger.debug(self.__node.state.addr, "Tried to add a model while learning is not running")


class PushSumWeightInformationUpdatingCommand(Command):
    """PushSumWeightInformationUpdatingCommand."""

    def __init__(self, node: Node) -> None:
        """Initialize the command."""
        super().__init__()
        self.__node = node

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "push_sum_weight_information_updating"

    def execute(self, source: str, round: int, push_sum_weight: str, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command.
            push_sum_weight: Push-sum weight of the source.
            **kwargs: The command keyword arguments.

        """
        # Save push-sum weight
        if push_sum_weight is not None:
            logger.info(self.__node.state.addr, "Push-sum weight received.")
            self.__node.state.push_sum_weights[source] = float(push_sum_weight)
