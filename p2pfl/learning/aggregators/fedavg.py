#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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

"""Federated Averaging (FedAvg) Aggregator."""

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016].

    Paper: https://arxiv.org/abs/1602.05629.
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Create a Zero Model using numpy
        first_model_weights = models[0].get_parameters()
        accum = [np.zeros_like(layer) for layer in first_model_weights]

        # Add weighted models
        for m in models:
            for i, layer in enumerate(m.get_parameters()):
                accum[i] = np.add(accum[i], layer * m.get_num_samples())

        # Normalize Accum
        accum = [np.divide(layer, total_samples) for layer in accum]

        # Get contributors
        contributors: list[str] = []
        for m in models:
            contributors = contributors + m.get_contributors()

        # Return an aggregated p2pfl model
        return models[0].build_copy(params=accum, num_samples=total_samples, contributors=contributors)
