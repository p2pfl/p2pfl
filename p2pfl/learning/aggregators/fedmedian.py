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

"""Federated Median (FedMedian) Aggregator."""

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FedMedian(Aggregator):
    """
    Federated Median (FedMedian) [Yin et al., 2018].

    Paper: https://arxiv.org/pdf/1803.01498v1.pdf
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = False

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Compute the median of the models.

        Args:
            models: Dict with the models (node: model, num_samples).

        Returns:
            A P2PFLModel with the aggregated

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there are no models")

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Create a list to store the weights of each model
        weights = [m.get_parameters() for m in models]

        # Calculate the median for each layer
        median_weights = []
        for layer_index in range(len(weights[0])):
            layer_weights = [weights[model][layer_index] for model in range(len(weights))]
            median_weights.append(np.median(np.array(layer_weights), axis=0))

        # Return an aggregated p2pfl model
        return models[0].build_copy(params=median_weights, num_samples=total_samples)
