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

"""Krum Aggregator."""

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class Krum(Aggregator):
    """
    Krum [Blanchard et al., 2017].

    Paper: https://arxiv.org/pdf/1703.02757
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = False  # Krum doesn't support partial aggregation

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate using Krum algorithm.

        Krum selects the model with the minimum sum of distances to all other models.

        Args:
            models: List of P2PFLModel objects to aggregate.

        Returns:
            A P2PFLModel with the selected model (lowest distance sum).

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there are no models")

        # If only one model, return it
        if len(models) == 1:
            return models[0]

        # Extract parameters from models
        model_params = [model.get_parameters() for model in models]
        total_models = len(models)

        # Calculate pairwise distances and sum for each model
        distance_sums = []
        for i in range(total_models):
            distance_sum = 0.0
            for j in range(total_models):
                if i != j:
                    # Calculate L2 distance between models i and j
                    distance = 0.0
                    for layer_i, layer_j in zip(model_params[i], model_params[j], strict=False):
                        # Compute L2 norm of the difference
                        distance += float(np.linalg.norm(layer_i - layer_j))
                    distance_sum += distance
            distance_sums.append(distance_sum)

        # Find the model with minimum distance sum
        min_index = int(np.argmin(distance_sums))
        selected_model = models[min_index]

        # Aggregate contributors from all models
        contributors = []
        for model in models:
            contributors.extend(model.get_contributors())
        # Remove duplicates while preserving order
        seen = set()
        unique_contributors = []
        for x in contributors:
            if x not in seen:
                seen.add(x)
                unique_contributors.append(x)
        contributors = unique_contributors

        # Total samples (sum from all models)
        total_samples = sum([m.get_num_samples() for m in models])

        # Return the selected model with updated metadata
        return selected_model.build_copy(params=selected_model.get_parameters(), num_samples=total_samples, contributors=contributors)
