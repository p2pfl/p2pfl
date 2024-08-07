#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

"""FedAvg Aggregator."""

from typing import Dict, Tuple

import torch

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError


class FedAvg(Aggregator):
    """Federated Averaging (FedAvg) [McMahan et al., 2016] | Paper: https://arxiv.org/abs/1602.05629."""

    def aggregate(self, models: Dict[str, Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.node_name}) Trying to aggregate models when there is no models")

        models_list = list(models.values())

        # Total Samples
        total_samples = sum([y for _, y in models_list])

        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models_list[-1][0].items()}

        # Add weighted models
        for m, w in models_list:
            for layer in m:
                accum[layer] = accum[layer] + m[layer] * w

        # Normalize Accum
        for layer in accum:
            accum[layer] = accum[layer] / total_samples

        return accum
