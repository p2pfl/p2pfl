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

from typing import Dict, Tuple

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.model_parameters_dto import LearnerStateDTO


class FedMedian(Aggregator):
    """
    Federated Median (FedMedian) [Yin et al., 2018].

    Paper: https://arxiv.org/pdf/1803.01498v1.pdf
    """

    def aggregate(self, models: Dict[str, Tuple[LearnerStateDTO, int]]) -> LearnerStateDTO:
        """
        Compute the median of the models.

        Args:
            models: Dict with the models (node: model, num_samples).

        Returns:
            LearnerStateDTO: The aggregated model state with median weights.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.node_name}) Trying to aggregate models when there are no models")

        models_list = list(models.values())  # list of tuples (model, num_samples)
        model_weights_list = [model.get_weights() for model, _ in models_list]

        first_model_weights = models_list[0][0].get_weights()

        accum = {layer: np.zeros_like(param) for layer, param in first_model_weights.items()}

        # compute the median
        for layer in accum:
            layer_weights = [model[layer] for model in model_weights_list]
            np.median(
                layer_weights,
                axis=0,
                out=accum[layer],
            )

        # Create a LearnerStateDTO to return
        aggregated_state = LearnerStateDTO()
        aggregated_state.add_weights_dict(accum)
        return aggregated_state
