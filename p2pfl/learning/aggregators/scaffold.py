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

"""Callback for SCAFFOLD operations."""

from typing import List

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.p2pfl_model import P2PFLModel


class ScaffoldAggregator(Aggregator):
    """
    SCAFFOLD Aggregator.

    Paper: https://arxiv.org/pdf/1910.06378
    The aggregator acts like the server in centralized learning, handling both model and control variate updates.
    """

    REQUIRED_INFO_KEYS = ['delta_y_i', 'delta_c_i']

    def __init__(self, node_name:str, **kwargs):
        """
        Initialize the aggregator.

        Args:
            node_name: The name of the node.
            kwargs: Additional arguments.

        """
        super().__init__(node_name)
        self.c: List[np.ndarray] = [] # global control variates
        self.global_lr = kwargs.get('global_lr', 0.1)

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models and control variates from clients.

        Args:
            models: List of models to aggregate.

        """
        if not models:
            raise NoModelsToAggregateError(f"({self.node_name}) Trying to aggregate models when there is no models")

        total_samples = sum([m.get_num_samples() for m in models])
        # initialize the accumulators for the model and the control variates
        first_model_weights = models[0].get_parameters()
        accum_y = [np.zeros_like(layer) for layer in first_model_weights]

        # Accumulate weighted model updates
        for m in models:
            self._validate_model_info(m)
            delta_y_i = m.get_info('delta_y_i')
            if delta_y_i is None:
                raise ValueError(f"Model is missing required info keys: {self.REQUIRED_INFO_KEYS}") # mypi check
            num_samples = m.get_num_samples()
            for i, layer in enumerate(delta_y_i):
                accum_y[i] += layer * num_samples

        # Normalize the accumulated model updates
        accum_y = [layer / total_samples for layer in accum_y]
        accum_y = [layer * self.global_lr for layer in accum_y] # apply global learning rate

        # Accumulate control variates
        delta_c_i_first = models[0].get_info('delta_c_i')
        if delta_c_i_first is None:
            raise ValueError("delta_c_i cannot be None after validation")
        accum_c = [np.zeros_like(layer) for layer in delta_c_i_first]
        for m in models:
            delta_c_i = m.get_info('delta_c_i')
            if delta_c_i is None:
                raise ValueError("delta_c_i cannot be None after validation")
            for i in range(len(accum_c)):
                accum_c[i] += delta_c_i[i]

        # Normalize the accumulated control variates
        num_models = len(models)
        accum_c = [layer / num_models for layer in accum_c]

        # Update global c
        if not self.c:
            self.c = [np.zeros_like(layer) for layer in accum_c]

        for i in range(len(self.c)):
            self.c[i] += accum_c[i]

        # Get contributors
        contributors = []
        for m in models:
            contributors.extend(m.get_contributors())

        aggregated_model = models[0].build_copy(
            params=accum_y,
            num_samples=total_samples,
            contributors=contributors
        )
        aggregated_model.add_info('global_c', self.c)
        return aggregated_model

    def get_required_callbacks(self) -> List[str]:
        """Retrieve the list of required callback keys for this aggregator."""
        return ["scaffold"]

    def supports_partial_aggr(self) -> bool:
        """Check if the aggregator supports partial aggregations."""
        return False

    def _validate_model_info(self, model: P2PFLModel) -> None:
        """
        Validate the model.

        Args:
            model: The model to validate.

        """
        if not all(key in model.additional_info for key in self.REQUIRED_INFO_KEYS):
            raise ValueError(f"Model is missing required info keys: {self.REQUIRED_INFO_KEYS}"
                             f"Model info keys: {model.additional_info.keys()}")

