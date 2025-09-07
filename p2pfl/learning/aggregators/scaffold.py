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

from typing import Any

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class Scaffold(Aggregator):
    """
    SCAFFOLD Aggregator.

    Paper: https://arxiv.org/pdf/1910.06378
    The aggregator acts like the server in centralized learning, handling both model and control variate updates.

    Due to the complete decentralization of the enviroment, a global model is also maintained in the aggregator.
    This consumes additional bandwidth.

    ::todo:: Improve efficiency by sharing the global model only each n rounds.
    """

    REQUIRED_INFO_KEYS = ["delta_y_i", "delta_c_i"]
    SUPPORTS_PARTIAL_AGGREGATION: bool = False

    def __init__(self, global_lr: float = 0.1, disable_partial_aggregation: bool = False):
        """
        Initialize the aggregator.

        Args:
            global_lr: The global learning rate.
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.global_lr = global_lr
        self.c: list[np.ndarray] = []  # global control variates
        self.global_model_params: list[np.ndarray] = []  # simulate global model

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models and control variates from clients.

        Args:
            models: List of models to aggregate.

        """
        if not models:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        total_samples = sum([m.get_num_samples() for m in models])
        # initialize the accumulators for the model and the control variates
        first_model_weights = models[0].get_parameters()
        accum_delta_y = [np.zeros_like(layer) for layer in first_model_weights]

        # Accumulate weighted model updates
        for m in models:
            delta_y_i = self._get_and_validate_model_info(m)["delta_y_i"]
            num_samples = m.get_num_samples()
            for i, layer in enumerate(delta_y_i):
                accum_delta_y[i] += layer * num_samples

        # Normalize the accumulated model updates and apply global learning rate
        accum_delta_y = [(layer / total_samples) * self.global_lr for layer in accum_delta_y]

        # Update global model
        if not self.global_model_params:
            self.global_model_params = models[0].get_parameters()
        self.global_model_params = [param + delta for param, delta in zip(self.global_model_params, accum_delta_y, strict=False)]

        # Accumulate control variates
        delta_c_i_first = self._get_and_validate_model_info(models[0])["delta_c_i"]  # take first model as reference
        accum_c = [np.zeros_like(layer) for layer in delta_c_i_first]

        if delta_c_i_first is None:
            raise ValueError("delta_c_i cannot be None after validation")

        for m in models:
            delta_c_i = self._get_and_validate_model_info(m)["delta_c_i"]
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

        # Return the aggregated model with the global model parameters and the control variates
        aggregated_model = models[0].build_copy(params=self.global_model_params, num_samples=total_samples, contributors=contributors)
        aggregated_model.add_info("scaffold", {"global_c": self.c})

        return aggregated_model

    def get_required_callbacks(self) -> list[str]:
        """Retrieve the list of required callback keys for this aggregator."""
        return ["scaffold"]

    def _get_and_validate_model_info(self, model: P2PFLModel) -> dict[str, Any]:
        """
        Validate the model.

        Args:
            model: The model to validate.

        """
        info = model.get_info("scaffold")
        if not all(key in info for key in self.REQUIRED_INFO_KEYS):
            raise ValueError(f"Model is missing required info keys: {self.REQUIRED_INFO_KEYS}Model info keys: {info.keys()}")
        return info
