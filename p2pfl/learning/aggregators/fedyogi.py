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

"""FedYogi Aggregator - Adaptive Federated Optimization using Yogi."""

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FedYogi(Aggregator):
    """
    FedYogi - Adaptive Federated Optimization using Yogi [Reddi et al., 2020].

    FedYogi adapts the Yogi optimizer to federated settings, maintaining adaptive
    learning rates on the server side to handle heterogeneous data distributions.

    Paper: https://arxiv.org/abs/2003.00295
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = False  # FedYogi needs all updates for proper optimization

    def __init__(
        self,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        disable_partial_aggregation: bool = False,
    ) -> None:
        """
        Initialize the FedYogi aggregator.

        Args:
            eta: Server-side learning rate. Defaults to 1e-2.
            eta_l: Client-side learning rate. Defaults to 0.0316.
            beta_1: Momentum parameter. Defaults to 0.9.
            beta_2: Second moment parameter. Defaults to 0.99.
            tau: Controls the algorithm's degree of adaptability. Defaults to 1e-3.
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

        # Hyperparameters
        self.eta = eta  # server learning rate
        self.eta_l = eta_l  # client learning rate (for client configuration)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau

        # State variables (persist across rounds)
        self.m_t: list[np.ndarray] = []  # momentum (first moment)
        self.v_t: list[np.ndarray] = []  # second moment
        self.current_weights: list[np.ndarray] = []  # current global model weights

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate models using FedYogi algorithm.

        Args:
            models: List of P2PFLModel objects to aggregate.

        Returns:
            A P2PFLModel with the FedYogi-optimized parameters.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there are no models")

        # First, perform FedAvg-style aggregation
        total_samples = sum([m.get_num_samples() for m in models])

        # Create weighted average of models
        first_model_weights = models[0].get_parameters()
        # Ensure fedavg_weights uses float dtype to handle weighted averaging
        fedavg_weights = [np.zeros_like(layer, dtype=np.float64) for layer in first_model_weights]

        # Add weighted models
        for m in models:
            weight = m.get_num_samples() / total_samples
            for i, layer in enumerate(m.get_parameters()):
                # Convert to float64 to ensure proper arithmetic
                fedavg_weights[i] += layer.astype(np.float64) * weight

        # Initialize current_weights on first round
        if not self.current_weights:
            self.current_weights = [np.copy(layer) for layer in fedavg_weights]
            # Return the initial model without optimization on first round
            contributors = []
            for m in models:
                contributors.extend(m.get_contributors())
            return models[0].build_copy(params=self.current_weights, num_samples=total_samples, contributors=contributors)

        # Compute delta_t: difference between aggregated and current weights
        delta_t = [x - y for x, y in zip(fedavg_weights, self.current_weights)]

        # Update momentum (m_t)
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]

        self.m_t = [self.beta_1 * x + (1 - self.beta_1) * y for x, y in zip(self.m_t, delta_t)]

        # Update second moment (v_t) - Yogi's unique update rule
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]

        # Yogi update: v_t = v_t - (1 - β₂) * delta_t² * sign(v_t - delta_t²)
        self.v_t = [x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y)) for x, y in zip(self.v_t, delta_t)]

        # Update weights: w_t = w_t + η * m_t / (√v_t + τ)
        self.current_weights = [
            x + self.eta * y / (np.sqrt(np.abs(z)) + self.tau)  # abs to handle potential negative values
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        # Get contributors
        contributors = []
        for m in models:
            contributors.extend(m.get_contributors())
        # Remove duplicates while preserving order
        seen = set()
        unique_contributors = []
        for x in contributors:
            if x not in seen:
                seen.add(x)
                unique_contributors.append(x)
        contributors = unique_contributors

        # Return the optimized model
        return models[0].build_copy(
            params=self.current_weights,
            num_samples=total_samples,
            contributors=contributors,
            additional_info={"fedyogi": {"eta_l": self.eta_l}},  # Pass client learning rate
        )
