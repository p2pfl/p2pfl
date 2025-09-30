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

"""Base class for FedOpt family of aggregators."""

from abc import abstractmethod
from typing import Any

import numpy as np

from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FedOptBase(FedAvg):
    """
    Base class for Federated Optimization (FedOpt) family [Reddi et al., 2020].

    This class extends FedAvg to provide common functionality for adaptive
    federated optimization algorithms like FedAdagrad, FedAdam, and FedYogi.

    Paper: https://arxiv.org/abs/2003.00295
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = False  # FedOpt algorithms need all updates for proper optimization

    def __init__(self, eta: float = 1e-1, beta_1: float = 0.9, tau: float = 1e-9, disable_partial_aggregation: bool = False) -> None:
        """
        Initialize the FedOpt base aggregator.

        Args:
            eta: Server-side learning rate.
            beta_1: Momentum parameter (used by algorithms with momentum).
            tau: Small constant for numerical stability.
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

        # Hyperparameters
        self.eta = eta  # η: server learning rate for adaptive optimization
        self.beta_1 = beta_1  # β1: momentum coefficient
        self.tau = tau  # τ: small constant for numerical stability in division

        # State variables (persist across rounds)
        self.current_weights: list[np.ndarray] = []  # x^t: current global model parameters
        self.m_t: list[np.ndarray] = []  # momentum (first moment)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate models using FedOpt algorithm.

        Args:
            models: List of P2PFLModel objects to aggregate.

        Returns:
            A P2PFLModel with the optimized parameters.

        """
        # Compute weighted average of client updates
        # Δt = (1/|S|) Σ Δt_i where Δt_i = client_model - global_model
        fedavg_model = super().aggregate(models)
        fedavg_weights = fedavg_model.get_parameters()

        # Initialize global model on first round
        if not self.current_weights:
            self.current_weights = [np.copy(layer) for layer in fedavg_weights]
            return fedavg_model

        # Compute pseudo-gradient: difference between averaged models and current global model
        # Δt = fedavg_weights - current_weights
        delta_t = [x - y for x, y in zip(fedavg_weights, self.current_weights, strict=False)]

        # Apply adaptive optimization with momentum and adaptive learning rates
        # x^{t+1} = x^t + η * m_t / √(v_t + τ)
        self.current_weights = self._optimizer_update(delta_t)

        # Return the optimized model with additional info
        return models[0].build_copy(
            params=self.current_weights,
            num_samples=fedavg_model.get_num_samples(),
            contributors=fedavg_model.get_contributors(),
            additional_info=self._get_additional_info(),
        )

    def _compute_momentum(self, delta_t: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute momentum update (shared across all FedOpt algorithms).

        Formula: m_t = β1 * m_{t-1} + (1 - β1) * Δt

        Args:
            delta_t: The average client updates Δt.

        Returns:
            Updated momentum values.

        """
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]

        self.m_t = [np.multiply(self.beta_1, x) + (1 - self.beta_1) * y for x, y in zip(self.m_t, delta_t, strict=False)]

        return self.m_t

    @abstractmethod
    def _optimizer_update(self, delta_t: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply optimizer-specific update to the current weights.

        This method implements:
        - Second moment update: v_t (algorithm-specific)
        - Weight update: x^{t+1} = x^t + η*m_t/√(v_t+τ)

        Args:
            delta_t: The average client updates Δt.

        Returns:
            Updated weights after applying the optimizer.

        """
        raise NotImplementedError("Subclasses must implement _optimizer_update")

    def _get_additional_info(self) -> dict[str, Any]:
        """
        Get additional info to pass to the model.

        Returns:
            Dictionary with additional info.

        """
        # Subclasses can override this to add optimizer-specific info
        return {}
