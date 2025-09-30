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

"""FedAdam Aggregator - Adaptive Federated Optimization using Adam."""

import numpy as np

from p2pfl.learning.aggregators.fedopt.base import FedOptBase


class FedAdam(FedOptBase):
    """
    FedAdam - Adaptive Federated Optimization using Adam [Reddi et al., 2020].

    FedAdam adapts the Adam optimizer to federated settings, maintaining both
    momentum and adaptive learning rates on the server side.

    Paper: https://arxiv.org/abs/2003.00295
    """

    def __init__(
        self,
        eta: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
        disable_partial_aggregation: bool = False,
    ) -> None:
        """
        Initialize the FedAdam aggregator.

        Args:
            eta: Server-side learning rate. Defaults to 1e-1.
            beta_1: Momentum parameter. Defaults to 0.9.
            beta_2: Second moment parameter. Defaults to 0.99.
            tau: Small constant for numerical stability. Defaults to 1e-9.
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(eta=eta, beta_1=beta_1, tau=tau, disable_partial_aggregation=disable_partial_aggregation)

        # Hyperparameters specific to Adam
        self.beta_2 = beta_2

        # State variables specific to Adam
        self.v_t: list[np.ndarray] = []  # second moment

    def _optimizer_update(self, delta_t: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply Adam optimizer update to the current weights.

        Args:
            delta_t: The difference between fedavg weights and current weights.

        Returns:
            Updated weights after applying Adam.

        """
        # Compute momentum using base class method
        self._compute_momentum(delta_t)

        # Second moment update (Adam): v_t = β2 * v_{t-1} + (1 - β2) * Δt²
        # Exponential moving average of squared pseudo-gradients
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]

        self.v_t = [self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y) for x, y in zip(self.v_t, delta_t, strict=False)]

        # Note: The bias correction computation requires server_round to be passed
        # For now, we'll track rounds internally
        if not hasattr(self, "_round_counter"):
            self._round_counter = 0
        self._round_counter += 1

        # Bias correction for Adam - improves convergence in early rounds
        # From Kingma & Ba, 2014 "Adam: A Method for Stochastic Optimization"
        # This is α_t in the formula line right before Section 2.1
        # η_corrected = η * √(1 - β2^t) / (1 - β1^t)
        eta_norm = self.eta * np.sqrt(1 - np.power(self.beta_2, self._round_counter)) / (1 - np.power(self.beta_1, self._round_counter))

        # Weight update: x^{t+1} = x^t + η_corrected * m_t / √(v_t + τ)
        updated_weights = [
            x + eta_norm * y / (np.sqrt(z) + self.tau) for x, y, z in zip(self.current_weights, self.m_t, self.v_t, strict=False)
        ]

        return updated_weights
