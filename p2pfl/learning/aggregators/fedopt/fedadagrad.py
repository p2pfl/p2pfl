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

"""FedAdagrad Aggregator - Adaptive Federated Optimization using Adagrad."""

import numpy as np

from p2pfl.learning.aggregators.fedopt.base import FedOptBase


class FedAdagrad(FedOptBase):
    """
    FedAdagrad - Adaptive Federated Optimization using Adagrad [Reddi et al., 2020].

    FedAdagrad adapts the Adagrad optimizer to federated settings, maintaining adaptive
    learning rates on the server side based on accumulated squared gradients.

    Paper: https://arxiv.org/abs/2003.00295
    """

    def __init__(self, eta: float = 1e-1, beta_1: float = 0.9, tau: float = 1e-9, disable_partial_aggregation: bool = False) -> None:
        """
        Initialize the FedAdagrad aggregator.

        Args:
            eta: Server-side learning rate. Defaults to 1e-1.
            beta_1: Momentum parameter. Defaults to 0.9.
            tau: Controls the algorithm's degree of adaptability. Defaults to 1e-9.
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(eta=eta, beta_1=beta_1, tau=tau, disable_partial_aggregation=disable_partial_aggregation)

        # State variables specific to Adagrad
        self.v_t: list[np.ndarray] = []  # accumulated squared gradients

    def _optimizer_update(self, delta_t: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply Adagrad optimizer update to the current weights.

        Args:
            delta_t: The difference between fedavg weights and current weights.

        Returns:
            Updated weights after applying Adagrad.

        """
        # Compute momentum using base class method
        self._compute_momentum(delta_t)

        # Second moment update (Adagrad): v_t = v_{t-1} + Δt²
        # Accumulates squared pseudo-gradients without decay
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]

        self.v_t = [x + np.multiply(y, y) for x, y in zip(self.v_t, delta_t, strict=False)]

        # Weight update: x^{t+1} = x^t + η * m_t / √(v_t + τ)
        # This variant uses momentum instead of Δt directly
        updated_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau) for x, y, z in zip(self.current_weights, self.m_t, self.v_t, strict=False)
        ]

        return updated_weights
