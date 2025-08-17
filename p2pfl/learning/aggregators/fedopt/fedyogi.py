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

from p2pfl.learning.aggregators.fedopt.base import FedOptBase


class FedYogi(FedOptBase):
    """
    FedYogi - Adaptive Federated Optimization using Yogi [Reddi et al., 2020].

    FedYogi adapts the Yogi optimizer to federated settings, maintaining adaptive
    learning rates on the server side to handle heterogeneous data distributions.

    Paper: https://arxiv.org/abs/2003.00295
    """

    def __init__(
        self,
        eta: float = 1e-2,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        disable_partial_aggregation: bool = False,
    ) -> None:
        """
        Initialize the FedYogi aggregator.

        Args:
            eta: Server-side learning rate. Defaults to 1e-2.
            beta_1: Momentum parameter. Defaults to 0.9.
            beta_2: Second moment parameter. Defaults to 0.99.
            tau: Controls the algorithm's degree of adaptability. Defaults to 1e-3.
            disable_partial_aggregation: Whether to disable partial aggregation.

        """
        super().__init__(eta=eta, beta_1=beta_1, tau=tau, disable_partial_aggregation=disable_partial_aggregation)

        # Hyperparameters specific to Yogi
        self.beta_2 = beta_2

        # State variables specific to Yogi
        self.v_t: list[np.ndarray] = []  # second moment

    def _optimizer_update(self, delta_t: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply Yogi optimizer update to the current weights.

        Args:
            delta_t: The difference between fedavg weights and current weights.

        Returns:
            Updated weights after applying Yogi.

        """
        # Compute momentum using base class method
        self._compute_momentum(delta_t)

        # Second moment update (Yogi): v_t = v_{t-1} - (1 - β2) * Δt² * sign(v_{t-1} - Δt²)
        # Adaptive update that increases v_t when v_t < Δt² and decreases when v_t > Δt²
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]

        self.v_t = [
            x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y)) for x, y in zip(self.v_t, delta_t, strict=False)
        ]

        # Weight update: x^{t+1} = x^t + η * m_t / √(v_t + τ)
        updated_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau) for x, y, z in zip(self.current_weights, self.m_t, self.v_t, strict=False)
        ]

        return updated_weights
