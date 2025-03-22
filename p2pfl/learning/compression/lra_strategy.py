#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""LRA compression strategy."""

import numpy as np

from p2pfl.learning.compression.base_compression_strategy import TensorCompressor


class LowRankApproximation(TensorCompressor):
    """Low Rank optimization strategy."""

    def apply_strategy(self, params: list[np.ndarray], threshold: float = 0.95) -> tuple[list[np.ndarray], dict]:
        """
        Approximate the parameters preserving a target rank or energy threshold.

        Args:
            params: The parameters to compress.
            threshold: Percentage between 0 and 1 of the energy to preserve.

        """
        params_to_share = []  # layers without low rank approximation
        compressed_states = {}  # compressed layers

        for pos, layer in enumerate(params):
            # if threshold is provided, compute target rank

            if layer.ndim != 2:
                params_to_share.append(layer)  # list(np.ndarray)
            else:
                u, s, vt = np.linalg.svd(layer, full_matrices=False)
                # compute number of values to keep so cumulative energy sum is above threshold
                energy_total = np.sum(s**2)
                cumulative_energies = np.cumsum(s**2)
                target_rank = np.searchsorted(cumulative_energies / energy_total, threshold) + 1
                u, s, vt = u[:, :target_rank], s[:target_rank], vt[:target_rank, :]
                # remove layer from data and store compressed layer
                compressed_states[pos] = u, s, vt

        return params_to_share, {"lowrank_compressed_state": compressed_states}

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Restore the payload by computing dot product of LRA components.

        Args:
            params: The parameters to compress.
            additional_info: Additional information to compress.

        """
        final_params = []

        total_length = params.__len__() + additional_info["lowrank_compressed_state"].__len__()
        for pos in range(total_length):
            if pos in additional_info["lowrank_compressed_state"]:
                u, s, vt = additional_info["lowrank_compressed_state"][pos]
                final_params.append(u @ np.diag(s) @ vt)  # params approximation
            else:
                final_params.append(params.pop(0))

        return final_params
