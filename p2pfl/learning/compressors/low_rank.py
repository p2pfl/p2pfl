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

import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


class LowRankApproximation(CompressionStrategy):
    """Low Rank compression strategy."""

    def apply_strategy(self, payload: dict, threshold: float = 0.95):
        """
        Approximate the parameters preserving a target rank or energy threshold.

        Args:
            payload: The payload to compress.
            threshold: Percentage between 0 and 1 of the energy to preserve.

        """
        params_to_share = [] # layers without low rank approximation
        compressed_states = {} # compressed layers

        for pos, layer in enumerate(payload["params"]):
            # if threshold is provided, compute target rank

            if layer.ndim != 2:
                params_to_share.append(layer) # list(np.ndarray)
            else:
                u, s, vt = np.linalg.svd(layer, full_matrices=False)
                # compute number of values to keep so cumulative energy sum is above threshold
                energy_total = np.sum(s ** 2)
                cumulative_energies = np.cumsum(s ** 2)
                target_rank = np.searchsorted(cumulative_energies / energy_total, threshold) + 1
                u, s, vt = u[:, :target_rank], s[:target_rank], vt[:target_rank, :]
                # remove layer from data and store compressed layer
                compressed_states[pos] = u, s, vt


        payload["params"] = params_to_share
        # si añades más cosas (e.g: PTQ), no se usarían en los compressed params, pero está bien así
        payload["additional_info"]["lowrank_compressed_state"] = compressed_states
        return payload


    def reverse_strategy(self, payload: dict): # TODO: Revisar esto
        """Decompress the parameters."""
        decompressed_params = []
        total_length = payload["params"].__len__() + payload["additional_info"]["lowrank_compressed_state"].__len__()
        for pos in range(total_length):
            if pos in payload["additional_info"]["lowrank_compressed_state"]:
                u, s, vt = payload["additional_info"]["lowrank_compressed_state"][pos]
                decompressed_params.append(u @ np.diag(s) @ vt) # params approximation (np.ndarray)
            else:
                decompressed_params.append(payload["params"].pop(0))

        decompressed_params["additional_info"].pop("lowrank_compressed_state")
        return decompressed_params

    def get_category(self):
        """Return the category of the strategy."""
        return "compressor"