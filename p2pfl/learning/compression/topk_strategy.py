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

"""Top K Sparsification optimization strategy."""

import numpy as np

from p2pfl.learning.compression.base_compression_strategy import TensorCompressor


class TopKSparsification(TensorCompressor):
    """
    Top-K sparsification.

    Keeps only the top k largest values in model parameters.
    """

    def apply_strategy(self, params: list[np.ndarray], k: float = 0.1) -> tuple[list[np.ndarray], dict]:
        """
        Reduces params by taking only the top k ones.

        Args:
            params: The parameters to compress.
            k: Percentage of parameters to keep between 0 and 1.

        """
        new_params = []
        sparse_metadata = {}

        for pos, param in enumerate(params):
            k_elements = max(1, int(k * param.size))
            flattened = param.flatten()
            indices = np.argpartition(flattened, -k_elements)[-k_elements:]
            values = flattened[indices]

            sparse_metadata[pos] = {"indices": indices.astype(np.uint32), "shape": param.shape}

            new_params.append(values)
        return new_params, {"topk_sparse_metadata": sparse_metadata}

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Decompress params.

        Args:
            params: The parameters to decompress.
            additional_info: Additional information to decompress.

        """
        reconstructed_params = []
        sparse_metadata = additional_info["topk_sparse_metadata"]

        for pos, values in enumerate(params):
            if pos in sparse_metadata:
                # this layer was compressed, reconstruct it
                # storing 0s where the values were not present
                meta = sparse_metadata[pos]
                indices = meta["indices"]
                shape = meta["shape"]

                full_array = np.zeros(np.prod(shape), dtype=values.dtype)
                full_array[indices] = values
                full_array = full_array.reshape(shape)
                reconstructed_params.append(full_array)
            else:
                # if not in sparse_metadata, it was not compressed
                reconstructed_params.append(values)

        return reconstructed_params
