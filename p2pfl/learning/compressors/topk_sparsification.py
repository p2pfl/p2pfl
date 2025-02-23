import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


class TopKSparsification(CompressionStrategy):
    """Top-K sparsification."""

    def apply_strategy(self, data:dict, k:int = 0.1) -> dict:
        """Compress the parameters."""
        new_params = []
        sparse_metadata = {}

        for pos, param in enumerate(data["params"]):
            k_elements = max(1, int(k * param.size))
            flattened = param.flatten()
            indices = np.argsort(flattened)[-k_elements:]
            values = flattened[indices]

            sparse_metadata[pos] = {
                "indices": indices.astype(np.uint32),
                "shape": param.shape
            }

            new_params.append(values)
        data["params"] = new_params
        data["additional_info"]["sparse_metadata"] = sparse_metadata
        return data

    def reverse_strategy(self, data:dict) -> dict:
        """Decompress params."""
        reconstructed_params = []
        sparse_metadata = data["additional_info"].get("sparse_metadata", {})

        for pos, values in enumerate(data["params"]):
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

        data["params"] = reconstructed_params
        return data

    def get_category(self) -> str:
        """Get the category of the compression strategy."""
        return "sparsification"
