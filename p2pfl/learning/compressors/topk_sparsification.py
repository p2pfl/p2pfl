import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


class TopKSparsification(CompressionStrategy):
    """Top-K sparsification."""

    def apply_strategy(self, payload:dict, k:float = 0.1) -> dict:
        """
        Keep only the top k largest values in model parameters.

        Args:
            payload: Model parameters.
            k: Percentage of parameters to keep between 0 and 1.

        """
        new_params = []
        sparse_metadata = {}

        for pos, param in enumerate(payload["params"]):
            k_elements = max(1, int(k * param.size))
            flattened = param.flatten()
            indices = np.argpartition(flattened, -k_elements)[-k_elements:]
            values = flattened[indices]

            sparse_metadata[pos] = {
                "indices": indices.astype(np.uint32),
                "shape": param.shape
            }

            new_params.append(values)
        payload["params"] = new_params
        payload["additional_info"]["topk_sparse_metadata"] = sparse_metadata
        return payload

    def reverse_strategy(self, data:dict) -> dict:
        """Decompress params."""
        reconstructed_params = []
        sparse_metadata = data["additional_info"].get("topk_sparse_metadata", {})

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
        data["additional_info"].pop("topk_sparse_metadata", None)
        return data

    def get_category(self) -> str:
        """Return the category of the strategy."""
        return "compressor"