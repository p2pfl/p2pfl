import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


class TopKSparsification(CompressionStrategy):
    """Top-K sparsification."""

    def apply_strategy(self, params: list[np.ndarray], k=0.1) -> bytes:
        """Compress the parameters."""
        compressed_data = []

        for param in params:
            k_elements = max(1, int(k * param.size))
            flattened = param.flatten()
            indices = np.argsort(flattened)[-k_elements:]
            values = flattened[indices]

            compressed_data.append({
                "indices": indices.astype(np.uint32),
                "values": values,
                "shape": param.shape
            })
        return compressed_data

    def reverse_strategy(self, compressed_params: bytes) -> list[np.ndarray]:
        """Decompress the parameters."""
        params = []

        for data in compressed_params:
            indices = data["indices"]
            values = data["values"]
            shape = data["shape"]

            param = np.zeros(np.prod(shape))
            param[indices] = values
            param = param.reshape(shape)

            params.append(param)
        return params

    def get_category(self) -> str:
        """Get the category of the compression strategy."""
        return "sparsification"
