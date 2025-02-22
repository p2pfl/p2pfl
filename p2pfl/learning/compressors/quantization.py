import numpy as np

from p2pfl.learning.compressors import CompressionStrategy


class PTQuantization(CompressionStrategy):
    """Post-Training Quantization (PTQ)."""

    def apply_strategy(self, params: list[np.ndarray], dtype=np.float16) -> list[np.ndarray]:
        """Compress the parameters."""
        return [param.astype(dtype) for param in params]

    def reverse_strategy(self, params: list[np.ndarray]) -> list[np.ndarray]:
        """Decompress the parameters."""
        return [param.astype(np.float32) for param in params]

    def get_category(self) -> str:
        """Get the category of the compression strategy."""
        return "quantization"
