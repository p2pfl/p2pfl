import zlib

import numpy as np

from p2pfl.learning.compressors.compression import CompressionStrategy


class ZlibCompressor(CompressionStrategy):
    """Lossless compression using zlib."""

    def apply_strategy(self, params: list[np.ndarray], level=6) -> bytes:
        """Compress the parameters."""
        return zlib.compress(params, level=level)

    def reverse_strategy(self, compressed_params: bytes) -> list[np.ndarray]:
        """Decompress the parameters."""
        return zlib.decompress(compressed_params)

    def get_category(self) -> str:
        return "lossless_compression"