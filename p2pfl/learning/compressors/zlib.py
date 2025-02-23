import zlib

import numpy as np

from p2pfl.learning.compressors.compression import CompressionStrategy

# TODO: NEEDS TO BE APPLIED AT END, SINCE NEEDS SERIALIZED DATA...
class ZlibCompressor(CompressionStrategy):
    """Lossless compression using zlib."""

    def apply_strategy(self, payload: dict, level=6) -> bytes:
        """Compress the parameters."""
        payload["params"] = zlib.compress(payload["params"], level=level)
        return payload

    def reverse_strategy(self, payload: dict) -> list[np.ndarray]:
        """Decompress the parameters."""
        payload["params"] = zlib.decompress(payload["params"])
        return payload

