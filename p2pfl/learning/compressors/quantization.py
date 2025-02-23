import numpy as np

from p2pfl.learning.compressors import CompressionStrategy


class PTQuantization(CompressionStrategy):
    """Post-Training Quantization (PTQ)."""

    def apply_strategy(self, payload: dict, dtype=np.float16) -> list[np.ndarray]:
        """Compress the parameters."""
        quantized_params = [param.astype(dtype) for param in payload["params"]]
        payload["params"] = quantized_params
        return payload

    def reverse_strategy(self, payload:dict) -> list[np.ndarray]:
        """Decompress the parameters."""
        original_params = [param.astype(np.float32) for param in payload["params"]]
        payload["params"] = original_params
        return payload

    def get_category(self) -> str:
        """Get the category of the compression strategy."""
        return "quantization"
