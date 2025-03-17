import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


class PTQuantization(CompressionStrategy):
    """Post-Training Quantization (PTQ)."""

    def apply_strategy(self, payload: dict, dtype=np.float16) -> list[np.ndarray]:
        """
        Reduce the precission of model parameters.

        Args:
            payload: Dict with payload to quantize
            dtype: The desired precision

        """
        payload["additional_info"]["ptq_original_dtype"] = payload["params"][0].dtype
        quantized_params = [param.astype(dtype) for param in payload["params"]]
        payload["params"] = quantized_params

        return payload

    def reverse_strategy(self, payload:dict) -> list[np.ndarray]:
        """
        Return model parameters to saved original precission.

        Args:
            payload: Dict with payload to restore.

        """
        original_dtype = payload["additional_info"]["ptq_original_dtype"]
        original_params = [param.astype(original_dtype) for param in payload["params"]]
        payload["params"] = original_params
        payload["additional_info"].pop("ptq_original_dtype", None)
        return payload

    def get_category(self) -> str:
        """Return the category of the strategy."""
        return "compressor"