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
# along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Post-Training Quantization (PTQ) compression strategy."""

import numpy as np

from p2pfl.learning.compression.base_compression_strategy import CompressionStrategy


class PTQuantization(CompressionStrategy):
    """Post-Training Quantization (PTQ)."""

    def apply_strategy(self, payload: dict, dtype=np.float16) -> list[np.ndarray]:
        """
        Reduce the precission of model parameters.

        Args:
            payload: Payload to quantize.
            dtype: The desired precision.

        """
        payload["additional_info"]["ptq_original_dtype"] = payload["params"][0].dtype
        quantized_params = [param.astype(dtype) for param in payload["params"]]
        payload["params"] = quantized_params

        return payload

    def reverse_strategy(self, payload:dict) -> list[np.ndarray]:
        """
        Return model parameters to saved original precission.

        Args:
            payload: Payload to restore.

        """
        original_dtype = payload["additional_info"]["ptq_original_dtype"]
        original_params = [param.astype(original_dtype) for param in payload["params"]]
        payload["params"] = original_params
        payload["additional_info"].pop("ptq_original_dtype", None)
        return payload

    def get_category(self) -> str:
        """Return the category of the strategy."""
        return "loseless_compressor"
