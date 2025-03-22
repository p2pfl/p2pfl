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

from p2pfl.learning.compression.base_compression_strategy import TensorCompressor


class PTQuantization(TensorCompressor):
    """Post-Training Quantization (PTQ)."""

    def apply_strategy(self, params: list[np.ndarray], dtype: str = "float16") -> tuple[list[np.ndarray], dict]:
        """
        Reduce the precission of model parameters.

        Args:
            params: The parameters to compress.
            dtype: The desired precision.

        """
        return [param.astype(np.dtype(dtype)) for param in params], {"ptq_original_dtype": params[0].dtype}

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Return model parameters to saved original precission.

        Args:
            params: The parameters to decompress.
            additional_info: Additional information to decom

        """
        return [param.astype(additional_info["ptq_original_dtype"]) for param in params]
