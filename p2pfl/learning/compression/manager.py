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
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Transmission compression manager."""

import pickle
from typing import Any

import numpy as np

from p2pfl.learning.compression import COMPRESSION_STRATEGIES_REGISTRY
from p2pfl.learning.compression.base_compression_strategy import ByteCompressor


class CompressionManager:
    """Manager for compression strategies."""

    @staticmethod
    def get_registry() -> dict[str, Any]:
        """Return the registry of compression strategies."""
        return COMPRESSION_STRATEGIES_REGISTRY

    @staticmethod
    def apply(params: list[np.ndarray], additional_info: dict, techniques: dict[str, dict[str, Any]]) -> bytes:
        """
        Apply compression techniques in sequence to the data.

        Args:
            params: The parameters to compress.
            additional_info: Additional information to compress.
            techniques: The techniques to apply.

        """
        # Init
        registry = CompressionManager.get_registry()
        applied_techniques = []
        byte_compressor: ByteCompressor | None = None
        encoder_key: str | None = None

        # apply techniques in sequence
        for name, fn_params in techniques.items():
            if name not in registry:
                raise ValueError(f"Unknown compression technique: {name}")
            instance = registry[name]()
            # encoder gets applied at the end since needs serialized data
            if isinstance(instance, ByteCompressor):
                if byte_compressor is not None:
                    raise ValueError("Only one encoder can be applied at a time")
                byte_compressor = instance
                encoder_key = name
            else:
                params, compression_settings = instance.apply_strategy(params, **fn_params)
                applied_techniques.append([name, compression_settings])

        # Build data transfer dict
        data = {
            "params": params,
            "additional_info": additional_info | {"applied_techniques": applied_techniques},
        }
        data_bytes = pickle.dumps(data)

        # apply byte_compressor if exists
        if byte_compressor is not None:
            data_bytes = byte_compressor.apply_strategy(data_bytes)

        # Return data
        return pickle.dumps(
            {
                "byte_compressor": encoder_key,
                "bytes": data_bytes,
            }
        )

    @staticmethod
    def reverse(data: bytes) -> tuple[list[np.ndarray], dict]:
        """
        Reverse compression techniques in sequence.

        Args:
            data: The deserialized data to reverse (inner data is serialized).

        """
        # Init
        registry = CompressionManager.get_registry()
        raw_data = pickle.loads(data)

        # Check if byte compressor was applied
        encoder_key = raw_data.get("byte_compressor", None)
        if encoder_key is not None:
            byte_compressor = registry[encoder_key]()
            data_bytes = byte_compressor.reverse_strategy(raw_data["bytes"])
        else:
            data_bytes = raw_data["bytes"]
        data = pickle.loads(data_bytes)
        if not isinstance(data, dict):
            raise ValueError("Invalid data format")

        # Get applied techniques
        params = data["params"]
        if "additional_info" not in data:
            raise ValueError("No additional info found in data. Impossible to reverse the compression.")
        applied_techniques = data["additional_info"].pop("applied_techniques")
        for compressor_name, compressor_info in reversed(applied_techniques):
            instance = registry[compressor_name]()
            params = instance.reverse_strategy(params, compressor_info)

        return params, data["additional_info"]
