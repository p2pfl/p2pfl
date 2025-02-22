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

"""Compression manager."""

import pickle
from typing import Any

import numpy as np

from p2pfl.learning.compressors import COMPRESSION_BITMASK, COMPRESSION_REGISTRY


class CompressionManager():
    """Manager for compression strategies."""

    @staticmethod
    def compress(data_to_serialize: dict[str, Any], techniques: dict[str, dict[str, Any]]) -> bytes:
        """Apply compression techniques in sequence."""
        bitmask = 0b00000000
        params, additional_info, metadata = data_to_serialize.items()
        for technique_name, technique_params in techniques.items():
            if technique_name not in COMPRESSION_REGISTRY:
                raise ValueError(f"Unknown compression technique: {technique_name}")
            if technique_name in COMPRESSION_BITMASK:
                bitmask |= COMPRESSION_BITMASK[technique_name]
            technique_instance = COMPRESSION_REGISTRY[technique_name](**technique_params)
            params = technique_instance.compress(params)

        metadata["bitmask"] = bitmask
        data_to_serialize["metadata"] = metadata
        return pickle.dumps(data_to_serialize)

    @staticmethod
    def decompress(data_deserialized: bytes) -> list[np.ndarray]:
        """Apply decompression techniques in sequence."""
        bitmask = data_deserialized["metadata"]["bitmask"] # COMPROBAR EN DECODE si existe para entrar aquí ya
        params = data_deserialized["params"]
        decompression_pipeline = [
            name for name, mask in COMPRESSION_BITMASK.items() if bitmask & mask
        ]
        for strategy_name in reversed(decompression_pipeline):
            strategy = COMPRESSION_REGISTRY[strategy_name]()
            params = strategy.decompress(params)
        return params
