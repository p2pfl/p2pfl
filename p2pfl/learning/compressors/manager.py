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

from p2pfl.learning.compressors import COMPRESSION_REGISTRY


class CompressionManager():
    """Manager for compression strategies."""

    @staticmethod
    def compress(data: dict[str, Any], techniques: dict[str, dict[str, Any]]) -> bytes:
        """Apply compression techniques in sequence."""
        applied_techniques = []
        encoder_instance = None
        payload = data["payload"]

        for name, fn_params in techniques.items():
            if name not in COMPRESSION_REGISTRY:
                raise ValueError(f"Unknown compression technique: {name}")
            instance = COMPRESSION_REGISTRY[name](**fn_params)
            if instance.get_category() == "encoder":
                encoder_instance = instance
                encoder_key = name
            else:

                payload = instance.apply_strategy(payload, **fn_params)
            applied_techniques.append(name)

        data["header"]["applied_techniques"] = applied_techniques
        # apply encoder
        payload = pickle.dumps(payload)
        if encoder_instance is not None:
            payload = encoder_instance.apply_strategy(payload)
            data["header"]["encoder"] = encoder_key

        data["params"] = payload
        return data


    @staticmethod
    def decompress(data: dict) -> list[np.ndarray]:
        """Apply decompression techniques in sequence."""
        applied_techniques = data["header"]["applied_techniques"]
        encoder_key = data["header"]["encoder"]
        payload = data["payload"]

        if encoder_key is not None:
            encoder_instance = COMPRESSION_REGISTRY[encoder_key]()
            payload = encoder_instance.reverse_strategy(payload)

        payload = pickle.loads(payload)
        for name in reversed(applied_techniques):
            instance = COMPRESSION_REGISTRY[name]()
            payload = instance.reverse_strategy(payload)

        return payload

