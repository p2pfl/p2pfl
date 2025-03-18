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

from p2pfl.learning.compression import COMPRESSION_STRATEGIES_REGISTRY


class CompressionManager:
    """Manager for compression strategies."""

    @staticmethod
    def get_registry():
        """Return the registry of compression strategies."""
        return COMPRESSION_STRATEGIES_REGISTRY

    @staticmethod
    def apply(data: dict[str, Any], techniques: dict[str, dict[str, Any]]) -> bytes:
        """
        Apply compression techniques in sequence to the data.

        Args:
            data: The data to optimize.
            techniques: The techniques to apply.

        """
        registry = CompressionManager.get_registry()
        applied_techniques = []
        encoder_instance = None
        payload = data["payload"]

        # if no payload, skip compression
        if not payload:
            return pickle.dumps(data)

        # apply techniques in sequence
        for name, fn_params in techniques.items():
            if name not in registry:
                raise ValueError(f"Unknown compression technique: {name}")
            instance = registry[name]()
            # encoder gets applied at the end since needs serialized data
            if instance.get_category() == "encoder":
                encoder_instance = instance
                encoder_key = name
            else:
                payload = instance.apply_strategy(payload, **fn_params)
                applied_techniques.append(name)

        # save applied techniques to reverse
        data["header"]["applied_techniques"] = applied_techniques

        # apply encoder if exists on serialized payload
        payload_serialized = pickle.dumps(payload)
        if encoder_instance is not None:
            payload_serialized = encoder_instance.apply_strategy(payload_serialized)
            data["header"]["encoder"] = encoder_key

        data["payload"] = payload_serialized
        return pickle.dumps(data)  # reserialize entire payload after encoder

    @staticmethod
    def reverse(data: dict) -> dict:
        """
        Reverse compression techniques in sequence.

        Args:
            data: The deserialized data to reverse (inner data is serialized).

        """
        registry = CompressionManager.get_registry()
        applied_techniques = data["header"]["applied_techniques"]
        encoder_key = data["header"].get("encoder", None)
        payload = data["payload"]

        # if encoder was applied, inner data needs to be decoded before deserialize
        if encoder_key is not None:
            encoder_instance = registry[encoder_key]()
            payload = encoder_instance.reverse_strategy(payload)
        payload = pickle.loads(payload)  # deserialize
        # reverse strategies in reverse order
        for name in reversed(applied_techniques):
            instance = registry[name]()
            payload = instance.reverse_strategy(payload)

        return payload
