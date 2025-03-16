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

import zlib

import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


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

    def get_category(self):
        return "encoder"