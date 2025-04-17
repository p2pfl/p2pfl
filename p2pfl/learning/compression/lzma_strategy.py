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

"""LZMA optimization strategy."""

import lzma

from p2pfl.learning.compression.base_compression_strategy import ByteCompressor


class LZMACompressor(ByteCompressor):
    """
    Lossless compression strategy using LZMA algorithm.

    LZMA provides higher compression ratio than zlib, but it is slower to compress and decompress.

    """

    def apply_strategy(self, data: bytes, preset: int = 5) -> bytes:
        """

        Apply LZMA compression strategy to the parameters.

        Args:
            data : The input data to be compressed.
            preset : Compression level, 0 (fastest) to 9 (most compressed).

        """
        return lzma.compress(data, preset=preset)

    def reverse_strategy(self, data: bytes):
        """Reverse the LZMA compression strategy."""
        return lzma.decompress(data)
