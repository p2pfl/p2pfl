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

"""Compression strategy interface."""

from abc import ABC, abstractmethod

import numpy as np


class CompressionStrategy(ABC):
    """Abstract class for optimization strategies."""

    @abstractmethod
    def apply_strategy(self,  payload: dict) -> bytes:
        """Apply strategy to the parameters."""
        pass

    @abstractmethod
    def reverse_strategy(self,  payload: dict) -> list[np.ndarray]:
        """Reverse the strategy."""
        pass

class BaseCompressor(CompressionStrategy):
    """Subclass for compression strategies that use raw data."""

    pass

class EncoderStrategy(CompressionStrategy):
    """Subclass for compression strategies that need to handle binary data."""

    pass
