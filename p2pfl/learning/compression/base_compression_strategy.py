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
from typing import Any

import numpy as np


class CompressionStrategy(ABC):
    """Abstract class for optimization strategies."""

    @abstractmethod
    def apply_strategy(self, *args, **kwargs) -> Any:
        """Apply strategy to the parameters."""
        pass

    @abstractmethod
    def reverse_strategy(self, *args, **kwargs) -> Any:
        """Reverse the strategy."""
        pass


class TensorCompressor(CompressionStrategy):
    """Subclass for tensor compression strategies."""

    @abstractmethod
    def apply_strategy(self, params: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
        """Apply strategy to the parameters."""
        pass

    @abstractmethod
    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """Reverse the strategy."""
        pass


class ByteCompressor(CompressionStrategy):
    """Subclass for byte compression strategies."""

    @abstractmethod
    def apply_strategy(self, data: bytes) -> bytes:
        """Apply strategy to the parameters."""
        pass

    @abstractmethod
    def reverse_strategy(self, data: bytes) -> bytes:
        """Reverse the strategy."""
        pass
