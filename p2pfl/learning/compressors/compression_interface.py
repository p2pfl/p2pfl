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

from abc import ABC, abstractmethod

import numpy as np


class CompressionStrategy(ABC):
    """Abstract class for compression strategies."""

    @abstractmethod
    def apply_strategy(self,  payload: dict) -> bytes:
        """Compress the parameters."""
        pass

    @abstractmethod
    def reverse_strategy(self,  payload: dict) -> list[np.ndarray]:
        """Decompress the parameters."""
        pass

    @abstractmethod
    def get_category(self) -> str:
        """Get the category of the compression strategy."""
        pass

