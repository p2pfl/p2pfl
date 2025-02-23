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

