#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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

"""P2PFL model abstraction."""

import copy
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from p2pfl.learning.compression.manager import CompressionManager
from p2pfl.learning.frameworks.exceptions import DecodingParamsError


class P2PFLModel(ABC):
    """
    Abstract base class for all P2PFL models.

    This class encapsulates the different models across all the possible frameworks.
    The key concept is the extraction of the model weights in a common format for all the frameworks.

    Important:
        Inherit from ``WeightBasedModel`` or ``TreeBasedModel`` instead, which handle
        neural networks and tree ensembles respectively with proper aggregator compatibility.

    Args:
        model: The model to encapsulate.

    Note:
        The model type is ANY because different model types have different parameter formats.

    """

    def __init__(
        self,
        model: Any,
        params: Any = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the model."""
        self.model = model
        self.contributors: list[str] = []
        if contributors is not None:
            self.contributors = contributors
        self.num_samples = 0
        if num_samples is not None:
            self.num_samples = num_samples
        self.additional_info: dict[str, Any] = {}
        if additional_info is not None:
            self.additional_info = additional_info
        if params is not None:
            self.set_parameters(params)
        if compression is not None:
            self.compression = compression
        else:
            self.compression = {}

    def get_model(self) -> Any:
        """Get the model."""
        return self.model

    def encode_parameters(self, params: Any = None) -> bytes:
        """
        Encode the parameters of the model.

        Args:
            params: The parameters of the model.

        """
        if params is None:
            params = self.get_parameters()

        return CompressionManager.apply(params, self.additional_info, self.compression)

    def decode_parameters(self, data: bytes) -> tuple[Any, dict[str, Any]]:
        """
        Decode the parameters of the model.

        Args:
            data: The serialized parameters.

        """
        try:
            return CompressionManager.reverse(data)
        except Exception as e:
            raise DecodingParamsError("Error decoding parameters") from e

    @abstractmethod
    def get_parameters(self) -> Any:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model.

        """
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, params: Any) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        raise NotImplementedError

    def add_info(self, callback: str, info: Any) -> None:
        """
        Add additional information to the learner state.

        Args:
            callback: The callback to add the information
            info: The information for the callback.

        """
        self.additional_info[callback] = info

    def get_info(self, callback: str | None = None) -> Any:
        """
        Get additional information from the learner state.

        Args:
            callback: The callback to add the information
            key: The key of the information.

        """
        if callback is None:
            return self.additional_info
        return self.additional_info[callback]

    def set_contribution(self, contributors: list[str], num_samples: int) -> None:
        """
        Set the contribution of the model.

        Args:
            contributors: The contributors of the model.
            num_samples: The number of samples used to train this model.

        """
        self.contributors = contributors
        self.num_samples = num_samples

    def get_contributors(self) -> list[str]:
        """Get the contributors of the model."""
        if self.contributors == []:
            raise ValueError("Contributors are empty")
        return self.contributors

    def get_num_samples(self) -> int:
        """Get the number of samples used to train this model."""
        if self.num_samples == 0:
            raise ValueError("Number of samples required")
        return self.num_samples

    def build_copy(self, **kwargs) -> "P2PFLModel":
        """
        Build a copy of the model.

        Args:
            **kwargs: Parameters of the model initialization.

        Returns:
            A copy of the model.

        """
        return self.__class__(copy.deepcopy(self.model), **kwargs)

    @abstractmethod
    def get_framework(self) -> str:
        """
        Retrieve the model framework name.

        Returns:
            The name of the model framework.

        """
        raise NotImplementedError


class P2PFLModelDecorator(P2PFLModel):
    """Dynamic wrapper for P2PFLModel. Delegates all attribute access to the wrapped model."""

    def __init__(self, wrapped_model: "P2PFLModel") -> None:
        """Initialize wrapper with a P2PFLModel instance."""
        object.__setattr__(self, "_wrapped_model", wrapped_model)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped model."""
        return getattr(self._wrapped_model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to wrapped model."""
        if name == "_wrapped_model":
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped_model, name, value)

    def get_model(self) -> Any:
        """Get the model."""
        return self._wrapped_model.get_model()

    def get_parameters(self) -> Any:
        """Get model parameters."""
        return self._wrapped_model.get_parameters()

    def set_parameters(self, params: Any) -> None:
        """Set model parameters."""
        self._wrapped_model.set_parameters(params)

    def get_framework(self) -> str:
        """Get framework name."""
        return self._wrapped_model.get_framework()

    def encode_parameters(self, params: Any = None) -> bytes:
        """Encode the parameters of the model."""
        return self._wrapped_model.encode_parameters(params)

    def decode_parameters(self, data: bytes) -> tuple[Any, dict[str, Any]]:
        """Decode the parameters of the model."""
        return self._wrapped_model.decode_parameters(data)

    def add_info(self, callback: str, info: Any) -> None:
        """Add additional information to the learner state."""
        self._wrapped_model.add_info(callback, info)

    def get_info(self, callback: str | None = None) -> Any:
        """Get additional information from the learner state."""
        return self._wrapped_model.get_info(callback)

    def set_contribution(self, contributors: list[str], num_samples: int) -> None:
        """Set the contribution of the model."""
        self._wrapped_model.set_contribution(contributors, num_samples)

    def get_contributors(self) -> list[str]:
        """Get the contributors of the model."""
        return self._wrapped_model.get_contributors()

    def get_num_samples(self) -> int:
        """Get the number of samples used to train this model."""
        return self._wrapped_model.get_num_samples()

    def build_copy(self, **kwargs: Any) -> "P2PFLModel":
        """Build a copy of the model."""
        return self._wrapped_model.build_copy(**kwargs)


class WeightBasedModel(P2PFLModel):
    """
    Base class for neural network models (PyTorch, TensorFlow, Flax).

    Returns parameters as ``list[np.ndarray]`` weight tensors. Compatible with ``WeightAggregator``.

    """

    def decode_parameters(self, data: bytes) -> tuple[list[np.ndarray], dict[str, Any]]:
        """
        Decode the parameters of the model.

        Args:
            data: The serialized parameters.

        Returns:
            Tuple of (weight arrays, additional info).

        """
        params, info = super().decode_parameters(data)
        return list(params), info

    @abstractmethod
    def get_parameters(self) -> list[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            List of numpy arrays, one per layer.

        """
        raise NotImplementedError


class TreeBasedModel(P2PFLModel):
    """
    Base class for tree ensemble models (XGBoost).

    Returns parameters as ``dict[str, Any]`` tree structures. Compatible with ``TreeAggregator``.

    """

    def decode_parameters(self, data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Decode the parameters of the model.

        Args:
            data: The serialized parameters.

        Returns:
            Tuple of (tree structure dict, additional info).

        """
        params, info = super().decode_parameters(data)
        return dict(params), info

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """
        Get the parameters of the model.

        Returns:
            Parsed tree structure as a dictionary (XGBoost JSON format).

        """
        raise NotImplementedError
