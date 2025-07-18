#
# This file is part of the federated_learning_p2p (p2pfl) distribution
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

"""Flax Model for P2PFL."""

import copy
import pickle
from typing import Any

import numpy as np
from flax import linen as nn

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

#####################
#    FlaxModel      #
#####################


class FlaxModel(P2PFLModel):
    """P2PFL model abstraction for Flax."""

    def __init__(
        self,
        model: nn.Module,
        init_params: dict[str, Any],
        params: list[np.ndarray] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize Flax model."""
        # TODO: fix: when using arg params for jax params, fedavg.aggregate fails in models[0].build_copy(params=accum, ...)
        # FlaxModel.__init__() got multiple values for argument 'params'. Fix in __init__ or build_copy.
        super().__init__(model, None, num_samples, contributors, additional_info, compression)
        self.model_params = init_params
        if params:
            if isinstance(params, bytes):
                params, _ = self.decode_parameters(params)
            self.model_params = self.__np_to_dict(self.model_params, params)

    @staticmethod
    def __dict_to_np(params: dict[str, Any]) -> list[np.ndarray]:
        return [np.array(v) for layer in params.values() for v in layer.values()]

    @staticmethod
    def __np_to_dict(target: dict[str, Any], params: list[np.ndarray]) -> dict[str, Any]:
        for i, layer_name in enumerate(target.keys()):
            for j, param_name in enumerate(target[layer_name].keys()):
                target[layer_name][param_name] = params[i * len(target[layer_name]) + j]
        return target

    def get_parameters(self) -> list[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model as a list of NumPy arrays.

        """
        return self.__dict_to_np(self.model_params) if self.model_params else []

    def set_parameters(self, params: list[np.ndarray] | bytes) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        if isinstance(params, bytes):
            params, _ = self.decode_parameters(params)

        try:
            if isinstance(params, list):
                self.__np_to_dict(self.model_params, params)
            else:
                raise ValueError("Unvalid parameters.")
        except Exception as e:
            raise ModelNotMatchingError("Not matching models") from e

    def encode_parameters(self, params: list[np.ndarray] | None = None) -> bytes:
        """
        Encode the parameters of the model.

        Args:
            params: The parameters of the model.

        """
        model_params = (
            self.model_params
            if not (params and self.model_params)
            else self.__np_to_dict(
                copy.deepcopy(self.model_params),
                params,
            )
        )
        data_to_serialize = {
            "params": model_params,
            "additional_info": self.additional_info,
        }
        return pickle.dumps(data_to_serialize)

    def decode_parameters(self, data: bytes) -> tuple[list[np.ndarray], dict[str, Any]]:
        """
        Decode the parameters of the model.

        Args:
            data: The parameters of the model.

        """
        try:
            loaded_data = pickle.loads(data)
            params_dict = loaded_data["params"]
            additional_info = loaded_data.get("additional_info", {})
            params = self.__dict_to_np(params_dict)
            return params, additional_info
        except Exception as e:
            raise ModelNotMatchingError("Error decoding parameters") from e

    def build_copy(self, **kwargs) -> "P2PFLModel":
        """
        Build a copy of the model.

        Args:
            **kwargs: Parameters of the model initialization.

        Returns:
            A copy of the model.

        """
        flax_model = self.__class__(copy.deepcopy(self.model), copy.deepcopy(self.model_params), **kwargs)
        return flax_model

    def get_framework(self) -> str:
        """
        Retrieve the model framework name.

        Returns:
            The name of the model framework.

        """
        return Framework.FLAX.value
