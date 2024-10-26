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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flax import linen as nn

from p2pfl.learning.exceptions import ModelNotMatchingError
from p2pfl.learning.p2pfl_model import P2PFLModel

#####################
#    FlaxModel      #
#####################


class FlaxModel(P2PFLModel):
    """P2PFL model abstraction for Flax."""

    model: nn.Module = None
    model_params: Optional[Dict[str, Any]] = None

    @staticmethod
    def __dict_to_np(params: Dict[str, Any]) -> List[np.ndarray]:
        return [np.array(v) for layer in params.values() for v in layer.values()]

    @staticmethod
    def __np_to_dict(target: Dict[str, Any], params: List[np.ndarray]) -> Dict[str, Any]:
        for i, layer_name in enumerate(target.keys()):
            for j, param_name in enumerate(target[layer_name].keys()):
                target[layer_name][param_name] = params[i * len(target[layer_name]) + j]
        return target

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model as a list of NumPy arrays.

        """
        return self.__dict_to_np(self.model_params)

    def set_parameters(self, params: Union[List[np.ndarray], bytes]) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        if isinstance(params, bytes):
            params = self.decode_parameters(params)

        try:
            if self.model_params is None:
                self.model_params = params
            else:
                self.__np_to_dict(self.model_params, params)
        except Exception as e:
            raise ModelNotMatchingError("Not matching models") from e

    def encode_parameters(self, params: Optional[List[np.ndarray]] = None) -> bytes:
        """
        Encode the parameters of the model.

        Args:
            params: The parameters of the model.

        """
        params = self.model_params if params is None else self.__np_to_dict(copy.deepcopy(self.model_params), params)
        return pickle.dumps(params)

    def decode_parameters(self, data: bytes) -> List[np.ndarray]:
        """
        Decode the parameters of the model.

        Args:
            data: The parameters of the model.

        """
        params_dict = pickle.loads(data)
        return self.__dict_to_np(params_dict)

    def build_copy(self, **kwargs) -> "P2PFLModel":
        """
        Build a copy of the model.

        Args:
            **kwargs: Parameters of the model initialization.

        Returns:
            A copy of the model.

        """
        flax_model = super().build_copy(**kwargs)
        flax_model.model_params = copy.deepcopy(self.model_params)
        return flax_model


####
# Example MLP in Flax
####


class MLP(nn.Module):
    """Multilayer Perceptron (MLP) for MNIST classification using Flax."""

    hidden_sizes: Tuple[int, int] = (256, 128)
    out_channels: int = 10

    @nn.compact
    def __call__(self, x):
        """
        Define the forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input tensor, expected to be a flattened MNIST image,
                             or a batch of images with shape (batch_size, image_size).

        Returns:
            jnp.ndarray: The output logits of the MLP, with shape (batch_size, out_channels).
                         These represent the unnormalized scores for each class.

        """
        x = x.reshape((1, -1))
        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(features=size)(x))
        x = nn.Dense(features=self.out_channels)(x)
        return x
