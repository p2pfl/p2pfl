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

"""P2PFL model abstraction."""

import copy
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from p2pfl.learning.frameworks.exceptions import DecodingParamsError


class P2PFLModel:
    """
    P2PFL model abstraction.

    This class encapsulates the different models across all the possible frameworks.

    The key concept is the extraction of the model weights in a common format for all the frameworks.

    Args:
        model: The model to encapsulate.

    :: note :: The model type is ANY because the different frameworks do not share a common model type.

    """

    def __init__(
        self,
        model: Any,
        params: Optional[Union[List[np.ndarray], bytes]] = None,
        num_samples: Optional[int] = None,
        contributors: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the model."""
        self.model = model
        self.contributors: List[str] = []
        if contributors is not None:
            self.contributors = contributors
        self.num_samples = 0
        if num_samples is not None:
            self.num_samples = num_samples
        self.additional_info: Dict[str, Any] = {}
        if additional_info is not None:
            self.additional_info = additional_info
        if params is not None:
            self.set_parameters(params)

    def get_model(self) -> Any:
        """Get the model."""
        return self.model

    def encode_parameters(self, params: Optional[List[np.ndarray]] = None) -> bytes:
        """
        Encode the parameters of the model.

        Args:
            params: The parameters of the model.

        """
        if params is None:
            params = self.get_parameters()
        data_to_serialize = {
            "params": params,
            "additional_info": self.additional_info,
        }
        return pickle.dumps(data_to_serialize)

    def decode_parameters(self, data: bytes) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Decode the parameters of the model.

        Args:
            data: The parameters of the model.

        """
        try:
            loaded_data = pickle.loads(data)
            params = loaded_data["params"]
            additional_info = loaded_data["additional_info"]
            return params, additional_info
        except Exception as e:
            raise DecodingParamsError("Error decoding parameters") from e

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        raise NotImplementedError

    def set_parameters(self, params: Union[List[np.ndarray], bytes]) -> None:
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

    def get_info(self, callback: Optional[str] = None) -> Any:
        """
        Get additional information from the learner state.

        Args:
            callback: The callback to add the information
            key: The key of the information.

        """
        if callback is None:
            return self.additional_info
        return self.additional_info[callback]

    def set_contribution(self, contributors: List[str], num_samples: int) -> None:
        """
        Set the contribution of the model.

        Args:
            contributors: The contributors of the model.
            num_samples: The number of samples used to train this model.

        """
        self.contributors = contributors
        self.num_samples = num_samples

    def get_contributors(self) -> List[str]:
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

    def get_framework(self) -> str:
        """
        Retrieve the model framework name.

        Returns:
            The name of the model framework.

        """
        raise NotImplementedError
