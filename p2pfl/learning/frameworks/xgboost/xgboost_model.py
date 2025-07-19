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

"""XGBoost model wrapper for P2PFL."""
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import xgboost as xgb
from sklearn.exceptions import NotFittedError

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class XGBoostModel(P2PFLModel):
    """
    P2PFL model abstraction for XGBoost.

    Wraps an XGBoost Booster or XGBClassifier for federated updates.

    Args:
        model: The XGBoost model to encapsulate (Booster or XGBClassifier).
        params: Serialized parameters (list of ndarrays or bytes).
        num_samples: Number of samples used in training.
        contributors: List of contributor IDs.
        additional_info: Extra metadata.
        compression: Optional compression settings.
    """

    def __init__(
        self,
        model: xgb.XGBModel,
        params: Optional[Union[List[np.ndarray], bytes]] = None,
        num_samples: Optional[int] = None,
        contributors: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        compression: Optional[Dict[str, Dict[str, Any]]] = None,
        id: Optional[int] = None
    ) -> None:
        if not isinstance(model, xgb.XGBModel):
            raise ModelNotMatchingError("Provided model is not an XGBoost sklearn model")
        super().__init__(model, params, num_samples, contributors, additional_info, compression)
        self.id = id if id is not None else 0  # Default ID if not provided

    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as numpy arrays.

        Returns:
            List of parameter arrays corresponding to each model attribute.
        """
        # Serialize booster to raw bytes
        # raw = self.model.get_booster().save_raw()
        # arr = np.frombuffer(raw, dtype=np.uint8)
        try:
            file_name = f"temp_model{self.id}_{self.model._estimator_type}.json"
            self.model.save_model(file_name)  # Save to JSON for compatibility
            print(f"MODEL SAVED TO {file_name}")
            return [np.array(file_name)]
        except NotFittedError:
            print(f"NOT FITTED FOR MODEL {self.id}")
            return []

    def get_file_name(self) -> str:
        """
        Returns the file name of the model.
        This is used to identify the model in federated learning.
        """
        try:
            file_name = f"temp_model{self.id}_{self.model._estimator_type}.json"
            self.model.save_model(file_name)  # Save to JSON for compatibility
            print(f"MODEL SAVED TO {file_name}")
            return file_name
        except NotFittedError:
            print(f"NOT FITTED FOR MODEL {self.id}")
            return ""

    def set_parameters(self, params: Union[List[np.ndarray], bytes]) -> None:
        """
        Set model parameters from numpy arrays or serialized bytes.

        Args:
            params: List of ndarrays or serialized bytes.
        Raises:
            ModelNotMatchingError: if loading fails.
        """
        # If bytes, decode compression first
        if isinstance(params, bytes):
            params = self.decode_parameters(params)
        if params is None or len(params) == 0:
            pass
            # type_of_model = self.model._estimator_type
            # if type_of_model == "classifier":
            #     # Load as XGBClassifier
            #     model = xgb.XGBClassifier()
            # else:
            #     model = xgb.XGBRegressor()
            # self.model = model
        else:
            params = np.array2string(params[0]).replace("'","")
            file_name = params
            type_of_model = file_name.replace(".json", "").split("_")[-1] # Extract type from filename
            if type_of_model == "classifier":
                # Load as XGBClassifier
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
            model.load_model(file_name)  # Load from JSON for compatibility
            # remove the temporary file
            # os.remove(file_name)
            # booster.load_model(raw_bytes)

            # Attach loaded booster to sklearn API model
            self.model = model

    def get_framework(self) -> str:
        """
        Returns the framework name identifier.
        """
        return Framework.XGBOOST.value

