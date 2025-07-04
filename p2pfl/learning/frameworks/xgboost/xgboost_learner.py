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

"""XGBoost Learner for P2PFL."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import xgboost as xgb
from sklearn.exceptions import NotFittedError

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel
from p2pfl.utils.node_component import allow_no_addr_check
from p2pfl.learning.frameworks.xgboost.xgboost_dataset import XGBoostExportStrategy


class XGBoostLearner(Learner):
    """
    Learner implementation using XGBoost boosting framework.

    Args:
        model: Wrapped XGBoostModel instance.
        data: P2PFLDataset for train/eval split.
        aggregator: Aggregator to use for model updates.
    """

    def interrupt_fit(self) -> None:
        pass

    def __init__(
            self,
            model: Optional[P2PFLModel] = None,
            data: Optional[P2PFLDataset] = None,
            aggregator: Optional[Aggregator] = None
    ) -> None:
        super().__init__(model=model, data=data, aggregator=aggregator)
        # self.__model = model

    def __get_xgb_model_data(self, train: bool = True) -> Tuple[xgb.XGBModel, np.ndarray, np.ndarray]:
        # Get Model
        xgb_model = self.get_model().get_model()
        if not isinstance(xgb_model, xgb.XGBModel):
            raise ValueError("The model must be an XGBoost model")
        # Get Data
        X, y = self.get_data().export(XGBoostExportStrategy, train=train
        )
        return xgb_model, X, y

    @allow_no_addr_check
    def fit(self) -> P2PFLModel:
        """
        Fit the XGBoost sklearn model on training data for the configured number of epochs.
        """
        model, X_train, y_train = self.__get_xgb_model_data(train=True)
        # prepare callbacks
        xgb_callbacks = []
        for cb in self.callbacks:
            # each P2PFLCallback should expose an XGBoost-compatible callback
            if hasattr(cb, "to_xgb_callback"):
                xgb_callbacks.append(cb.to_xgb_callback())
        model.fit(
            X_train,
            y_train,
            verbose=True
        )
        self.get_model().set_contribution([self.addr], self.get_data().get_num_samples(train=True))
        # store callback info back to model
        self.add_callback_info_to_model()
        return self.get_model()

    @allow_no_addr_check
    def interrupt_fit(self) -> None:
        """
        Interrupting an in-progress XGBoost training is not supported via sklearn API.
        """
        # Placeholder: XGBoost sklearn does not support interrupt; could set a flag for custom callback
        raise NotImplementedError("Interrupting XGBoost sklearn fit is not supported")

    # def set_model(self, model: Union[P2PFLModel, list[np.ndarray], bytes]) -> None:
    #
    #     self.__model = model

    @allow_no_addr_check
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data, returning metrics.
        """
        model, X_test, y_test = self.__get_xgb_model_data(train=False)
        results: Dict[str, float] = {}
        try:
            preds = model.predict(X_test)
        except NotFittedError:
            return results

        # classification vs regression metric
        if np.issubdtype(y_test.dtype, np.integer):
            accuracy = float(np.mean(preds == y_test))
            results['accuracy'] = accuracy
        else:
            mse = float(np.mean((preds - y_test) ** 2))
            results['mse'] = mse
        return results

    @allow_no_addr_check
    def get_framework(self) -> str:
        """Return framework identifier."""
        return Framework.XGBOOST.value
