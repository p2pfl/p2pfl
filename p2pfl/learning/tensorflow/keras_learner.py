#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

import logging
import pickle
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import tensorflow as tf
from keras import mixed_precision
import os

from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.learner import NodeLearner
from p2pfl.management.logger import logger
from p2pfl.learning.pytorch.lightning_logger import FederatedLogger
from p2pfl.learning.LearnerStateDTO import LearnerStateDTO


class KerasLearner(NodeLearner):
    def __init__(
        self,
        model: tf.keras.Model,
        data: Tuple[tf.data.Dataset, tf.data.Dataset],
        self_addr: str,
        epochs: int,
        compile_params: Optional[Dict] = None,
    ):
        self.model = model
        self.data = data
        self.self_addr = self_addr
        
        self.epochs = epochs
        self.logger = FederatedLogger(self_addr)
        
        self.learner_state = LearnerStateDTO()
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        
        if compile_params is not None:
            self.compile_model(**compile_params)
    
    def set_epochs(self, epochs: int) -> None:
        self.epochs = epochs
    
    def set_model(self, model: tf.keras.Model) -> None:
        self.model = model
    
    def set_data(self, data: Tuple[tf.data.Dataset, tf.data.Dataset]) -> None:
        self.data = data
    
    def get_num_samples(self) -> Tuple[int, int]:
            return len(self.data[0]), len(self.data[1])
        
    ####
    # Sharing learning parameters
    ####
    
    def encode_parameters(self, params: Optional[LearnerStateDTO] = None) -> bytes:
        if params is None:
            params = self.get_parameters()
        return pickle.dumps(params)
    
    def decode_parameters(self, data: bytes) -> LearnerStateDTO:
        try:
            params_dict = pickle.loads(data)
            return params_dict
        except Exception as e:
            raise DecodingParamsError(
                f"Error in decoding parameters with Keras: {e}"
            )        
            
        
    def set_parameters(self, params: LearnerStateDTO) -> None:
        try:
            weights = params.get_weights()
            weights_list = [weights[layer] for layer in sorted(weights.keys())]
            self.model.set_weights(weights_list)
        except Exception as e:
            raise ModelNotMatchingError(f"Not matching models: {e}")

    def get_parameters(self) -> LearnerStateDTO:
        learner_state = LearnerStateDTO()
        weights_list = self.model.get_weights()
        weights_dict = {f"layer_{i}": weights_list[i] for i in range(len(weights_list))}
        learner_state.add_weights_dict(weights_dict)
        return learner_state
        
        
    ####
    # Training
    ####
    
    def set_epochs(self, epochs: int) -> None:
        self.epochs = epochs
        
    def compile_model(self, **kwargs) -> None:
        self.model.compile(**kwargs)
    
    def fit(self) -> None:
        try:
            if self.epochs > 0:
                self.model.fit(
                    self.data[0],
                    epochs=self.epochs,
                    validation_data=self.data[1],
                    verbose=0
                )
        except Exception as e:
            logger.error(f"Error in training with Keras: {e}")
        
    def interrupt_fit(self) -> None:
        logger.log_info(self.self_addr, "Interrupting training.")
        self.model.stop_training = True
        
    
    def evaluate(self) -> Dict[str, float]:
        try:
            if self.epochs > 0:
                results = self.model.evaluate(self.data[1], verbose=0)
                results_dict = dict(zip(self.model.metrics_names, results))
                for k, v in results_dict.items():
                    logger.log_metric(self.self_addr, k, v)
                return results_dict
            
        except Exception as e:
            logger.error(
                self.self_addr,
                f"Evaluation error. Something went wrong with Keras. {e}",
            )
            raise e