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

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, Trainer

from p2pfl.learning.exceptions import (
    DecodingParamsError,
    ModelNotMatchingError,
)
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.pytorch.lightning_logger import FederatedLogger
from p2pfl.management.logger import logger

torch.set_num_threads(1)

###########################
#    LightningLearner     #
###########################

"""
---------->  Add information of the node that owns the learner.  <----------
"""

"""
        decoded_model = self.learner.decode_parameters(weights)
        if self.learner.check_parameters(decoded_model):
            models_added = self.aggregator.add_model(
                decoded_model,
                list(contributors),
                weight,
            )
            if models_added != []:
                # Communicate Aggregation
                self._neighbors.broadcast_msg(
                    self._neighbors.build_msg(
                        LearningNodeMessages.MODELS_AGGREGATED,
                        models_added,
                    )
                )
        else:
            raise ModelNotMatchingError(
                "Not matching models"
            )  # esta excepción mejor tenerla en add_model (así nos ahorramos in if raise que no aporta robustez en el método que importa)

"""


class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        log_name: Name of the log.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(
        self,
        model: pl.LightningModule,
        data: LightningDataModule,
        self_addr: str,
        epochs: int,
    ):
        self.model = model
        self.data = data
        self.__trainer: Optional[Trainer] = None
        self.epochs = epochs
        self.__self_addr = self_addr
        # Start logging
        self.logger = FederatedLogger(self_addr)
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def set_model(self, model: pl.LightningModule) -> None:
        self.model = model

    def set_data(self, data: LightningDataModule) -> None:
        self.data = data

    def get_num_samples(self) -> Tuple[int, int]:
        """
        TODO: USE IT TO OBTAIN A MORE ACCURATE METRIC AGG
        """
        train_len = len(self.data.train_dataloader().dataset)  # type: ignore
        test_len = len(self.data.test_dataloader().dataset)  # type: ignore
        return (train_len, test_len)

    ####
    # Model weights
    ####

    def encode_parameters(self, params: Optional[Dict[str, torch.Tensor]] = None) -> bytes:
        if params is None:
            params = self.get_parameters()
        array = [val.cpu().numpy() for _, val in params.items()]
        return pickle.dumps(array)

    def decode_parameters(self, data: bytes) -> Dict[str, torch.Tensor]:
        try:
            params_dict = zip(self.get_parameters().keys(), pickle.loads(data))
            return OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        except Exception:
            raise DecodingParamsError("Error decoding parameters")

    """es esto realmente necesario?
    def check_parameters(self, params: OrderedDict[str, torch.Tensor]) -> bool:
        # Check ordered dict keys
        if set(params.keys()) != set(self.get_parameters().keys()):
            return False
        # Check tensor shapes
        for key, value in params.items():
            if value.shape != self.get_parameters()[key].shape:
                return False
        return True
    """

    def set_parameters(self, params: OrderedDict[str, torch.Tensor]) -> None:
        try:
            self.model.load_state_dict(params)
        except Exception:
            raise ModelNotMatchingError("Not matching models")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    ####
    # Training
    ####

    def set_epochs(self, epochs: int) -> None:
        self.epochs = epochs

    def fit(self) -> None:
        try:
            if self.epochs > 0:
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=self.logger,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                )
                self.__trainer.fit(self.model, self.data)
                self.__trainer = None
        except Exception as e:
            logger.error(
                self.__self_addr,
                f"Fit error. Something went wrong with pytorch lightning. {e}",
            )
            raise e

    def interrupt_fit(self) -> None:
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self) -> Dict[str, float]:
        try:
            if self.epochs > 0:
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=False,
                    log_every_n_steps=0,
                    enable_checkpointing=False,
                )
                results = self.__trainer.test(self.model, self.data, verbose=False)[0]
                self.__trainer = None
                # Log metrics
                for k, v in results.items():
                    logger.log_metric(self.__self_addr, k, v)
                return results
            else:
                return {}
        except Exception as e:
            logger.error(
                self.__self_addr,
                f"Evaluation error. Something went wrong with pytorch lightning. {e}",
            )
            raise e
