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

from collections import OrderedDict
import pickle
from typing import Dict, Optional, Tuple
import torch
from pytorch_lightning import Trainer
from p2pfl.learning.learner import NodeLearner, ZeroEpochsError
from p2pfl.learning.pytorch.logger import FederatedLogger, LogsUnionType
from p2pfl.learning.exceptions import (
    DecodingParamsError,
    ModelNotMatchingError,
)
import logging
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl

###########################
#    LightningLearner     #
###########################


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
        self, model: pl.LightningModule, data: LightningDataModule, self_addr: str
    ):
        self.model = model
        self.data = data
        self.logger = FederatedLogger(self_addr)
        self.__trainer: Optional[Trainer] = None
        self.epochs = 1
        self.__self_addr = self_addr
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def set_model(self, model: pl.LightningModule) -> None:
        self.model = model

    def set_data(self, data: LightningDataModule) -> None:
        self.data = data

    ####
    # Model weights
    ####

    def encode_parameters(
        self, params: Optional[Dict[str, torch.Tensor]] = None
    ) -> bytes:
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

    def check_parameters(self, params: OrderedDict[str, torch.Tensor]) -> bool:
        # Check ordered dict keys
        if set(params.keys()) != set(self.get_parameters().keys()):
            return False
        # Check tensor shapes
        for key, value in params.items():
            if value.shape != self.get_parameters()[key].shape:
                return False
        return True

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
            logging.error(f"Something went wrong with pytorch lightning. {e}")

    def interrupt_fit(self) -> None:
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self) -> Tuple[float, float]:
        try:
            if self.epochs > 0:
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=False,
                    log_every_n_steps=0,
                    enable_checkpointing=False,
                )
                results = self.__trainer.test(self.model, self.data, verbose=False)
                loss = results[0]["test_loss"]
                metric = results[0]["test_metric"]
                self.__trainer = None
                self.log_validation_metrics(loss, metric)
                return loss, metric
            else:
                raise ZeroEpochsError("Zero epochs to evaluate.")
        except Exception as e:
            if not isinstance(e, ZeroEpochsError):
                logging.error(f"Something went wrong with pytorch lightning. {e}")
            raise e

    ####
    # Logging
    ####

    def create_new_exp(self) -> None:
        self.logger.create_new_exp()

    def log_validation_metrics(
        self,
        loss: float,
        metric: float,
        round: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if self.logger is not None:
            self.logger.log_round_metric("test_loss", loss, name=name, round=round)
            self.logger.log_round_metric("test_metric", metric, name=name, round=round)

    def get_logs(
        self, node: Optional[str] = None, exp: Optional[str] = None
    ) -> LogsUnionType:
        return self.logger.get_logs(node=node, exp=exp)

    def get_num_samples(self) -> Tuple[int, int]:
        """
        TODO: USE IT TO OBTAIN A MORE ACCURATE METRIC AGG
        """
        train_len = len(self.data.train_dataloader().dataset)  # type: ignore
        test_len = len(self.data.test_dataloader().dataset)  # type: ignore
        return (train_len, test_len)

    def finalize_round(self) -> None:
        if self.logger is not None:
            self.logger.finalize_round()
