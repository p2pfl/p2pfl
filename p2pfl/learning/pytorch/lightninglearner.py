#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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
import torch
from pytorch_lightning import Trainer
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.pytorch.logger import FederatedLogger
import logging

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

    def __init__(self, model, data, self_addr):
        self.model = model
        self.data = data
        self.logger = FederatedLogger(self_addr)
        self.__trainer = None
        self.epochs = 1
        self.__self_addr = self_addr
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data

    ####
    # Model weights
    ####

    def encode_parameters(self, params=None):
        if params is None:
            params = self.model.state_dict()
        array = [val.cpu().numpy() for _, val in params.items()]
        return pickle.dumps(array)

    def decode_parameters(self, data):
        try:
            params_dict = zip(self.model.state_dict().keys(), pickle.loads(data))
            return OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        except:
            raise DecodingParamsError("Error decoding parameters")

    def check_parameters(self, params):
        # Check ordered dict keys
        if set(params.keys()) != set(self.model.state_dict().keys()):
            return False
        # Check tensor shapes
        for key, value in params.items():
            if value.shape != self.model.state_dict()[key].shape:
                return False
        return True

    def set_parameters(self, params):
        try:
            self.model.load_state_dict(params)
        except:
            raise ModelNotMatchingError("Not matching models")

    def get_parameters(self):
        return self.model.state_dict()

    ####
    # Training
    ####

    def set_epochs(self, epochs):
        self.epochs = epochs

    def fit(self):
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

    def interrupt_fit(self):
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self):
        try:
            if self.epochs > 0:
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=None,
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
                return None
        except Exception as e:
            logging.error(f"Something went wrong with pytorch lightning. {e}")
            return None

    ####
    # Logging
    ####

    def create_new_exp(self):
        self.logger.create_new_exp()

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        if self.logger is not None:
            self.logger.log_round_metric("test_loss", loss, name=name, round=round)
            self.logger.log_round_metric("test_metric", metric, name=name, round=round)

    def get_logs(self, node=None, exp=None):
        return self.logger.get_logs(node=node, exp=exp)

    def get_num_samples(self):
        """
        TODO: USE IT TO OBTAIN A MORE ACCURATE METRIC AGG
        """
        return (
            len(self.data.train_dataloader().dataset),
            len(self.data.test_dataloader().dataset),
        )

    def finalize_round(self):
        if self.logger is not None:
            self.logger.finalize_round()
