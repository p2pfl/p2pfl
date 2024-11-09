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

"""Lightning Learner for P2PFL."""

import logging
import traceback
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from p2pfl.experiment import Experiment
from p2pfl.learning.callbacks.pytorch.lightning_logger import FederatedLogger
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.framework_identifier import FrameworkIdentifier
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.pytorch.lightning_dataset import PyTorchExportStrategy
from p2pfl.management.logger import logger

torch.set_num_threads(1)


class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        self_addr: The address of the learner.

    """

    def __init__(
        self,
        model: P2PFLModel,
        data: P2PFLDataset,
        self_addr: str = "unknown-node",
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """Initialize the learner."""
        if callbacks is None:
            callbacks = []
        super().__init__(model, data, self_addr, callbacks)
        self.__trainer: Optional[Trainer] = None
        self.experiment: Optional[Experiment] = None

        # append federated logger
        # TODO: Federated logger is not a Lighting callback
        # self.callbacks.append(FederatedLogger(self_addr)) if callbacks is not None else [FederatedLogger(self_addr)]
        # Start logging
        self.logger = FederatedLogger(self_addr)
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def __get_pt_model_data(self, train: bool = True) -> Tuple[L.LightningModule, DataLoader]:
        # Get Model
        pt_model = self.model.get_model()
        if not isinstance(pt_model, L.LightningModule):
            raise ValueError("The model must be a PyTorch Lightning model")
        # Get Data
        pt_data = self.data.export(PyTorchExportStrategy, train=train)
        if not isinstance(pt_data, DataLoader):
            raise ValueError("The data must be a PyTorch DataLoader")
        return pt_model, pt_data

    def fit(self) -> P2PFLModel:
        """Fit the model."""
        try:
            if self.epochs > 0:
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=self.logger,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    callbacks=self.callbacks,
                )
                pt_model, pt_data = self.__get_pt_model_data()

                self.set_callbacks_additional_info(self.callbacks)
                self.__trainer.fit(pt_model, pt_data)
                self.get_callbacks_additional_info(self.callbacks)

                self.__trainer = None
            # Set model contribution
            self.model.set_contribution([self._self_addr], self.data.get_num_samples())

            return self.model

        except Exception as e:
            print(traceback.format_exc())
            logger.error(
                self._self_addr,
                f"Fit error. Something went wrong with pytorch lightning. {e}",
            )
            raise e

    def interrupt_fit(self) -> None:
        """Interrupt the fit."""
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        try:
            if self.epochs > 0:
                self.__trainer = Trainer()
                pt_model, pt_data = self.__get_pt_model_data(train=False)
                results = self.__trainer.test(pt_model, pt_data, verbose=True)[0]
                self.__trainer = None
                # Log metrics
                for k, v in results.items():
                    logger.log_metric(self._self_addr, k, v)
                return dict(results)

            else:
                return {}
        except Exception as e:
            logger.error(
                self._self_addr,
                f"Evaluation error. Something went wrong with pytorch lightning. {e}",
            )
            raise e

    @staticmethod
    def get_framework() -> str:
        """
        Retrieve the framework name used by the learner.

        Returns:
            str: The framework name ('pytorch').

        """
        return FrameworkIdentifier.PYTORCH.value
