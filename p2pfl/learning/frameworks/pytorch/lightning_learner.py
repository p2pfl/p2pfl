#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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

import asyncio
import logging
import threading
import traceback
from typing import Any

import lightning as L
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.pytorch.lightning_dataset import PyTorchExportStrategy
from p2pfl.learning.frameworks.pytorch.lightning_logger import FederatedLogger
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed
from p2pfl.workflow.engine.experiment import Experiment

torch.set_num_threads(1)

_trainer_lock = threading.Lock()


def resolve_optimizer(model: L.LightningModule) -> torch.optim.Optimizer:
    """Resolve configure_optimizers() into a single Optimizer instance."""
    opt = model.configure_optimizers()
    if isinstance(opt, torch.optim.Optimizer):
        return opt
    if isinstance(opt, list | tuple):
        return opt[0]
    raise TypeError(f"Unsupported optimizer type from configure_optimizers: {type(opt)}")


class LightningLearner(Learner):
    """
    Learner with PyTorch Lightning.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        addr: The address of the learner.

    """

    def __init__(self, model: P2PFLModel | None = None, data: P2PFLDataset | None = None, aggregator: Aggregator | None = None) -> None:
        """Initialize the learner."""
        super().__init__(model, data, aggregator)
        self.__trainer: Trainer | None = None
        self.__eval_trainer: Trainer | None = None
        self.experiment: Experiment | None = None
        self._batch_iterator: Any = None
        self._optimizer: torch.optim.Optimizer | None = None

        # Start logging
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def set_address(self, address: str) -> str:
        """Set the address of the node."""
        self.logger = FederatedLogger(address)
        return super().set_address(address)

    def __get_pt_model_data(self, train: bool = True) -> tuple[L.LightningModule, DataLoader]:
        # Get Model
        pt_model = self.get_model().get_model()
        if not isinstance(pt_model, L.LightningModule):
            raise ValueError("The model must be a PyTorch Lightning model")
        # Get Data
        pt_data = self.get_data().export(PyTorchExportStrategy, train=train)
        if not isinstance(pt_data, DataLoader):
            raise ValueError("The data must be a PyTorch DataLoader")
        return pt_model, pt_data

    async def fit(self) -> P2PFLModel:
        """Fit the model."""
        try:
            if self.epochs > 0:
                set_seed(Settings.general.SEED, self.get_framework())
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=self.logger,  # type: ignore
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    callbacks=self.callbacks.copy(),  # type: ignore
                )
                pt_model, pt_data = self.__get_pt_model_data()
                def _do_fit() -> None:
                    with _trainer_lock:
                        self.__trainer.fit(pt_model, pt_data)  # type: ignore[union-attr]

                await asyncio.to_thread(_do_fit)
                self.__trainer = None

            # Set model contribution
            self.get_model().set_contribution([self.address], self.get_data().get_num_samples())

            # Set callback info
            self.add_callback_info_to_model()

            return self.get_model()

        except Exception as e:
            print(traceback.format_exc())
            logger.error(
                self.address,
                f"Fit error. Something went wrong with pytorch lightning. {e}",
            )
            raise e

    async def train_on_batch(self) -> P2PFLModel:
        """Train the model on the next batch manually."""
        set_seed(Settings.general.SEED, self.get_framework())
        pt_model = self.get_model().get_model()
        if not isinstance(pt_model, L.LightningModule):
            raise ValueError("The model must be a PyTorch Lightning model")

        if self._batch_iterator is None:
            pt_data = self.get_data().export(PyTorchExportStrategy, train=True)
            self._batch_iterator = iter(pt_data)

        try:
            batch = next(self._batch_iterator)
        except StopIteration:
            pt_data = self.get_data().export(PyTorchExportStrategy, train=True)
            self._batch_iterator = iter(pt_data)
            batch = next(self._batch_iterator)

        def _train_step() -> float:
            with _trainer_lock:
                if hasattr(pt_model, "train_on_batch"):
                    return pt_model.train_on_batch(batch)  # type: ignore[no-any-return, operator]
                pt_model.train()
                if self._optimizer is None:
                    self._optimizer = resolve_optimizer(pt_model)
                original_log = pt_model.log
                pt_model.log = lambda *a, **kw: None  # type: ignore[assignment]
                try:
                    loss_tensor: torch.Tensor = pt_model.training_step(batch, 0)  # type: ignore[assignment]
                finally:
                    pt_model.log = original_log  # type: ignore[assignment]
                loss_tensor.backward()  # type: ignore[no-untyped-call]
                self._optimizer.step()
                self._optimizer.zero_grad()
                return loss_tensor.item()

        loss = await asyncio.to_thread(_train_step)

        self.get_model().last_training_loss = loss
        self.get_model().set_contribution([self.address], self.get_data().get_num_samples())
        self.add_callback_info_to_model()
        return self.get_model()

    async def interrupt_fit(self) -> None:
        """Interrupt the fit."""
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    async def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        try:
            if self.epochs > 0:
                if self.__eval_trainer is None:
                    self.__eval_trainer = Trainer()
                eval_trainer = self.__eval_trainer
                pt_model, pt_data = self.__get_pt_model_data(train=False)
                def _do_test() -> dict[str, float]:
                    with _trainer_lock:
                        return dict(eval_trainer.test(pt_model, pt_data, verbose=True)[0])

                results = await asyncio.to_thread(_do_test)
                # Log metrics
                for k, v in results.items():
                    logger.log_metric(self.address, k, v)
                return dict(results)

            else:
                return {}
        except Exception as e:
            logger.error(
                self.address,
                f"Evaluation error. Something went wrong with pytorch lightning. {e}",
            )
            raise e

    def get_framework(self) -> str:
        """
        Retrieve the learner name.

        Returns:
            The name of the learner class.

        """
        return Framework.PYTORCH.value
