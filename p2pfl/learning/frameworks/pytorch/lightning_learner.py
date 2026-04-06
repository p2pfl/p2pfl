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

import logging
import traceback

import lightning as L
import numpy as np
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
from p2pfl.utils.check_ray import ray_installed
from p2pfl.utils.seed import set_seed
from p2pfl.workflow.engine.experiment import Experiment


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
        self.experiment: Experiment | None = None
        self._batch_dataloader = None
        self._batch_iter = None
        self._batch_optimizer = None

        # Start logging
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def set_model(self, model: P2PFLModel | list[np.ndarray] | bytes) -> None:
        """Set the model, resetting cached batch training state."""
        self._batch_dataloader = None
        self._batch_iter = None
        self._batch_optimizer = None
        super().set_model(model)

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
                if Settings.general.SEED is not None and not ray_installed():
                    raise ValueError(
                        "You must use Ray to set a seed with PyTorch Lightning. Not working on a same process. | pip install ray"
                    )
                set_seed(Settings.general.SEED, self.get_framework())
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=self.logger,  # type: ignore
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                    callbacks=self.callbacks.copy(),  # type: ignore
                )
                pt_model, pt_data = self.__get_pt_model_data()
                self.__trainer.fit(pt_model, pt_data)
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
        """
        Train the model on the next batch using raw PyTorch.

        Maintains a DataLoader iterator across calls. Each call processes
        one batch and returns the updated model. When the iterator is
        exhausted it wraps around.
        """
        try:
            set_seed(Settings.general.SEED, self.get_framework())
            pt_model = self.get_model().get_model()
            if not isinstance(pt_model, L.LightningModule):
                raise ValueError("The model must be a PyTorch Lightning model")

            # Initialize DataLoader, iterator, and optimizer on first call.
            # Check all three fields: set_model() resets them to None and may
            # run concurrently (via the actor event loop) when train_on_batch
            # is offloaded to a thread.
            if self._batch_dataloader is None or self._batch_optimizer is None:
                self._batch_dataloader = self.get_data().export(PyTorchExportStrategy, train=True)
                self._batch_iter = iter(self._batch_dataloader)
                optim_result = pt_model.configure_optimizers()
                # Handle dict return: {"optimizer": ..., "lr_scheduler": ...}
                if isinstance(optim_result, dict):
                    self._batch_optimizer = optim_result["optimizer"]
                elif isinstance(optim_result, tuple):
                    self._batch_optimizer = optim_result[0]
                    if isinstance(self._batch_optimizer, list):
                        self._batch_optimizer = self._batch_optimizer[0]
                elif isinstance(optim_result, list):
                    self._batch_optimizer = optim_result[0]
                else:
                    self._batch_optimizer = optim_result

            try:
                batch = next(self._batch_iter)
            except StopIteration:
                self._batch_iter = iter(self._batch_dataloader)
                batch = next(self._batch_iter)

            # Manual training step (bypass self.log since there's no Trainer)
            pt_model.train()
            self._batch_optimizer.zero_grad()
            original_log = pt_model.log
            pt_model.log = lambda *args, **kwargs: None
            try:
                loss = pt_model.training_step(batch, 0)
            finally:
                pt_model.log = original_log
            loss.backward()
            self._batch_optimizer.step()

            self.get_model().last_training_loss = float(loss.item())
            self.get_model().set_contribution([self.address], self.get_data().get_num_samples())
            self.add_callback_info_to_model()

            return self.get_model()

        except Exception as e:
            logger.error(
                self.address,
                f"Error in train_on_batch: {e}",
            )
            raise e

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
                self.__trainer = Trainer(
                    accelerator="auto",
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                )
                pt_model, pt_data = self.__get_pt_model_data(train=False)
                results = self.__trainer.test(pt_model, pt_data, verbose=False)[0]
                self.__trainer = None
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
