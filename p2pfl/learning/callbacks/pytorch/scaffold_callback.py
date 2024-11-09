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

"""Callback for SCAFFOLD operations (PyTorch Lighting)."""

import copy
from typing import Any, Dict, List, Optional

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback

from p2pfl.learning.callbacks.decorators import register
from p2pfl.learning.framework_identifier import FrameworkIdentifier


@register(callback_key='scaffold', framework=FrameworkIdentifier.PYTORCH.value)
class SCAFFOLDCallback(Callback):
    """
    Callback for scaffold operations to use with PyTorch Lightning.

    At the beginning of the training, the callback needs to store the global model and the initial learning rate. Then, after optimization,
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.c_i: List[torch.Tensor] = []
        self.c: List[torch.Tensor] = []
        self.initial_model_params: List[torch.Tensor] = []
        self.saved_lr: Optional[float] = None
        self.K: int = 0
        self.additional_info: Dict[str, Any] = {}


    def on_train_start(self, trainer: pl.Trainer , pl_module: pl.LightningModule) -> None:
        """
        Store the global model and the initial learning rate.

        Args:
            trainer: The trainer
            pl_module: The model.

        """
        if not self.c_i:
            self.c_i = [torch.zeros_like(param) for param in self._get_parameters(pl_module)]

        if self.K == 0:
            if 'global_c' not in self.additional_info:
                self.additional_info['global_c'] = None

            c = self.additional_info['global_c']
            if c is None:
                self.c = [torch.zeros_like(param) for param in self._get_parameters(pl_module)]
            else:
                self.c = [
                    torch.from_numpy(c_np).to(pl_module.device)
                    for c_np in c
                ]

        self.initial_model_params = copy.deepcopy(self._get_parameters(pl_module))
        self.K = 0

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        """
        Store the learning rate.

        Args:
            trainer: The trainer
            pl_module: The model.
            batch: The batch.
            batch_idx: The batch index.

        """
        optimizer = trainer.optimizers[0]  # Access the first optimizer
        self.saved_lr = optimizer.param_groups[0]['lr']


    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        """
        Modify model by applying control variate adjustment.

        As the optimizer already computed y_i(g_i), we can compute the control variate adjustment as:
        y_i ← y_i + eta_l * c_i - eta_l * c

        Args:
            trainer: The trainer
            pl_module: The model.
            optimizer: The optimizer.

        """
        if self.saved_lr is None:
            raise AttributeError("Learning rate has not been set.")

        eta_l = self.saved_lr
        for param, c_i_param, c_param in zip(self._get_parameters(pl_module), self.c_i, self.c):
            if param.grad is not None:
                param.grad += eta_l * c_i_param - eta_l * c_param
        self.K += 1

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Restore the global model.

        Args:
            trainer: The trainer
            pl_module: The model.

        """
        if not self.initial_model_params or self.saved_lr is None:
            raise AttributeError("Necessary attributes are not initialized.")

        y_i = [param.clone().detach() for param in self._get_parameters(pl_module)]
        x_g = self.initial_model_params
        previous_c_i = [c.clone() for c in self.c_i]

        for idx, (c_i, x, y) in enumerate(zip(self.c_i, x_g, y_i)):
            adjustment = (x - y) / (self.K * self.saved_lr)
            self.c_i[idx] = c_i + adjustment

        # Compute delta y_i and delta c_i
        delta_y_i = [y - x for y, x in zip(y_i, x_g)]
        delta_c_i = [c_new - c_old for c_new, c_old in zip(self.c_i, previous_c_i)]

        delta_y_i_np = [dyi.detach().cpu().numpy() for dyi in delta_y_i]  # to numpy for transmission
        delta_c_i_np = [dci.detach().cpu().numpy() for dci in delta_c_i]

        self.additional_info['delta_y_i'] = delta_y_i_np
        self.additional_info['delta_c_i'] = delta_c_i_np

    def set_global_c(self, global_c_np: Optional[List[torch.Tensor]]) -> None:
        """
        Get the global control variate from the aggregator.

        Args:
            global_c_np : The global control variate.

        """
        self.additional_info['global_c'] = global_c_np

    def _get_parameters(self, pl_module: pl.LightningModule) -> List[torch.Tensor]:
        return [param.cpu() for _, param in pl_module.state_dict().items()]
