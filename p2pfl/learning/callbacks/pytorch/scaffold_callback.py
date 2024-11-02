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

"""Callback for SCAFFOLD operations."""

import copy
from typing import List, Optional

import torch
from lightning.pytorch.callbacks import Callback

from p2pfl.learning.callbacks.decorators import register
from p2pfl.learning.p2pfl_model import P2PFLModel


@register(callback_key='scaffold', framework='pytorch')
class SCAFFOLDCallback(Callback):
    """
    Callback for scaffold operations to use with PyTorch Lightning.

    At the beginning of the training, the callback needs to store the global model and the initial learning rate. Then, after optimization,


    Args:
        ...

    """

    def __init__(self):
        """Initialize the callback."""
        super().__init__()
        self.c_i: Optional[List[torch.Tensor]] = None
        self.c: Optional[List[torch.Tensor]] = None
        self.initial_model_params: Optional[List[torch.Tensor]] = None
        self.saved_lr: Optional[float] = None
        self.K: int = 0

    def on_train_start(self, pl_module: P2PFLModel):
        """
        Store the global model and the initial learning rate.

        Args:
            pl_module: The model.

        """
        if self.c_i is None:
            self.c_i = [torch.zeros_like(param) for param in pl_module.parameters()]

        if self.K == 0:
            self._get_global_c(pl_module)

        self.initial_model_params = copy.deepcopy(pl_module.get_parameters())
        self.K = 0

    def on_before_optimizer_step(self, pl_module: P2PFLModel):
        """
        Store the learning rate.

        Args:
            pl_module: The model.

        """
        self.saved_lr = pl_module.optimizer.param_groups[0]["lr"]

    def on_after_optimizer_step(self, pl_module: P2PFLModel):
        """
        Modify model by applying control variate adjustment.

        As the optimizer already computed y_i(g_i), we can compute the control variate adjustment as:
        y_i ← y_i + eta_l * c_i - eta_l * c

        Args:
            pl_module: The model.

        """
        eta_l = self.saved_lr
        for param, c_i_param, c_param in zip(pl_module.parameters(), self.c_i, self.c):
            param.data.add_(eta_l, c_i_param)
            param.data.sub_(eta_l, c_param)
        self.K += 1

    def on_train_end(self, pl_module: P2PFLModel):
        """
        Restore the global model.

        Args:
            pl_module: The model.

        """
        y_i = [param.clone().detach() for param in pl_module.parameters()]
        x_g = self.initial_model_params
        previous_c_i = [c.clone() for c in self.c_i]

        for idx, (c_i, x, y) in enumerate(zip(self.c_i, x_g, y_i)):
            adjustment = (x - y) / (self.K * self.saved_lr)
            self.c_i[idx] = c_i + adjustment

        # Compute delta y_i and delta c_i
        delta_y_i = [y - x for y,x in zip(y_i, x_g)]
        delta_c_i = [c_new - c_old for c_new, c_old in zip(self.c_i, previous_c_i)]

        delta_y_i_np = [dyi.detach().cpu().numpy() for dyi in delta_y_i] # to numpy for transmission
        delta_c_i_np = [dci.detach().cpu().numpy() for dci in delta_c_i]

        pl_module.model.add_info('delta_y_i', delta_y_i_np)
        pl_module.model.add_info('delta_c_i', delta_c_i_np)

    def _get_global_c(self, pl_module: P2PFLModel):
        """
        Get the global control variate from the aggregator.

        Args:
            pl_module: The model.

        """
        global_c_np = pl_module.model.get_info('global_c')
        if global_c_np is None:
            self.c = [torch.zeros_like(param) for param in pl_module.parameters()]
        self.c = [
                torch.from_numpy(c_np).to(pl_module.device)
                for c_np in global_c_np
            ]
