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

"""FedProx Callback for PyTorch Lightning."""

from typing import TYPE_CHECKING

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from p2pfl.learning.frameworks.callback import P2PFLCallback

if TYPE_CHECKING:
    from torch.optim import Optimizer


class FedProxCallback(Callback, P2PFLCallback):
    """
    PyTorch Lightning Callback to implement the FedProx algorithm.

    This callback modifies the gradients before the optimizer step by adding
    the gradient of the proximal term: mu * (w - w_t).
    """

    def __init__(self) -> None:
        """Initialize the FedProxCallback."""
        super().__init__()
        self.proximal_mu: float | None = None
        self.initial_params: list[torch.Tensor] | None = None

    @staticmethod
    def get_name() -> str:
        """Return the name of the callback."""
        return "fedprox"

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Store the initial model parameters and proximal coefficient.

        Args:
            trainer: The trainer
            pl_module: The model.

        """
        # Get proximal_mu from additional_info if available
        if hasattr(pl_module, "p2pfl_model") and hasattr(pl_module.p2pfl_model, "additional_info"):
            self.proximal_mu = pl_module.p2pfl_model.additional_info.get("proximal_mu")
            initial_params_np = pl_module.p2pfl_model.additional_info.get("initial_round_params")

            if initial_params_np is not None:
                # Convert numpy arrays to tensors on the same device as the model
                self.initial_params = [
                    torch.tensor(param_np, dtype=torch.float32, device=pl_module.device) for param_np in initial_params_np
                ]

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: "Optimizer") -> None:
        """
        Add the proximal gradient term: mu * (w - w_global).

        Args:
            trainer: The trainer
            pl_module: The model
            optimizer: The optimizer

        """
        if self.proximal_mu is not None and self.proximal_mu > 0 and self.initial_params is not None:
            model_params = list(pl_module.parameters())

            if len(model_params) != len(self.initial_params):
                # Log warning but don't crash
                print(
                    f"FedProxCallback: Mismatch between model parameters ({len(model_params)}) "
                    f"and initial_params ({len(self.initial_params)}). Skipping proximal term."
                )
                return

            for model_param, initial_param in zip(model_params, self.initial_params, strict=False):
                if model_param.grad is not None:
                    if model_param.data.shape != initial_param.shape:
                        print(
                            f"FedProxCallback: Shape mismatch. Model: {model_param.data.shape}, Initial: {initial_param.shape}. Skipping."
                        )
                        continue

                    # Add proximal gradient: mu * (w - w_global)
                    proximal_grad = self.proximal_mu * (model_param.data - initial_param)
                    model_param.grad.add_(proximal_grad)
