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

"""Convolutional Neural Network (for MNIST) with PyTorch Lightning."""

from typing import Dict, List, Optional, Union

import lightning as L
import numpy as np
import torch
from torchmetrics import Accuracy, Metric

from p2pfl.learning.exceptions import ModelNotMatchingError
from p2pfl.learning.p2pfl_model import P2PFLModel

#########################
#    LightningModel     #
#########################


class LightningModel(P2PFLModel):
    """
    P2PFL model abstraction for PyTorch Lightning.

    Args:
        model: The model to encapsulate.
        params: The parameters of the model.
        num_samples: The number of samples.
        contributors: The contributors of the model.
        aditional_info: Additional information.

    """

    def __init__(
        self,
        model: L.LightningModule,
        params: Optional[Union[List[np.ndarray], bytes]] = None,
        num_samples: Optional[int] = None,
        contributors: Optional[List[str]] = None,
        aditional_info: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(model, params, num_samples, contributors, aditional_info)

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        return [param.cpu().numpy() for _, param in self.model.state_dict().items()]

    def set_parameters(self, params: Union[List[np.ndarray], bytes]) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        # Decode parameters
        if isinstance(params, bytes):
            params = self.decode_parameters(params)

        # Build state_dict
        state_dict = self.model.state_dict()
        for (layer_name, param), new_param in zip(state_dict.items(), params):
            if param.shape != new_param.shape:
                raise ModelNotMatchingError("Not matching models")
            state_dict[layer_name] = torch.tensor(new_param)

        # Load
        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise ModelNotMatchingError("Not matching models") from e


####
# Example MLP
####


with torch.autograd.set_detect_anomaly(True):

    class MLP(L.LightningModule):
        """Multilayer Perceptron (MLP) with configurable parameters."""

        def __init__(
            self,
            input_size: int = 28 * 28,
            hidden_sizes: Optional[list[int]] = None,
            out_channels: int = 10,
            activation: str = "relu",
            metric: type[Metric] = Accuracy,
            lr_rate: float = 0.001,
            seed: Optional[int] = None,
        ) -> None:
            """Initialize the MLP."""
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = [256, 128]
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self.lr_rate = lr_rate
            if out_channels == 1:
                self.metric = metric(task="binary")
            else:
                self.metric = metric(task="multiclass", num_classes=out_channels)

            self.layers = torch.nn.ModuleList()

            # Input layer
            self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
            self.layers.append(self._get_activation(activation))

            # Hidden layers
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                self.layers.append(self._get_activation(activation))

            # Output layer
            self.layers.append(torch.nn.Linear(hidden_sizes[-1], out_channels))

        def _get_activation(self, activation_name: str) -> torch.nn.Module:
            if activation_name == "relu":
                return torch.nn.ReLU()
            elif activation_name == "sigmoid":
                return torch.nn.Sigmoid()
            elif activation_name == "tanh":
                return torch.nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation function: {activation_name}")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass of the MLP."""
            # Flatten the input
            batch_size, _, _ = x.size()
            x = x.view(batch_size, -1)

            for layer in self.layers:
                x = layer(x)

            x = torch.log_softmax(x, dim=1)
            return x

        def configure_optimizers(self) -> torch.optim.Optimizer:
            """Configure the optimizer."""
            return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

        def training_step(self, batch: Dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
            """Training step of the MLP."""
            x = batch["image"].float()
            y = batch["label"]
            loss = torch.nn.functional.cross_entropy(self(x), y)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch: Dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
            """Perform validation step for the MLP."""
            raise NotImplementedError("Validation step not implemented")

        def test_step(self, batch: Dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
            """Test step for the MLP."""
            x = batch["image"].float()
            y = batch["label"]
            logits = self(x)
            loss = torch.nn.functional.cross_entropy(self(x), y)
            out = torch.argmax(logits, dim=1)
            metric = self.metric(out, y)
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_metric", metric, prog_bar=True)
            return loss
