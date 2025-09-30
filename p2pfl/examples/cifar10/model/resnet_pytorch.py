#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
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

"""ResNet model on PyTorch Lightning for CIFAR10."""

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Metric
from torchvision import models

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class ResNetCIFAR10(L.LightningModule):
    """ResNet model for CIFAR10 classification."""

    def __init__(
        self,
        num_classes: int = 10,
        metric: type[Metric] = Accuracy,
        lr_rate: float = 0.001,
    ) -> None:
        """Initialize the ResNet model."""
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        self.lr_rate = lr_rate
        if num_classes == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=num_classes)

        # Use ResNet18 with modifications for CIFAR10
        model = models.resnet18(weights=None, num_classes=num_classes)
        # Adapt first conv layer for 32x32 images (instead of 224x224)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove maxpool which is designed for larger images
        model.maxpool = torch.nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Handle channel dimension if needed
        if len(x.shape) == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Training step of the model."""
        x = batch["image"].float()
        y = batch["label"]
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Validate step not implemented for this model."""
        raise NotImplementedError("Validation step not implemented")

    def test_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Test step for the model."""
        x = batch["image"].float()
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
        return loss


# Export P2PFL model
def model_build_fn(*args, **kwargs) -> LightningModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return LightningModel(ResNetCIFAR10(*args, **kwargs), compression=compression)
