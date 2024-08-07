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

"""
FederatedDataModule for MNIST.

.. todo:: Create a P2PFL Dataset
"""

from math import floor
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# To Avoid Crashes with a lot of nodes
# import torch.multiprocessing
# type: ignore
# torch.multiprocessing.set_sharing_strategy("file_system")


class MnistFederatedDM(LightningDataModule):
    """
    LightningDataModule of partitioned MNIST. Its used to generate **IID** distribucions over MNIS. Toy Problem.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.

    """

    # Singleton
    mnist_train: Optional[MNIST] = None
    mnist_val: Optional[MNIST] = None

    def __init__(
        self,
        sub_id: int = 0,
        number_sub: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        val_percent: float = 0.1,
        iid: bool = True,
    ) -> None:
        """
        Initialize the MNIST Federated DataModule.

        Args:
            sub_id: Subset id of partition. (0 <= sub_id < number_sub)
            number_sub: Number of subsets.
            batch_size: The batch size of the data.
            num_workers: The number of workers of the data.
            val_percent: The percentage of the validation set.
            iid: If True, the data is IID, if False, the data is non-IID.

        """
        super().__init__()
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent

        # Singletons of MNIST train and test datasets
        if MnistFederatedDM.mnist_train is None:
            MnistFederatedDM.mnist_train = MNIST("", train=True, download=True, transform=transforms.ToTensor())
            if not iid:
                sorted_indexes = MnistFederatedDM.mnist_train.targets.sort()[1]
                MnistFederatedDM.mnist_train.targets = MnistFederatedDM.mnist_train.targets[sorted_indexes]
                MnistFederatedDM.mnist_train.data = MnistFederatedDM.mnist_train.data[sorted_indexes]
        if MnistFederatedDM.mnist_val is None:
            MnistFederatedDM.mnist_val = MNIST(
                "",
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            if not iid:
                sorted_indexes = MnistFederatedDM.mnist_val.targets.sort()[1]
                MnistFederatedDM.mnist_val.targets = MnistFederatedDM.mnist_val.targets[sorted_indexes]
                MnistFederatedDM.mnist_val.data = MnistFederatedDM.mnist_val.data[sorted_indexes]
        if self.sub_id + 1 > self.number_sub:
            raise ValueError(f"Not exist the subset {self.sub_id}")

        # Training / validation set
        trainset = MnistFederatedDM.mnist_train
        rows_by_sub = floor(len(trainset) / self.number_sub)
        tr_subset = Subset(
            trainset,
            range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub),
        )
        mnist_train, mnist_val = random_split(
            tr_subset,
            [
                round(len(tr_subset) * (1 - self.val_percent)),
                round(len(tr_subset) * self.val_percent),
            ],
        )

        # Test set
        testset = MnistFederatedDM.mnist_val
        rows_by_sub = floor(len(testset) / self.number_sub)
        te_subset = Subset(
            testset,
            range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub),
        )

        if len(testset) < self.number_sub:
            raise ValueError("Too much partitions")

        # DataLoaders
        self.train_loader = DataLoader(
            mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            te_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        # print(f"Train: {len(mnist_train)} Val:{len(mnist_val)} Test:{len(te_subset)}")

    def train_dataloader(self) -> DataLoader:
        """Get the training DataLoader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Get the validation DataLoader."""
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Get the test DataLoader."""
        return self.test_loader
