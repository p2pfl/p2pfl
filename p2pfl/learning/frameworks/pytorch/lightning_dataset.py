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

"""PyTorch dataset integration."""

from collections.abc import Callable, Generator
from pathlib import Path

import torchvision.datasets as datasets
from datasets import Dataset, DatasetDict  # type: ignore
from torch.utils.data import DataLoader

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy, P2PFLDataset
from p2pfl.settings import Settings


class TorchvisionDatasetFactory:
    """Factory class for loading PyTorch Vision datasets in P2PFL."""

    @staticmethod
    def get_mnist(cache_dir: str | Path, train: bool = True, download: bool = True) -> P2PFLDataset:
        """
        Get the MNIST dataset from PytorchVision.

        Args:
            cache_dir: The directory where the dataset will be stored.
            train: Whether to get the training or test dataset.
            download: Whether to download the dataset.

        """
        mnist_train = datasets.MNIST(
            root=cache_dir,
            train=train,
            download=download,
        )

        mnist_test = datasets.MNIST(
            root=cache_dir,
            train=False,
            download=download,
        )

        def get_generator(mnist_ds) -> Callable[[], Generator]:
            def generate_examples():
                for image, label in mnist_ds:  # Unpack image and label from the tuple
                    yield {"image": image, "label": label}  # Yield a dictionary

            return generate_examples

        return P2PFLDataset(
            DatasetDict(
                {
                    "train": Dataset.from_generator(get_generator(mnist_train)),
                    "test": Dataset.from_generator(get_generator(mnist_test)),
                }
            )
        )


class PyTorchExportStrategy(DataExportStrategy):
    """Export strategy for PyTorch tensors."""

    @staticmethod
    def export(
        data: Dataset,
        batch_size: int | None = None,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """
        Export the data using the PyTorch strategy.

        Args:
            data: The data to export. Transforms should already be applied to the dataset via set_transform.
            batch_size: The batch size to use for the exported data.
            num_workers: The number of workers to use for the exported
            kwargs: Additional keyword arguments.

        Returns:
            The exported data.

        """
        if not batch_size:
            batch_size = Settings.training.DEFAULT_BATCH_SIZE

        # Check if data is already in torch format or has transforms applied
        # If format type is None, it means transforms might be handling conversion
        if hasattr(data, "format") and data.format["type"] is not None:
            # No format applied, likely transforms are handling conversion
            return DataLoader(data, batch_size=batch_size, num_workers=num_workers)
        else:
            # Apply torch format
            return DataLoader(data.with_format(type="torch", output_all_columns=True), batch_size=batch_size, num_workers=num_workers)
