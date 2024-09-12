###########################
#    LightningDataset     #
###########################

"""
AQUI AL FINAL HAY DATASETS QUE NO SON COMPATIBLES O TIENEN LICENCIAS QUE NO PERMITEN SU RESUBIDA, POR LO QUE SE PROPORCIONA UTILIDAD PARA QUE CADA USUARIO LO CARGUE

"""

from pathlib import Path
from typing import Callable, Generator, Optional, Union

import torchvision.datasets as datasets
from datasets import Dataset, DatasetDict  # type: ignore
from pytorch_lightning import LightningDataModule

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy, P2PFLDataset


class TorchvisionDatasetFactory:
    """Factory class for loading PyTorch Vision datasets in P2PFL."""

    @staticmethod
    def get_mnist(cache_dir: Union[str, Path], train: bool = True, download: bool = True) -> P2PFLDataset:
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
        train_data: Dataset,
        test_data: Dataset,
        transforms: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ) -> LightningDataModule:
        """
        Export the data using the PyTorch strategy.

        Args:
            train_data: The training data to export.
            test_data: The test data to export.
            transforms: The transforms to apply to the data.
            batch_size: The batch size to use for the exported data.
            num_workers: The number of workers to use for the exported

        Returns:
            The exported data.

        """
        if transforms is not None:
            raise NotImplementedError("Transforms are not supported in this export strategy.")

        # Export to a PyTorch dataloader
        return LightningDataModule.from_datasets(
            train_dataset=train_data.with_format("torch", column=['image', 'label']),
            test_dataset=test_data.with_format("torch", column=['image', 'label']),
            batch_size=batch_size,
            num_workers=num_workers,
        )
