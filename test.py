"""P2PFL communication tests."""
import numpy as np
import pytest
import torch
from datasets import DatasetDict, load_dataset
from pytorch_lightning import LightningDataModule

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.exceptions import ModelNotMatchingError
from p2pfl.learning.pytorch.lightning_learner import LightningLearner, LightningModel
from p2pfl.learning.pytorch.torch_dataset import PyTorchExportStrategy, TorchvisionDatasetFactory
from p2pfl.learning.pytorch.torch_model import MLP
def test_learner_train_test():
    """Test the training and testing of the learner."""
    # Dataset
    dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/mnist", split="train[:100]"),
                "test": load_dataset("p2pfl/mnist", split="test[:10]"),
            }
        )
    )

    # Model
    p2pfl_model = LightningModel(MLP())

    # Learner
    learner = LightningLearner(p2pfl_model, dataset)

    # Train
    learner.set_epochs(1)
    learner.fit()

    # Test
    learner.evaluate()

test_learner_train_test()