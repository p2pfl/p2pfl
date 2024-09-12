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
"""Learning tests."""

import numpy as np
import pytest
import torch
from datasets import DatasetDict, load_dataset
from pytorch_lightning import LightningDataModule

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.exceptions import ModelNotMatchingError
from p2pfl.learning.pytorch.lightning_learner import LightningModel
from p2pfl.learning.pytorch.torch_dataset import PyTorchExportStrategy, TorchvisionDatasetFactory
from p2pfl.learning.pytorch.torch_model import MLP

####
# Params & Model
####


def test_get_set_params():
    """Test setting and getting parameters."""
    p2pfl_model = LightningModel(MLP())
    # Modify parameters
    params = p2pfl_model.get_parameters()
    params_og = [layer.copy() for layer in p2pfl_model.get_parameters()]
    for i, layer in enumerate(params):
        params[i] = layer + 1
    # Set parameters
    p2pfl_model.set_parameters(params)
    # Check if the parameters are different (+1)
    for layer_og, layer_new in zip(params_og, p2pfl_model.get_parameters()):
        assert np.all(layer_og + 1 == layer_new)


def test_encoding():
    """Test encoding and decoding of parameters."""
    p2pfl_model1 = LightningModel(MLP())
    encoded_params = p2pfl_model1.encode_parameters()

    p2pfl_model2 = LightningModel(MLP())
    decoded_params = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    assert encoded_params == p2pfl_model1.encode_parameters()


def test_wrong_encoding():
    """Test wrong encoding of parameters."""
    p2pfl_model1 = LightningModel(MLP())
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
    p2pfl_model2 = LightningModel(mobile_net)
    decoded_params = p2pfl_model2.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


####
# Data
####


def test_torchvision_dataset_factory_mnist():
    """Test the TorchvisionDatasetFactory for MNIST."""
    train_dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)
    test_dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=False, download=True)

    assert isinstance(train_dataset, P2PFLDataset)
    assert isinstance(test_dataset, P2PFLDataset)

    assert train_dataset.get_num_samples() > 0
    assert test_dataset.get_num_samples() > 0

    # Check if the data is loaded correctly
    sample = train_dataset.get(0)
    assert "image" in sample
    assert "label" in sample

    # Check if the data is loaded correctly
    assert sample["image"].size == (28, 28)


def test_pytorch_export_strategy():
    """Test the PyTorchExportStrategy."""
    dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)

    export_strategy = PyTorchExportStrategy()
    data_module = dataset.export(export_strategy)

    assert isinstance(data_module, LightningDataModule)

    # Check if data
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    assert len(train_loader) > 0
    assert len(test_loader) > 0

    # Check if the data is loaded correctly
    sample = next(iter(train_loader))
    assert "image" in sample
    assert "label" in sample

    # Check if the data is loaded correctly
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].size() == (1, 28, 28)


def test_learner_train():
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

    # Create the model
    p2pfl_model = LightningModel(MLP(), dataset)

    # Export the dataset
    data_module = dataset.export(PyTorchExportStrategy(), batch_size=1)

    # Train
    p2pfl_model.train(data_module, max_epochs=1)

    # Test
    p2pfl_model.test(data_module)
