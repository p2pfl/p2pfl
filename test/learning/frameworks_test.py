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
"""ML Framework tests."""

import numpy as np
import pytest
import tensorflow as tf
import torch
from datasets import DatasetDict, load_dataset  # type: ignore
from torch.utils.data import DataLoader

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.exceptions import ModelNotMatchingError
from p2pfl.learning.pytorch.lightning_dataset import PyTorchExportStrategy, TorchvisionDatasetFactory
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.pytorch.lightning_model import MLP as MLP_PT
from p2pfl.learning.pytorch.lightning_model import LightningModel
from p2pfl.learning.tensorflow.keras_dataset import KerasExportStrategy
from p2pfl.learning.tensorflow.keras_learner import KerasLearner
from p2pfl.learning.tensorflow.keras_model import MLP as MLP_KERAS
from p2pfl.learning.tensorflow.keras_model import KerasModel

####
# Params & Model
####


def test_get_set_params_torch():
    """Test setting and getting parameters."""
    # Create the model
    p2pfl_model = LightningModel(MLP_PT())
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


def test_get_set_params_tensorflow():
    """Test setting and getting parameters."""
    # Create the model
    model = MLP_KERAS()
    model(tf.zeros((1, 28, 28, 1)))
    p2pfl_model = KerasModel(model)
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


def test_encoding_torch():
    """Test encoding and decoding of parameters."""
    p2pfl_model1 = LightningModel(MLP_PT())
    encoded_params = p2pfl_model1.encode_parameters()

    p2pfl_model2 = LightningModel(MLP_PT())
    decoded_params = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    assert encoded_params == p2pfl_model1.encode_parameters()


def test_encoding_tensorflow():
    """Test encoding and decoding of parameters."""
    model = MLP_KERAS()
    model(tf.zeros((1, 28, 28, 1)))
    p2pfl_model1 = KerasModel(model)
    encoded_params = p2pfl_model1.encode_parameters()

    model = MLP_KERAS()
    model(tf.zeros((1, 28, 28, 1)))
    p2pfl_model2 = KerasModel(model)
    decoded_params = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    assert encoded_params == p2pfl_model1.encode_parameters()


def test_wrong_encoding_torch():
    """Test wrong encoding of parameters."""
    p2pfl_model1 = LightningModel(MLP_PT())
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
    p2pfl_model2 = LightningModel(mobile_net)
    decoded_params = p2pfl_model2.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


def test_wrong_encoding_tensorflow():
    """Test wrong encoding of parameters."""
    model = MLP_KERAS()
    model(tf.zeros((1, 28, 28, 1)))
    p2pfl_model1 = KerasModel(model)
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    p2pfl_model2 = KerasModel(mobile_net)
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
    train_dataloader = dataset.export(export_strategy, train_loader=True, batch_size=1)
    test_dataloader = dataset.export(export_strategy, train_loader=False, batch_size=1)

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)

    # Check if data
    assert len(train_dataloader) > 0
    assert len(test_dataloader) > 0

    # Check if the data is loaded correctly
    sample = next(iter(train_dataloader))
    assert "image" in sample
    assert "label" in sample

    # Check if the data is loaded correctly
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].size() == (1, 28, 28)


def test_tensorflow_export_strategy():
    """Test the PyTorchExportStrategy."""
    dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)

    export_strategy = KerasExportStrategy()
    train_data = dataset.export(export_strategy, train_loader=True, batch_size=1)
    test_data = dataset.export(export_strategy, train_loader=False, batch_size=1)

    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(test_data, tf.data.Dataset)

    # Check if data
    assert len(train_data) > 0
    assert len(test_data) > 0

    # Check if the data is loaded correctly
    sample = next(iter(train_data))
    assert isinstance(sample, tuple)

    # Check if the data is loaded correctly
    assert isinstance(sample[0], tf.Tensor)
    assert sample[0].shape == (1, 28, 28)


def test_learner_train_torch():
    """Test the training and testing of the learner."""
    # Dataset
    dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/MNIST", split="train[:100]"),
                "test": load_dataset("p2pfl/MNIST", split="test[:10]"),
            }
        )
    )

    # Create the model
    p2pfl_model = LightningModel(MLP_PT())

    # Learner
    learner = LightningLearner(p2pfl_model, dataset)

    # Train
    learner.set_epochs(1)
    learner.fit()

    # Test
    learner.evaluate()


def test_learner_train_tensorflow():
    """Test the training and testing of the learner."""
    # Dataset
    dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/MNIST", split="train[:100]"),
                "test": load_dataset("p2pfl/MNIST", split="test[:10]"),
            }
        )
    )

    # Create the model
    model = MLP_KERAS()
    model(tf.zeros((1, 28, 28, 1)))
    p2pfl_model = KerasModel(model)

    # Learner
    learner = KerasLearner(p2pfl_model, dataset)

    # Train
    learner.set_epochs(1)
    learner.fit()

    # Test
    learner.evaluate()
