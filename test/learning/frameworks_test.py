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

import contextlib
from collections.abc import Generator

import numpy as np
import pytest
from datasets import DatasetDict, load_dataset  # type: ignore

from p2pfl.experiment import Experiment
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

with contextlib.suppress(ImportError):
    import tensorflow as tf

    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow
    from p2pfl.learning.frameworks.tensorflow.keras_dataset import KerasExportStrategy
    from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel


with contextlib.suppress(ImportError):
    import jax
    import jax.numpy as jnp

    from p2pfl.examples.mnist.model.mlp_flax import MLP as MLP_FLASK
    from p2pfl.learning.frameworks.flax.flax_dataset import FlaxExportStrategy
    from p2pfl.learning.frameworks.flax.flax_model import FlaxModel

with contextlib.suppress(ImportError):
    import torch
    from torch.utils.data import DataLoader

    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_torch
    from p2pfl.learning.frameworks.pytorch.lightning_dataset import PyTorchExportStrategy, TorchvisionDatasetFactory
    from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel

####
# Params & Model
####


def test_get_set_params_torch():
    """Test setting and getting parameters."""
    # Create the model
    p2pfl_model = model_build_fn_torch()
    # Modify parameters
    params = p2pfl_model.get_parameters()
    params_og = [layer.copy() for layer in p2pfl_model.get_parameters()]
    for i, layer in enumerate(params):
        params[i] = layer + 1
    # Set parameters
    p2pfl_model.set_parameters(params)
    # Check if the parameters are different (+1)
    for layer_og, layer_new in zip(params_og, p2pfl_model.get_parameters(), strict=False):
        assert np.all(layer_og + 1 == layer_new)


def test_get_set_params_tensorflow():
    """Test setting and getting parameters."""
    # Create the model
    p2pfl_model = model_build_fn_tensorflow()
    # Modify parameters
    params = p2pfl_model.get_parameters()
    params_og = [layer.copy() for layer in p2pfl_model.get_parameters()]
    for i, layer in enumerate(params):
        params[i] = layer + 1
    # Set parameters
    p2pfl_model.set_parameters(params)
    # Check if the parameters are different (+1)
    for layer_og, layer_new in zip(params_og, p2pfl_model.get_parameters(), strict=False):
        assert np.all(layer_og + 1 == layer_new)


def __test_get_set_params_flax():
    """Test setting and getting parameters."""
    # Create the model
    model = MLP_FLASK()
    seed = jax.random.PRNGKey(0)
    model_params = model.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model = FlaxModel(model, model_params)

    # Save internal flax-model repr
    _flax_params = p2pfl_model.model_params.copy()
    params = p2pfl_model.get_parameters()
    p2pfl_model.set_parameters(params)

    # Check that flax to numpy arrays transformation works
    for layer in _flax_params:
        for param in _flax_params[layer]:
            assert np.array_equal(
                _flax_params[layer][param], p2pfl_model.model_params[layer][param]
            ), f"Mismatch found in {layer} - {param}"

    # Modify parameters
    params = p2pfl_model.get_parameters()
    params_og = [layer.copy() for layer in p2pfl_model.get_parameters()]
    for i, layer in enumerate(params):
        params[i] = layer + 1
    # Set parameters
    p2pfl_model.set_parameters(params)
    # Check if the parameters are different (+1)
    for layer_og, layer_new in zip(params_og, p2pfl_model.get_parameters(), strict=False):
        assert np.all(layer_og + 1 == layer_new)


def test_encoding_torch():
    """Test encoding and decoding of parameters."""
    p2pfl_model1 = model_build_fn_torch()
    encoded_params = p2pfl_model1.encode_parameters()

    p2pfl_model2 = model_build_fn_torch()
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)
    p2pfl_model2.additional_info = additional_info

    assert encoded_params == p2pfl_model1.encode_parameters()
    assert additional_info == p2pfl_model1.additional_info


def test_encoding_tensorflow():
    """Test encoding and decoding of parameters."""
    p2pfl_model1 = model_build_fn_tensorflow()
    encoded_params = p2pfl_model1.encode_parameters()
    p2pfl_model2 = model_build_fn_tensorflow()
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    assert encoded_params == p2pfl_model1.encode_parameters()
    assert additional_info == p2pfl_model1.additional_info


def __test_encoding_flax():
    """Test encoding and decoding of parameters."""
    model1 = MLP_FLASK()
    seed = jax.random.PRNGKey(0)
    model_params = model1.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model1 = FlaxModel(model=model1, init_params=model_params)
    encoded_params = p2pfl_model1.encode_parameters()

    model2 = MLP_FLASK()
    seed = jax.random.PRNGKey(1)
    model_params = model2.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model2 = FlaxModel(model2, init_params=model_params)
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    for arr1, arr2 in zip(p2pfl_model1.get_parameters(), p2pfl_model2.get_parameters(), strict=False):
        assert np.array_equal(arr1, arr2)
        assert p2pfl_model1.additional_info == p2pfl_model2.additional_info


def test_wrong_encoding_torch():
    """Test wrong encoding of parameters."""
    p2pfl_model1 = model_build_fn_torch()
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
    p2pfl_model2 = LightningModel(mobile_net)
    decoded_params, _ = p2pfl_model2.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


def test_wrong_encoding_tensorflow():
    """Test wrong encoding of parameters."""
    p2pfl_model1 = model_build_fn_tensorflow()
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    p2pfl_model2 = KerasModel(mobile_net)
    decoded_params = p2pfl_model2.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


def __test_wrong_encoding_flax():
    """Test wrong encoding of parameters."""
    model1 = MLP_FLASK()
    seed = jax.random.PRNGKey(0)
    model_params1 = model1.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model1 = FlaxModel(model=model1, init_params=model_params1)
    encoded_params = p2pfl_model1.encode_parameters()
    model2 = MLP_FLASK()
    model2.hidden_sizes = (256, 128, 256, 128)
    model_params2 = model2.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model2 = FlaxModel(model=model2, init_params=model_params2)
    decoded_params = p2pfl_model1.decode_parameters(encoded_params)
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
    dataset.set_batch_size(1)

    export_strategy = PyTorchExportStrategy()
    train_dataloader = dataset.export(export_strategy, train_loader=True)
    test_dataloader = dataset.export(export_strategy, train_loader=False)

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
    dataset.set_batch_size(1)

    export_strategy = KerasExportStrategy()
    train_data = dataset.export(export_strategy, train_loader=True)
    test_data = dataset.export(export_strategy, train_loader=False)

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


def __test_flax_export_strategy():
    """Test the FlaxExportStrategy."""
    dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)
    dataset.set_batch_size(1)

    export_strategy = FlaxExportStrategy()
    train_data = dataset.export(export_strategy, train_loader=True)
    test_data = dataset.export(export_strategy, train_loader=False)

    assert isinstance(train_data, Generator)
    assert isinstance(test_data, Generator)

    # Check if data
    assert train_data is not None
    assert test_data is not None

    # Check if the data is loaded correctly
    x, y = next(iter(train_data))

    assert isinstance(x, jnp.ndarray)
    assert x.shape == (1, 28, 28)

    assert isinstance(y, jnp.ndarray)
    assert y.shape == (1,)


@pytest.mark.parametrize("build_model_fn", [model_build_fn_torch, model_build_fn_tensorflow])  # TODO: Flax
def test_learner_train(build_model_fn):
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
    p2pfl_model = build_model_fn()

    # Dont care about the seed
    Settings.general.seed = None

    node_name = "unknown-node"
    with contextlib.suppress(Exception):
        logger.register_node(node_name)
    experiment = Experiment(exp_name="test_experiment-torch", total_rounds=1)
    logger.experiment_started(node_name, experiment)
    # Learner
    learner = LearnerFactory.create_learner(p2pfl_model)()
    learner.set_addr(node_name)
    learner.set_model(p2pfl_model)
    learner.set_data(dataset)

    # Train
    learner.set_epochs(1)
    learner.fit()

    # Test
    learner.evaluate()
