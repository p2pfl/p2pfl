#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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
"""Weight-based framework tests (PyTorch, TensorFlow, Flax)."""

import contextlib
from unittest.mock import MagicMock

import numpy as np
import pytest
from datasets import DatasetDict, load_dataset  # type: ignore

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.exceptions import ModelNotMatchingError
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.engine.experiment import Experiment

try:
    import tensorflow as tf

    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow
    from p2pfl.learning.frameworks.tensorflow.keras_dataset import KerasExportStrategy
    from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel
except ImportError:
    model_build_fn_tensorflow = pytest.param(None, marks=pytest.mark.skip(reason="TensorFlow not installed"))  # type: ignore[assignment]

try:
    import torch
    from torch.utils.data import DataLoader

    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_torch
    from p2pfl.learning.frameworks.pytorch.lightning_dataset import PyTorchExportStrategy, TorchvisionDatasetFactory
    from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
except ImportError:
    model_build_fn_torch = pytest.param(None, marks=pytest.mark.skip(reason="PyTorch not installed"))  # type: ignore[assignment]

###
# PyTorch Model Tests
###


def test_set_parameters_modifies_weights():
    """Adding 1 to each layer and setting it back produces params == original + 1."""
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


def test_encode_decode_preserves_parameters():
    """Encoding then decoding produces identical parameters and additional_info."""
    p2pfl_model1 = model_build_fn_torch()
    encoded_params = p2pfl_model1.encode_parameters()

    p2pfl_model2 = model_build_fn_torch()
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)
    p2pfl_model2.additional_info = additional_info

    assert encoded_params == p2pfl_model1.encode_parameters()
    assert additional_info == p2pfl_model1.additional_info


def test_decode_wrong_model_type_raises():
    """Decoding MLP params into MobileNet raises ModelNotMatchingError."""
    p2pfl_model1 = model_build_fn_torch()
    encoded_params = p2pfl_model1.encode_parameters()
    mobile_net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
    p2pfl_model2 = LightningModel(mobile_net)
    decoded_params, _ = p2pfl_model2.decode_parameters(encoded_params)
    # Check that raises
    with pytest.raises(ModelNotMatchingError):
        p2pfl_model2.set_parameters(decoded_params)


###
# TensorFlow Model Tests
###


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


def test_encoding_tensorflow():
    """Test encoding and decoding of parameters."""
    p2pfl_model1 = model_build_fn_tensorflow()
    encoded_params = p2pfl_model1.encode_parameters()
    p2pfl_model2 = model_build_fn_tensorflow()
    decoded_params, additional_info = p2pfl_model2.decode_parameters(encoded_params)
    p2pfl_model2.set_parameters(decoded_params)

    assert encoded_params == p2pfl_model1.encode_parameters()
    assert additional_info == p2pfl_model1.additional_info


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


###
# PyTorch Data Tests
###


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
    # datasets 4+ adds channel dimension: (B, C, H, W) = (1, 1, 28, 28)
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].size() == (1, 1, 28, 28)


###
# TensorFlow Data Tests
###


def test_tensorflow_export_strategy():
    """Test the KerasExportStrategy."""
    dataset = TorchvisionDatasetFactory.get_mnist(cache_dir=".", train=True, download=True)
    dataset.set_batch_size(1)

    train_data = dataset.export(KerasExportStrategy, train=True)
    test_data = dataset.export(KerasExportStrategy, train=False)

    assert isinstance(train_data, tuple) and len(train_data) == 2
    assert isinstance(test_data, tuple) and len(test_data) == 2

    x_train, y_train = train_data
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert len(x_train) > 0
    assert x_train.shape[1:] == (28, 28)


###
# Learner Training Tests
###


@pytest.mark.asyncio
@pytest.mark.parametrize("build_model_fn", [model_build_fn_torch, model_build_fn_tensorflow])  # TODO: Flax
async def test_learner_train(build_model_fn):
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
    Settings.general.SEED = None

    node_name = "unknown-node"
    with contextlib.suppress(Exception):
        logger.register_node(node_name)
    experiment = Experiment(exp_name="test_experiment-torch", total_rounds=1)
    logger.experiment_started(node_name, experiment)
    # Learner
    learner = LearnerFactory.create_learner(p2pfl_model)()
    learner.set_address(node_name)
    learner.set_model(p2pfl_model)
    learner.set_data(dataset)

    # Train
    learner.set_epochs(1)
    trained_model = await learner.fit()
    assert trained_model is not None
    assert len(trained_model.get_parameters()) > 0

    # Test
    metrics = await learner.evaluate()
    assert isinstance(metrics, dict)


###
# LearnerFactory Tests
###


def test_learner_factory_pytorch():
    """LearnerFactory returns LightningLearner for a PyTorch model."""
    from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

    model = model_build_fn_torch()
    learner_cls = LearnerFactory.create_learner(model)
    assert learner_cls is LightningLearner


def test_learner_factory_tensorflow():
    """LearnerFactory returns KerasLearner for a TensorFlow model."""
    from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner

    model = model_build_fn_tensorflow()
    learner_cls = LearnerFactory.create_learner(model)
    assert learner_cls is KerasLearner


def test_learner_factory_unsupported_framework():
    """LearnerFactory raises ValueError for an unknown framework string."""
    mock_model = MagicMock()
    mock_model.get_framework.return_value = "unknown_framework_xyz"
    with pytest.raises(ValueError, match="Unsupported framework"):
        LearnerFactory.create_learner(mock_model)


def test_learner_factory_flax():
    """LearnerFactory returns FlaxLearner for a Flax model."""
    pytest.importorskip("jax")
    from p2pfl.learning.frameworks.flax.flax_learner import FlaxLearner

    mock_model = MagicMock()
    mock_model.get_framework.return_value = Framework.FLAX.value
    learner_cls = LearnerFactory.create_learner(mock_model)
    assert learner_cls is FlaxLearner


def test_learner_factory_xgboost():
    """LearnerFactory returns XGBoostLearner for an XGBoost model."""
    pytest.importorskip("xgboost")
    from p2pfl.learning.frameworks.xgboost.xgboost_learner import XGBoostLearner

    mock_model = MagicMock()
    mock_model.get_framework.return_value = Framework.XGBOOST.value
    learner_cls = LearnerFactory.create_learner(mock_model)
    assert learner_cls is XGBoostLearner


###
# Learner Base Tests
###


class TestLearnerBase:
    """Tests for base Learner error paths and accessor methods."""

    def _make_learner(self):
        """Create a LightningLearner with no model/data (for testing accessors)."""
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        learner = LightningLearner()
        return learner

    def test_get_model_without_set_raises(self):
        """get_model raises ValueError when model is not set."""
        learner = self._make_learner()
        with pytest.raises(ValueError, match="Model not initialized"):
            learner.get_model()

    def test_get_data_without_set_raises(self):
        """get_data raises ValueError when data is not set."""
        learner = self._make_learner()
        with pytest.raises(ValueError, match="Data not initialized"):
            learner.get_data()

    def test_steps_per_epoch_accessors(self):
        """get/set_steps_per_epoch round-trips correctly."""
        learner = self._make_learner()
        assert learner.get_steps_per_epoch() is None
        learner.set_steps_per_epoch(100)
        assert learner.get_steps_per_epoch() == 100

    def test_init_with_aggregator(self):
        """Learner.__init__ with an aggregator calls indicate_aggregator."""
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        mock_agg = MagicMock()
        mock_agg.get_required_callbacks.return_value = []
        learner = LightningLearner()
        learner.set_address("test-addr")
        learner.indicate_aggregator(mock_agg)
        # Should not raise and callbacks may be empty since aggregator has no required callbacks
        assert isinstance(learner.callbacks, list)
