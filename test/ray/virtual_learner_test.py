#
# This file is part of the p2pfl distribution
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
"""
Virtual node tests.

These tests mock WorkerPool and ray to avoid actually running Ray,
which would fail with MagicMock learners (not serializable).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import ray  # noqa: F401

    RAY_INSTALLED = True
except ImportError:
    RAY_INSTALLED = False

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel

if RAY_INSTALLED:
    from p2pfl.learning.frameworks.ray.virtual_learner import VirtualNodeLearner

# Skip all tests in this module if Ray is not installed
pytestmark = pytest.mark.skipif(not RAY_INSTALLED, reason="Ray is not installed")


def create_mock_learner(address: str = "") -> MagicMock:
    """Create a mock Learner with the address attribute set."""
    learner = MagicMock(spec=Learner)
    learner.address = address
    return learner


@pytest.fixture
def mock_worker_pool():
    """Mock WorkerPool and ray to avoid Ray initialization."""
    mock_worker = MagicMock()
    mock_pool = MagicMock()
    mock_pool.assign_worker.return_value = mock_worker

    with (
        patch(
            "p2pfl.learning.frameworks.ray.virtual_learner.WorkerPool",
            return_value=mock_pool,
        ),
        patch("p2pfl.learning.frameworks.ray.virtual_learner.ray") as mock_ray,
    ):
        mock_ray.get.side_effect = lambda x: x
        mock_ray.put.side_effect = lambda x: x
        yield {"worker": mock_worker, "pool": mock_pool, "ray": mock_ray}


def test_initialization_registers_with_worker(mock_worker_pool):
    """Test that VirtualNodeLearner registers the learner with the assigned worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    mock_worker_pool["pool"].assign_worker.assert_called_once()
    mock_worker_pool["worker"].register_node.remote.assert_called_once_with(
        "node-1", learner
    )
    assert vl.address == "node-1"


def test_initialization_uses_id_when_no_address(mock_worker_pool):
    """Test that VirtualNodeLearner uses str(id(learner)) when address is empty."""
    learner = create_mock_learner(address="")
    vl = VirtualNodeLearner(learner)

    call_args = mock_worker_pool["worker"].register_node.remote.call_args
    node_key = call_args[0][0]
    assert node_key == str(id(learner))


def test_set_address_rekeys_on_worker(mock_worker_pool):
    """Test that set_address calls rekey_node and set_address on worker."""
    learner = create_mock_learner(address="old-key")
    vl = VirtualNodeLearner(learner)

    vl.set_address("new-key")

    mock_worker_pool["worker"].rekey_node.remote.assert_called_once_with(
        "old-key", "new-key"
    )
    mock_worker_pool["worker"].set_address.remote.assert_called_once_with(
        "new-key", "new-key"
    )
    assert vl.address == "new-key"


def test_set_model_delegates_to_worker(mock_worker_pool):
    """Test that set_model uses ray.put and delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    model = MagicMock(spec=P2PFLModel)

    # ray.put side_effect is identity, so ray.put(model) returns model
    vl.set_model(model)

    mock_worker_pool["ray"].put.assert_called_with(model)
    mock_worker_pool["worker"].set_model.remote.assert_called_once_with("node-1", model)


def test_get_model_delegates_to_worker(mock_worker_pool):
    """Test that get_model calls worker.get_model and resolves the object ref."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_worker_pool["worker"].get_model.remote.return_value = mock_model_ref
    mock_worker_pool["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = vl.get_model()

    mock_worker_pool["worker"].get_model.remote.assert_called_once_with("node-1")
    assert result == model


def test_set_data_delegates_to_worker(mock_worker_pool):
    """Test that set_data uses ray.put and delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    data = MagicMock(spec=P2PFLDataset)

    vl.set_data(data)

    mock_worker_pool["ray"].put.assert_called_with(data)
    mock_worker_pool["worker"].set_data.remote.assert_called_once_with("node-1", data)


def test_get_data_delegates_to_worker(mock_worker_pool):
    """Test that get_data delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    data = MagicMock(spec=P2PFLDataset)

    mock_worker_pool["worker"].get_data.remote.return_value = data

    result = vl.get_data()

    mock_worker_pool["worker"].get_data.remote.assert_called_once_with("node-1")
    assert result == data


def test_set_epochs_delegates(mock_worker_pool):
    """Test that set_epochs delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    vl.set_epochs(10)

    mock_worker_pool["worker"].set_epochs.remote.assert_called_once_with("node-1", 10)


def test_get_epochs_delegates(mock_worker_pool):
    """Test that get_epochs delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    mock_worker_pool["worker"].get_epochs.remote.return_value = 5

    result = vl.get_epochs()

    mock_worker_pool["worker"].get_epochs.remote.assert_called_once_with("node-1")
    assert result == 5


def test_set_steps_per_epoch_delegates(mock_worker_pool):
    """Test that set_steps_per_epoch delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    vl.set_steps_per_epoch(100)

    mock_worker_pool["worker"].set_steps_per_epoch.remote.assert_called_once_with("node-1", 100)


def test_get_steps_per_epoch_delegates(mock_worker_pool):
    """Test that get_steps_per_epoch delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    mock_worker_pool["worker"].get_steps_per_epoch.remote.return_value = 50

    result = vl.get_steps_per_epoch()

    mock_worker_pool["worker"].get_steps_per_epoch.remote.assert_called_once_with("node-1")
    assert result == 50


def test_configure_delegates(mock_worker_pool):
    """Test that configure delegates to worker with kwargs."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    vl.configure(epochs=10, steps_per_epoch=100)

    mock_worker_pool["worker"].configure.remote.assert_called_once_with(
        "node-1", epochs=10, steps_per_epoch=100
    )


def test_indicate_aggregator_delegates(mock_worker_pool):
    """Test that indicate_aggregator delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    aggregator = MagicMock()

    vl.indicate_aggregator(aggregator)

    mock_worker_pool["worker"].indicate_aggregator.remote.assert_called_once_with("node-1", aggregator)


def test_update_callbacks_with_model_info_delegates(mock_worker_pool):
    """Test that update_callbacks_with_model_info delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    vl.update_callbacks_with_model_info()

    mock_worker_pool["worker"].update_callbacks_with_model_info.remote.assert_called_once_with("node-1")


def test_add_callback_info_to_model_delegates(mock_worker_pool):
    """Test that add_callback_info_to_model delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    vl.add_callback_info_to_model()

    mock_worker_pool["worker"].add_callback_info_to_model.remote.assert_called_once_with("node-1")


def test_get_framework_delegates(mock_worker_pool):
    """Test that get_framework delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    mock_worker_pool["worker"].get_framework.remote.return_value = "pytorch"

    result = vl.get_framework()

    mock_worker_pool["worker"].get_framework.remote.assert_called_once_with("node-1")
    assert result == "pytorch"


@pytest.mark.asyncio
async def test_fit_delegates_to_worker(mock_worker_pool):
    """Test that fit delegates to worker and resolves the model ref."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_worker_pool["worker"].fit.remote = AsyncMock(return_value=mock_model_ref)
    mock_worker_pool["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = await vl.fit()

    mock_worker_pool["worker"].fit.remote.assert_called_once_with("node-1")
    assert result == model


@pytest.mark.asyncio
async def test_evaluate_delegates_to_worker(mock_worker_pool):
    """Test that evaluate delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    evaluation_result = {"accuracy": 0.9}

    mock_worker_pool["worker"].evaluate.remote = AsyncMock(return_value=evaluation_result)

    result = await vl.evaluate()

    mock_worker_pool["worker"].evaluate.remote.assert_called_once_with("node-1")
    assert result == evaluation_result


@pytest.mark.asyncio
async def test_train_on_batch_delegates(mock_worker_pool):
    """Test that train_on_batch delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_worker_pool["worker"].train_on_batch.remote = AsyncMock(return_value=mock_model_ref)
    mock_worker_pool["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = await vl.train_on_batch()

    mock_worker_pool["worker"].train_on_batch.remote.assert_called_once_with("node-1")
    assert result == model


@pytest.mark.asyncio
async def test_interrupt_fit_is_noop(mock_worker_pool):
    """Test that interrupt_fit is a no-op (logs only)."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    # Should not raise
    await vl.interrupt_fit()


@pytest.mark.asyncio
async def test_aset_model_async(mock_worker_pool):
    """Test async aset_model delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    model = MagicMock(spec=P2PFLModel)

    mock_worker_pool["worker"].set_model.remote = AsyncMock()

    await vl.aset_model(model)

    mock_worker_pool["ray"].put.assert_called_with(model)
    mock_worker_pool["worker"].set_model.remote.assert_called_once_with("node-1", model)


@pytest.mark.asyncio
async def test_aget_model_async(mock_worker_pool):
    """Test async aget_model delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_worker_pool["worker"].get_model.remote = AsyncMock(return_value=mock_model_ref)
    mock_worker_pool["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = await vl.aget_model()

    mock_worker_pool["worker"].get_model.remote.assert_called_once_with("node-1")
    assert result == model


@pytest.mark.asyncio
async def test_aset_data_async(mock_worker_pool):
    """Test async aset_data delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)
    data = MagicMock(spec=P2PFLDataset)

    mock_worker_pool["worker"].set_data.remote = AsyncMock()

    await vl.aset_data(data)

    mock_worker_pool["ray"].put.assert_called_with(data)
    mock_worker_pool["worker"].set_data.remote.assert_called_once_with("node-1", data)


@pytest.mark.asyncio
async def test_aset_address_async(mock_worker_pool):
    """Test async aset_address rekeys on worker."""
    learner = create_mock_learner(address="old-key")
    vl = VirtualNodeLearner(learner)

    mock_worker_pool["worker"].rekey_node.remote = AsyncMock()
    mock_worker_pool["worker"].set_address.remote = AsyncMock()

    result = await vl.aset_address("new-key")

    mock_worker_pool["worker"].rekey_node.remote.assert_called_once_with("old-key", "new-key")
    mock_worker_pool["worker"].set_address.remote.assert_called_once_with("new-key", "new-key")
    assert result == "new-key"
    assert vl.address == "new-key"


@pytest.mark.asyncio
async def test_aconfigure_async(mock_worker_pool):
    """Test async aconfigure delegates to worker."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    mock_worker_pool["worker"].configure.remote = AsyncMock()

    await vl.aconfigure(epochs=5, steps_per_epoch=50)

    mock_worker_pool["worker"].configure.remote.assert_called_once_with(
        "node-1", epochs=5, steps_per_epoch=50
    )


def test_has_base_attributes(mock_worker_pool):
    """Test that VirtualNodeLearner has required base attributes."""
    learner = create_mock_learner(address="node-1")
    vl = VirtualNodeLearner(learner)

    assert hasattr(vl, "callbacks")
    assert hasattr(vl, "epochs")
    assert hasattr(vl, "steps_per_epoch")
