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
FrameworkWorkerActor tests.

Tests access the underlying class directly (no Ray runtime needed).
"""

import asyncio
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
    from p2pfl.learning.frameworks.ray.framework_worker import FrameworkWorkerActor

# Skip all tests in this module if Ray is not installed
pytestmark = pytest.mark.skipif(not RAY_INSTALLED, reason="Ray is not installed")


def _unwrap_actor_class(cls):
    """Get the underlying class from a @ray.remote decorated class."""
    if hasattr(cls, "__ray_actor_class__"):
        return cls.__ray_actor_class__
    return cls


def _make_worker(max_concurrent: int = 2):
    """Create a FrameworkWorkerActor instance without Ray runtime."""
    cls = _unwrap_actor_class(FrameworkWorkerActor)
    worker = cls.__new__(cls)
    worker.__init__(max_concurrent=max_concurrent)
    return worker


def _make_mock_learner(address: str = "node_0") -> MagicMock:
    """Create a mock Learner for testing."""
    learner = MagicMock(spec=Learner)
    learner.address = address
    learner.get_model.return_value = MagicMock(spec=P2PFLModel)
    learner.fit = AsyncMock(return_value=MagicMock(spec=P2PFLModel))
    learner.train_on_batch = AsyncMock(return_value=MagicMock(spec=P2PFLModel))
    learner.evaluate = AsyncMock(return_value={"loss": 0.5, "accuracy": 0.9})
    return learner


# --- Registration tests ---


def test_register_node():
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    assert "node_0" in worker._registry


def test_unregister_node():
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    worker.unregister_node("node_0")
    assert "node_0" not in worker._registry


def test_rekey_node():
    worker = _make_worker()
    learner = _make_mock_learner("old_id")
    worker.register_node("old_id", learner)
    worker.rekey_node("old_id", "new_id")
    assert "old_id" not in worker._registry
    assert "new_id" in worker._registry


# --- Delegation tests ---


@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
def test_set_model(mock_ray):
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    model = MagicMock(spec=P2PFLModel)
    worker.set_model("node_0", model)
    learner.set_model.assert_called_once_with(model)


@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
def test_get_model(mock_ray):
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    mock_ray.put.return_value = "fake_ref"
    result = worker.get_model("node_0")
    learner.get_model.assert_called_once()
    mock_ray.put.assert_called_once_with(learner.get_model.return_value)
    assert result == "fake_ref"


@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
def test_set_data(mock_ray):
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    data = MagicMock(spec=P2PFLDataset)
    worker.set_data("node_0", data)
    learner.set_data.assert_called_once_with(data)


def test_get_data():
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    data = MagicMock(spec=P2PFLDataset)
    learner.get_data.return_value = data
    result = worker.get_data("node_0")
    assert result is data


# --- Configuration tests ---


def test_configure_batches_settings():
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    worker.set_epochs("node_0", 5)
    worker.set_steps_per_epoch("node_0", 100)
    learner.set_epochs.assert_called_once_with(5)
    learner.set_steps_per_epoch.assert_called_once_with(100)


# --- Error handling ---


def test_unregistered_node_raises():
    worker = _make_worker()
    with pytest.raises(KeyError):
        worker.set_model("unknown_node", MagicMock())


# --- Async training tests ---


@pytest.mark.asyncio
@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
async def test_fit_delegates_to_learner(mock_ray):
    """Test that fit() falls back to learner.fit() when train_on_batch is not supported."""
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    # Make train_on_batch raise NotImplementedError so fit() falls back
    learner.train_on_batch = AsyncMock(side_effect=NotImplementedError)
    learner.get_epochs.return_value = 1
    learner.get_steps_per_epoch.return_value = None
    data = MagicMock()
    data.get_num_samples.return_value = 100
    learner.get_data.return_value = data
    worker.register_node("node_0", learner)
    mock_ray.put.return_value = "fit_ref"
    result = await worker.fit("node_0")
    learner.fit.assert_awaited_once()
    mock_ray.put.assert_called_once()
    assert result == "fit_ref"


@pytest.mark.asyncio
@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
async def test_evaluate_delegates_to_learner(mock_ray):
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    result = await worker.evaluate("node_0")
    learner.evaluate.assert_awaited_once()
    assert result == {"loss": 0.5, "accuracy": 0.9}


@pytest.mark.asyncio
@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
async def test_train_on_batch_delegates_to_learner(mock_ray):
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)
    mock_ray.put.return_value = "batch_ref"
    result = await worker.train_on_batch("node_0")
    learner.train_on_batch.assert_awaited_once()
    mock_ray.put.assert_called_once()
    assert result == "batch_ref"


@pytest.mark.asyncio
@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
async def test_concurrent_fit_limited_by_semaphore(mock_ray):
    """With max_concurrent=1, verify fits don't interleave."""
    worker = _make_worker(max_concurrent=1)
    mock_ray.put.return_value = "fit_ref"

    # Track the order of operations
    order = []
    fit_event = asyncio.Event()

    learner_a = _make_mock_learner("node_a")
    learner_b = _make_mock_learner("node_b")

    # Make both learners fall back to learner.fit() via NotImplementedError
    for lrn in (learner_a, learner_b):
        lrn.train_on_batch = AsyncMock(side_effect=NotImplementedError)
        lrn.get_epochs.return_value = 1
        lrn.get_steps_per_epoch.return_value = None
        data = MagicMock()
        data.get_num_samples.return_value = 100
        lrn.get_data.return_value = data

    async def slow_fit_a():
        order.append("a_start")
        await fit_event.wait()
        order.append("a_end")
        return MagicMock(spec=P2PFLModel)

    async def fast_fit_b():
        order.append("b_start")
        order.append("b_end")
        return MagicMock(spec=P2PFLModel)

    learner_a.fit = slow_fit_a
    learner_b.fit = fast_fit_b

    worker.register_node("node_a", learner_a)
    worker.register_node("node_b", learner_b)

    task_a = asyncio.create_task(worker.fit("node_a"))
    # Give task_a a chance to acquire the semaphore
    await asyncio.sleep(0.01)
    task_b = asyncio.create_task(worker.fit("node_b"))
    # Give task_b a chance to attempt acquiring the semaphore
    await asyncio.sleep(0.01)

    # At this point, a should have started but b should be blocked
    assert "a_start" in order
    assert "b_start" not in order

    # Release a
    fit_event.set()
    await asyncio.gather(task_a, task_b)

    # a must finish before b starts
    assert order.index("a_end") < order.index("b_start")


# --- Batch-level interleaved training tests ---


@pytest.mark.asyncio
async def test_fit_uses_interleaved_batches():
    """Test that fit() uses train_on_batch loop when supported."""
    worker = _make_worker(max_concurrent=2)
    learner = _make_mock_learner("node_0")

    # Set up learner to support train_on_batch
    model = MagicMock(spec=P2PFLModel)
    learner.get_epochs.return_value = 2
    learner.get_steps_per_epoch.return_value = 3
    learner.train_on_batch = AsyncMock(return_value=model)
    learner.get_model.return_value = model
    data = MagicMock()
    data.get_num_samples.return_value = 100
    learner.get_data.return_value = data

    worker.register_node("node_0", learner)

    with patch("p2pfl.learning.frameworks.ray.framework_worker.ray") as mock_ray:
        mock_ray.put.return_value = MagicMock()
        await worker.fit("node_0")

    # Should have called train_on_batch 6 times (2 epochs * 3 steps)
    assert learner.train_on_batch.call_count == 6
    # Should NOT have called learner.fit()
    learner.fit.assert_not_called()


@pytest.mark.asyncio
async def test_fit_falls_back_when_train_on_batch_not_supported():
    """Test that fit() falls back to learner.fit() when train_on_batch raises NotImplementedError."""
    worker = _make_worker(max_concurrent=2)
    learner = _make_mock_learner("node_0")

    model = MagicMock(spec=P2PFLModel)
    learner.get_epochs.return_value = 1
    learner.get_steps_per_epoch.return_value = None
    learner.train_on_batch = AsyncMock(side_effect=NotImplementedError)
    learner.fit = AsyncMock(return_value=model)
    learner.get_model.return_value = model
    data = MagicMock()
    data.get_num_samples.return_value = 100
    learner.get_data.return_value = data

    worker.register_node("node_0", learner)

    with patch("p2pfl.learning.frameworks.ray.framework_worker.ray") as mock_ray:
        mock_ray.put.return_value = MagicMock()
        await worker.fit("node_0")

    # Should have fallen back to learner.fit()
    learner.fit.assert_called_once()


# --- Memory-aware LRU tracking tests ---


def test_touch_updates_last_used():
    """Test that operations update the last_used timestamp."""
    import time

    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)

    initial_time = worker._registry["node_0"].last_used
    time.sleep(0.01)
    worker._touch("node_0")
    assert worker._registry["node_0"].last_used > initial_time


def test_touch_nonexistent_node_is_noop():
    """Test that _touch on a missing node does not raise."""
    worker = _make_worker()
    worker._touch("does_not_exist")  # should not raise


def test_get_memory_stats():
    """Test that get_memory_stats returns expected keys."""
    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)

    stats = worker.get_memory_stats()
    assert stats["node_count"] == 1
    assert "rss_mb" in stats


def test_get_memory_stats_empty_registry():
    """Test get_memory_stats with no registered nodes."""
    worker = _make_worker()
    stats = worker.get_memory_stats()
    assert stats["node_count"] == 0


@patch("p2pfl.learning.frameworks.ray.framework_worker.ray")
def test_public_methods_update_last_used(mock_ray):
    """Test that public methods call _touch to update last_used."""
    import time

    worker = _make_worker()
    learner = _make_mock_learner("node_0")
    worker.register_node("node_0", learner)

    time.sleep(0.01)
    t_before = time.time()
    worker.set_model("node_0", MagicMock())
    assert worker._registry["node_0"].last_used >= t_before
