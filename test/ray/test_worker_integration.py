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
"""Integration test: multiple VirtualNodeLearners sharing a FrameworkWorker."""

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
    from p2pfl.learning.frameworks.ray.virtual_learner import VirtualNodeLearner

pytestmark = pytest.mark.skipif(not RAY_INSTALLED, reason="Ray is not installed")


def create_mock_learner(address: str = "") -> MagicMock:
    """Create a mock Learner with the address attribute set."""
    learner = MagicMock(spec=Learner)
    learner.address = address
    return learner


@pytest.fixture
def mock_pool_and_worker():
    """Mock WorkerPool and ray to avoid Ray initialization.

    Returns a single worker that will be shared by all VirtualNodeLearners.
    """
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


def test_multiple_nodes_share_one_worker(mock_pool_and_worker):
    """Test that multiple VirtualNodeLearners share the same worker.

    Create 5 VirtualNodeLearners and verify:
    1. All 5 call register_node.remote on the same worker
    2. All 5 share the same _worker reference
    """
    # Create 5 VirtualNodeLearners with different addresses
    nodes = []
    for i in range(5):
        learner = create_mock_learner(address=f"node-{i}")
        node = VirtualNodeLearner(learner)
        nodes.append(node)

    # Verify that assign_worker was called 5 times
    assert mock_pool_and_worker["pool"].assign_worker.call_count == 5

    # Verify that all nodes share the same worker reference
    shared_worker = mock_pool_and_worker["worker"]
    for node in nodes:
        assert node._worker is shared_worker

    # Verify that register_node.remote was called 5 times with the correct keys
    assert mock_pool_and_worker["worker"].register_node.remote.call_count == 5
    call_args_list = mock_pool_and_worker["worker"].register_node.remote.call_args_list
    for i, call in enumerate(call_args_list):
        args, _ = call
        node_key, _ = args
        assert node_key == f"node-{i}"


@pytest.mark.asyncio
async def test_multiple_nodes_can_fit_concurrently(mock_pool_and_worker):
    """Test that multiple nodes can call fit concurrently.

    Create 3 VirtualNodeLearners, mock worker.fit.remote as AsyncMock,
    call asyncio.gather(*[node.fit() for node in nodes]),
    verify fit.remote called 3 times and results have length 3.
    """
    # Create 3 VirtualNodeLearners
    nodes = []
    for i in range(3):
        learner = create_mock_learner(address=f"node-{i}")
        node = VirtualNodeLearner(learner)
        nodes.append(node)

    # Mock fit.remote to return a model ref
    mock_models = [MagicMock(spec=P2PFLModel) for _ in range(3)]
    mock_model_refs = [MagicMock() for _ in range(3)]

    # Set up the worker's fit.remote as AsyncMock
    call_count = 0

    async def fit_side_effect(node_key):
        nonlocal call_count
        result = mock_model_refs[call_count]
        call_count += 1
        return result

    mock_pool_and_worker["worker"].fit.remote = AsyncMock(side_effect=fit_side_effect)

    # Set up ray.get to resolve model refs to models
    def ray_get_side_effect(ref):
        if ref in mock_model_refs:
            idx = mock_model_refs.index(ref)
            return mock_models[idx]
        return ref

    mock_pool_and_worker["ray"].get.side_effect = ray_get_side_effect

    # Call fit concurrently on all nodes
    results = await asyncio.gather(*[node.fit() for node in nodes])

    # Verify fit.remote was called 3 times
    assert mock_pool_and_worker["worker"].fit.remote.call_count == 3

    # Verify results have length 3
    assert len(results) == 3

    # Verify each result is the corresponding model
    for i, result in enumerate(results):
        assert result is mock_models[i]


def test_node_address_change_rekeys_correctly(mock_pool_and_worker):
    """Test that changing a node's address rekeys correctly.

    Create 2 nodes with temp addresses, change only one's address,
    verify rekey_node.remote called once with correct old/new keys,
    verify the other node's key is unchanged.
    """
    # Create 2 nodes
    learner1 = create_mock_learner(address="node-0")
    learner2 = create_mock_learner(address="node-1")
    node1 = VirtualNodeLearner(learner1)
    node2 = VirtualNodeLearner(learner2)

    # Verify initial state
    assert node1._node_key == "node-0"
    assert node2._node_key == "node-1"

    # Reset the mock to clear registration calls
    mock_pool_and_worker["worker"].rekey_node.remote.reset_mock()
    mock_pool_and_worker["worker"].set_address.remote.reset_mock()

    # Change node1's address
    node1.set_address("node-0-new")

    # Verify rekey_node.remote was called once with correct old/new keys
    mock_pool_and_worker["worker"].rekey_node.remote.assert_called_once_with(
        "node-0", "node-0-new"
    )
    mock_pool_and_worker["worker"].set_address.remote.assert_called_once_with(
        "node-0-new", "node-0-new"
    )

    # Verify node1's key and address are updated
    assert node1._node_key == "node-0-new"
    assert node1.address == "node-0-new"

    # Verify node2's key is unchanged
    assert node2._node_key == "node-1"
    assert node2.address == "node-1"
