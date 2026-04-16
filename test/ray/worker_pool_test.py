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
"""Tests for WorkerPool singleton."""

from unittest.mock import MagicMock, patch

import pytest

from p2pfl.learning.frameworks.ray.worker_pool import WorkerPool
from p2pfl.settings import Settings

# Valid 28-byte hex node IDs for Ray's NodeAffinitySchedulingStrategy validation
_FAKE_NODE_ID_0 = "a" * 56
_FAKE_NODE_ID_1 = "b" * 56


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset WorkerPool singleton between tests."""
    yield
    # Force-reset the singleton so each test starts fresh
    WorkerPool._instance = None
    WorkerPool._initialized = False


def _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1, total_cpus=8.0, total_gpus=0.0):
    """Configure ray and actor mocks for common scenarios."""
    # Build alive nodes
    fake_ids = [_FAKE_NODE_ID_0, _FAKE_NODE_ID_1]
    nodes = []
    for i in range(num_nodes):
        nodes.append({"NodeID": fake_ids[i], "Alive": True, "NodeName": f"10.0.0.{i}"})
    # Add a dead node to ensure filtering works
    nodes.append({"NodeID": "dead-node", "Alive": False})
    mock_ray.nodes.return_value = nodes

    resources = {"CPU": total_cpus}
    if total_gpus > 0:
        resources["GPU"] = total_gpus
    mock_ray.cluster_resources.return_value = resources

    # Make the actor class chainable: FrameworkWorkerActor.options(...).remote(...)
    mock_actor_handle = MagicMock()
    mock_options_cls = MagicMock()
    mock_options_cls.remote.return_value = mock_actor_handle
    mock_actor_cls.options.return_value = mock_options_cls

    # Settings
    Settings.training.RAY_ACTOR_POOL_SIZE = 0  # auto

    return mock_actor_handle, mock_options_cls


class TestWorkerPoolSingleton:
    """Test WorkerPool singleton behaviour."""

    def test_worker_pool_singleton(self):
        """Two instantiations return the same object."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1)
            pool_a = WorkerPool()
            pool_b = WorkerPool()
            assert pool_a is pool_b

    def test_worker_pool_single_machine_spawns_one_worker(self):
        """1 alive node should spawn exactly 1 worker."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            mock_actor_handle, mock_options_cls = _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1, total_cpus=8.0, total_gpus=1.0)
            pool = WorkerPool()
            assert pool.worker_count == 1
            # Should have called options().remote() once
            mock_actor_cls.options.assert_called_once()
            mock_options_cls.remote.assert_called_once()

    def test_worker_pool_multi_node_spawns_per_node(self):
        """2 alive nodes should spawn 2 workers, one per node."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            mock_actor_handle, mock_options_cls = _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=2, total_cpus=16.0, total_gpus=2.0)
            pool = WorkerPool()
            assert pool.worker_count == 2
            assert mock_actor_cls.options.call_count == 2
            assert mock_options_cls.remote.call_count == 2

    def test_assign_worker_round_robins(self):
        """With 2 workers, 3 assignments should cycle a, b, a."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            handle_a = MagicMock(name="worker-a")
            handle_b = MagicMock(name="worker-b")
            mock_options_cls = MagicMock()
            mock_options_cls.remote.side_effect = [handle_a, handle_b]
            mock_actor_cls.options.return_value = mock_options_cls

            nodes = [
                {"NodeID": _FAKE_NODE_ID_0, "Alive": True, "NodeName": "10.0.0.0"},
                {"NodeID": _FAKE_NODE_ID_1, "Alive": True, "NodeName": "10.0.0.1"},
            ]
            mock_ray.nodes.return_value = nodes
            mock_ray.cluster_resources.return_value = {"CPU": 16.0, "GPU": 0.0}
            Settings.training.RAY_ACTOR_POOL_SIZE = 0

            pool = WorkerPool()
            first = pool.assign_worker()
            second = pool.assign_worker()
            third = pool.assign_worker()

            assert first is handle_a
            assert second is handle_b
            assert third is handle_a

    def test_shutdown_clears_singleton(self):
        """Shutdown kills workers and resets the singleton."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            mock_actor_handle, _ = _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1)
            pool = WorkerPool()
            assert pool.worker_count == 1

            pool.shutdown()

            mock_ray.kill.assert_called_once_with(mock_actor_handle)
            assert WorkerPool._instance is None

    def test_auto_detect_max_concurrent(self):
        """RAY_ACTOR_POOL_SIZE=0 auto-detects as total_cpus - 1 (keep 1 free)."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            _, mock_options_cls = _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1, total_cpus=8.0)
            Settings.training.RAY_ACTOR_POOL_SIZE = 0
            WorkerPool()
            # auto = ceil((8 - 1) / 2) = 4, with +10 headroom = max_concurrency 14
            call_kwargs = mock_actor_cls.options.call_args[1]
            assert call_kwargs["max_concurrency"] == 14

    def test_explicit_pool_size(self):
        """RAY_ACTOR_POOL_SIZE > 0 uses explicit value."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            _, mock_options_cls = _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1, total_cpus=8.0)
            Settings.training.RAY_ACTOR_POOL_SIZE = 3
            WorkerPool()
            # explicit 3 + 10 headroom = 13
            call_kwargs = mock_actor_cls.options.call_args[1]
            assert call_kwargs["max_concurrency"] == 13


class TestWorkerPoolRegistrations:
    """Test crash-recovery registration tracking."""

    def test_register_tracks_registration(self):
        """Test that register_node tracks the registration."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1)
            pool = WorkerPool()
            worker = pool.assign_worker()
            learner = MagicMock()

            pool.register_node(worker, "node_0", learner)

            idx = pool._worker_index(worker)
            assert len(pool._registrations[idx]) == 1
            assert pool._registrations[idx][0].node_id == "node_0"

    def test_rekey_updates_registration(self):
        """Test that rekey_node updates the node_id in registrations."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1)
            pool = WorkerPool()
            worker = pool.assign_worker()
            learner = MagicMock()

            pool.register_node(worker, "old_id", learner)
            pool.rekey_node(worker, "old_id", "new_id")

            idx = pool._worker_index(worker)
            assert pool._registrations[idx][0].node_id == "new_id"

    def test_unregister_removes_registration(self):
        """Test that unregister_node removes the registration record."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1)
            pool = WorkerPool()
            worker = pool.assign_worker()

            pool.register_node(worker, "node_0", MagicMock())
            pool.unregister_node(worker, "node_0")

            idx = pool._worker_index(worker)
            assert len(pool._registrations[idx]) == 0

    def test_shutdown_clears_registrations(self):
        """Test that shutdown clears all registration records."""
        with (
            patch("p2pfl.learning.frameworks.ray.worker_pool.ray") as mock_ray,
            patch("p2pfl.learning.frameworks.ray.worker_pool.FrameworkWorkerActor") as mock_actor_cls,
        ):
            _setup_ray_mocks(mock_ray, mock_actor_cls, num_nodes=1)
            pool = WorkerPool()
            worker = pool.assign_worker()

            pool.register_node(worker, "node_0", MagicMock())
            pool.shutdown()

            assert len(pool._registrations) == 0
