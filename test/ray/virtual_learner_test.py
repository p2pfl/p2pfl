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

These tests mock the Ray actor and placement group to avoid actually running Ray,
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
    from p2pfl.learning.frameworks.ray.virtual_learner import VirtualLearnerActor, VirtualNodeLearner

# Skip all tests in this module if Ray is not installed
pytestmark = pytest.mark.skipif(not RAY_INSTALLED, reason="Ray is not installed")


def create_mock_learner(address: str = "") -> MagicMock:
    """Create a mock Learner with the address attribute set (required by VirtualNodeLearner)."""
    learner = MagicMock(spec=Learner)
    learner.address = address  # NodeComponent sets this via metaclass; mocks need it explicitly
    return learner


@pytest.fixture
def mock_ray_components():
    """Mock VirtualLearnerActor and PlacementGroupManager to avoid Ray initialization."""
    mock_actor = MagicMock()
    mock_actor_options = MagicMock()
    mock_actor_options.remote.return_value = mock_actor

    mock_actor_class = MagicMock()
    mock_actor_class.options.return_value = mock_actor_options

    mock_pg_manager = MagicMock()
    mock_pg_manager.get_placement_group.return_value = None

    with (
        patch(
            "p2pfl.learning.frameworks.ray.virtual_learner.VirtualLearnerActor",
            mock_actor_class,
        ),
        patch(
            "p2pfl.learning.frameworks.ray.virtual_learner.PlacementGroupManager",
            return_value=mock_pg_manager,
        ),
        patch("p2pfl.learning.frameworks.ray.virtual_learner.ray") as mock_ray,
    ):
        # Setup ray.get to return whatever the remote call returns
        mock_ray.get.side_effect = lambda x: x
        yield {
            "actor": mock_actor,
            "actor_class": mock_actor_class,
            "pg_manager": mock_pg_manager,
            "ray": mock_ray,
        }


def test_virtual_node_learner_has_base_attributes(mock_ray_components):
    """Test that VirtualNodeLearner initializes Learner base attributes."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    assert hasattr(virtual_learner, 'callbacks')
    assert hasattr(virtual_learner, 'epochs')
    assert hasattr(virtual_learner, 'steps_per_epoch')


def test_virtual_node_learner_initialization(mock_ray_components):
    """Test the initialization of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    address = "test_addr"
    virtual_learner.set_address(address)
    assert virtual_learner.address == address
    # Verify actor was created
    mock_ray_components["actor_class"].options.assert_called_once()


def test_set_model(mock_ray_components):
    """Test the set_model method uses ray.put for object store transfer."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_ref = MagicMock()
    mock_ray_components["ray"].put.return_value = mock_ref

    virtual_learner.set_model(model)

    # Verify ray.put was called with the model and set_model_ref was called with the ref
    mock_ray_components["ray"].put.assert_called_once_with(model)
    mock_ray_components["actor"].set_model_ref.remote.assert_called_once_with(mock_ref)


def test_get_model(mock_ray_components):
    """Test the get_model method retrieves via object store ref."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_ray_components["actor"].get_model_ref.remote.return_value = mock_model_ref
    mock_ray_components["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = virtual_learner.get_model()

    mock_ray_components["actor"].get_model_ref.remote.assert_called_once()
    assert result == model


def test_set_data(mock_ray_components):
    """Test the set_data method uses ray.put for object store transfer."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    data = MagicMock(spec=P2PFLDataset)

    mock_ref = MagicMock()
    mock_ray_components["ray"].put.return_value = mock_ref

    virtual_learner.set_data(data)

    # Verify ray.put was called with the data and set_data_ref was called with the ref
    mock_ray_components["ray"].put.assert_called_once_with(data)
    mock_ray_components["actor"].set_data_ref.remote.assert_called_once_with(mock_ref)


def test_set_model_uses_ray_put(mock_ray_components):
    """Test that set_model uses ray.put for object store transfer."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_ref = MagicMock()
    mock_ray_components["ray"].put.return_value = mock_ref

    virtual_learner.set_model(model)

    mock_ray_components["ray"].put.assert_called_once_with(model)
    mock_ray_components["actor"].set_model_ref.remote.assert_called_once_with(mock_ref)


def test_get_model_uses_object_store(mock_ray_components):
    """Test that get_model retrieves via object store ref."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_ray_components["actor"].get_model_ref.remote.return_value = mock_model_ref
    mock_ray_components["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = virtual_learner.get_model()

    mock_ray_components["actor"].get_model_ref.remote.assert_called_once()
    assert result == model


def test_set_data_uses_ray_put(mock_ray_components):
    """Test that set_data uses ray.put for object store transfer."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    data = MagicMock(spec=P2PFLDataset)

    mock_ref = MagicMock()
    mock_ray_components["ray"].put.return_value = mock_ref

    virtual_learner.set_data(data)

    mock_ray_components["ray"].put.assert_called_once_with(data)
    mock_ray_components["actor"].set_data_ref.remote.assert_called_once_with(mock_ref)


def test_get_data(mock_ray_components):
    """Test the get_data method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    data = MagicMock(spec=P2PFLDataset)

    mock_ray_components["actor"].get_data.remote.return_value = data

    result = virtual_learner.get_data()

    mock_ray_components["actor"].get_data.remote.assert_called_once()
    assert result == data


def test_set_epochs(mock_ray_components):
    """Test the set_epochs method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    epochs = 10

    virtual_learner.set_epochs(epochs)

    mock_ray_components["actor"].set_epochs.remote.assert_called_once_with(epochs)


@pytest.mark.asyncio
async def test_fit(mock_ray_components):
    """Test the fit method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")

    # Mock the async remote call
    mock_ray_components["actor"].fit.remote = AsyncMock()

    await virtual_learner.fit()

    mock_ray_components["actor"].fit.remote.assert_called_once()


@pytest.mark.asyncio
async def test_interrupt_fit(mock_ray_components):
    """Test that interrupt_fit cancels the pending fit task."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")

    mock_ref = MagicMock()
    virtual_learner._pending_fit_ref = mock_ref

    await virtual_learner.interrupt_fit()

    mock_ray_components["ray"].cancel.assert_called_once_with(mock_ref, force=False)
    assert virtual_learner._pending_fit_ref is None


@pytest.mark.asyncio
async def test_interrupt_fit_no_pending(mock_ray_components):
    """Test that interrupt_fit is a no-op when no fit is running."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")

    await virtual_learner.interrupt_fit()

    mock_ray_components["ray"].cancel.assert_not_called()


@pytest.mark.asyncio
async def test_evaluate(mock_ray_components):
    """Test the evaluate method of the VirtualNodeLearner class."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    evaluation_result = {"accuracy": 0.9}

    # Mock the async remote call
    mock_ray_components["actor"].evaluate.remote = AsyncMock(return_value=evaluation_result)

    result = await virtual_learner.evaluate()

    mock_ray_components["actor"].evaluate.remote.assert_called_once()
    assert result == evaluation_result


def test_configure_batches_settings(mock_ray_components):
    """Test that configure sends all settings in one remote call."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")

    virtual_learner.configure(epochs=10, steps_per_epoch=100)

    mock_ray_components["actor"].configure.remote.assert_called_once_with(
        epochs=10, steps_per_epoch=100
    )


def test_placement_group_manager_shutdown_resets_singleton(mock_ray_components):
    """Test that shutdown resets the singleton so a new instance can be created."""
    from unittest.mock import patch, MagicMock

    with patch("p2pfl.learning.frameworks.ray.placement_group_manager.ray") as mock_ray, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.placement_group") as mock_pg, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.remove_placement_group") as mock_remove:
        mock_ray.cluster_resources.return_value = {"CPU": 4}
        mock_pg_obj = MagicMock()
        mock_pg_obj.ready.return_value = MagicMock()
        mock_pg.return_value = mock_pg_obj
        mock_ray.get.return_value = None

        from p2pfl.learning.frameworks.ray.placement_group_manager import PlacementGroupManager
        PlacementGroupManager._instance = None

        mgr1 = PlacementGroupManager()
        assert mgr1._initialized is True

        mgr1.shutdown()
        assert PlacementGroupManager._instance is None

        mgr2 = PlacementGroupManager()
        assert mgr2._initialized is True
        assert mgr2 is not mgr1


def test_virtual_learner_actor_has_all_learner_methods():
    """Test that VirtualLearnerActor has delegates for all public Learner methods."""
    import inspect
    from p2pfl.learning.frameworks.learner import Learner
    from p2pfl.learning.frameworks.ray.virtual_learner import VirtualLearnerActor

    learner_methods = {
        name for name, _ in inspect.getmembers(Learner, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    actor_cls = VirtualLearnerActor
    if hasattr(actor_cls, '__ray_actor_class__'):
        actor_cls = actor_cls.__ray_actor_class__
    actor_methods = {
        name for name, _ in inspect.getmembers(actor_cls, predicate=callable)
        if not name.startswith("_")
    }

    missing = learner_methods - actor_methods
    assert not missing, f"VirtualLearnerActor is missing delegates for: {missing}"


def test_explicit_actor_methods_not_overwritten_by_delegates():
    """Test that explicit methods on VirtualLearnerActor are not overwritten by _with_learner_delegates."""
    from p2pfl.learning.frameworks.ray.virtual_learner import VirtualLearnerActor

    actor_cls = VirtualLearnerActor
    if hasattr(actor_cls, '__ray_actor_class__'):
        actor_cls = actor_cls.__ray_actor_class__

    # These methods are explicitly defined on the actor class and must NOT be
    # replaced by the generic delegate (they have custom ray.put logic).
    for method_name in ("fit", "train_on_batch", "configure"):
        method = actor_cls.__dict__[method_name]
        # The delegate creates generic lambdas; explicit methods have distinct source.
        # A simple check: the explicit fit/train_on_batch contain 'ray.put' in source.
        import inspect
        source = inspect.getsource(method)
        if method_name in ("fit", "train_on_batch"):
            assert "ray.put" in source, (
                f"VirtualLearnerActor.{method_name} was overwritten by delegate — "
                f"missing ray.put in source"
            )
        elif method_name == "configure":
            assert "set_epochs" in source, (
                f"VirtualLearnerActor.{method_name} was overwritten by delegate — "
                f"missing set_epochs in source"
            )


def test_placement_group_manager_per_actor_bundles():
    """Test that num_actors divides resources into per-actor bundles."""
    from unittest.mock import patch, MagicMock
    from p2pfl.learning.frameworks.ray.placement_group_manager import PlacementGroupManager

    with patch("p2pfl.learning.frameworks.ray.placement_group_manager.ray") as mock_ray, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.placement_group") as mock_pg, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.remove_placement_group"):
        mock_ray.cluster_resources.return_value = {"CPU": 8, "GPU": 2}
        mock_ray.nodes.return_value = [{"NodeID": "node1", "Alive": True}]
        mock_pg_obj = MagicMock()
        mock_pg_obj.ready.return_value = MagicMock()
        mock_pg.return_value = mock_pg_obj
        mock_ray.get.return_value = None

        PlacementGroupManager._instance = None
        mgr = PlacementGroupManager(num_actors=4)

        assert len(mgr.bundles) == 4
        for bundle in mgr.bundles:
            assert bundle["CPU"] == 2.0
            assert bundle["GPU"] == 0.5

        mock_pg.assert_called_once_with(mgr.bundles, strategy="PACK")
        mgr.shutdown()


def test_placement_group_manager_spread_multi_node():
    """Test that multi-node clusters use SPREAD strategy."""
    from unittest.mock import patch, MagicMock
    from p2pfl.learning.frameworks.ray.placement_group_manager import PlacementGroupManager

    with patch("p2pfl.learning.frameworks.ray.placement_group_manager.ray") as mock_ray, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.placement_group") as mock_pg, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.remove_placement_group"):
        mock_ray.cluster_resources.return_value = {"CPU": 16, "GPU": 4}
        mock_ray.nodes.return_value = [{"NodeID": "n1", "Alive": True}, {"NodeID": "n2", "Alive": True}]
        mock_pg_obj = MagicMock()
        mock_pg_obj.ready.return_value = MagicMock()
        mock_pg.return_value = mock_pg_obj
        mock_ray.get.return_value = None

        PlacementGroupManager._instance = None
        mgr = PlacementGroupManager(num_actors=4)

        mock_pg.assert_called_once_with(mgr.bundles, strategy="SPREAD")
        mgr.shutdown()


@pytest.mark.asyncio
async def test_aset_model_uses_ray_put(mock_ray_components):
    """Test that aset_model uses ray.put and awaits the remote call."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_ref = MagicMock()
    mock_ray_components["ray"].put.return_value = mock_ref
    mock_ray_components["actor"].set_model_ref.remote = AsyncMock()

    await virtual_learner.aset_model(model)

    mock_ray_components["ray"].put.assert_called_with(model)
    mock_ray_components["actor"].set_model_ref.remote.assert_called_once_with(mock_ref)


@pytest.mark.asyncio
async def test_aget_model_returns_model(mock_ray_components):
    """Test that aget_model awaits actor and resolves ref."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_ray_components["actor"].get_model_ref.remote = AsyncMock(return_value=mock_model_ref)
    mock_ray_components["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = await virtual_learner.aget_model()

    assert result == model


@pytest.mark.asyncio
async def test_fit_returns_model_directly(mock_ray_components):
    """Test that fit returns the model without a separate get_model call."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_ray_components["actor"].fit.remote = AsyncMock(return_value=mock_model_ref)
    mock_ray_components["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = await virtual_learner.fit()

    assert result == model
    mock_ray_components["actor"].get_model_ref.remote.assert_not_called()


@pytest.mark.asyncio
async def test_train_on_batch_returns_model_directly(mock_ray_components):
    """Test that train_on_batch returns the model without a separate get_model call."""
    learner = create_mock_learner()
    virtual_learner = VirtualNodeLearner(learner)
    virtual_learner.set_address("test_addr")
    model = MagicMock(spec=P2PFLModel)

    mock_model_ref = MagicMock()
    mock_ray_components["actor"].train_on_batch.remote = AsyncMock(return_value=mock_model_ref)
    mock_ray_components["ray"].get.side_effect = lambda x: model if x == mock_model_ref else x

    result = await virtual_learner.train_on_batch()

    assert result == model


def test_placement_group_manager_default_unchanged():
    """Test that default behavior (no num_actors) is unchanged."""
    from unittest.mock import patch, MagicMock
    from p2pfl.learning.frameworks.ray.placement_group_manager import PlacementGroupManager

    with patch("p2pfl.learning.frameworks.ray.placement_group_manager.ray") as mock_ray, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.placement_group") as mock_pg, \
         patch("p2pfl.learning.frameworks.ray.placement_group_manager.remove_placement_group"):
        mock_ray.cluster_resources.return_value = {"CPU": 8}
        mock_ray.nodes.return_value = [{"NodeID": "node1", "Alive": True}]
        mock_pg_obj = MagicMock()
        mock_pg_obj.ready.return_value = MagicMock()
        mock_pg.return_value = mock_pg_obj
        mock_ray.get.return_value = None

        PlacementGroupManager._instance = None
        mgr = PlacementGroupManager()

        assert len(mgr.bundles) == 1
        assert mgr.bundles[0]["CPU"] == 8
        mock_pg.assert_called_once_with(mgr.bundles, strategy="PACK")
        mgr.shutdown()
