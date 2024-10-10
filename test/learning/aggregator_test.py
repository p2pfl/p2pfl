#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest

from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.pytorch.lightning_model import MLP, LightningModel


class P2PFLModelMock(P2PFLModel):
    """Mock model for testing purposes."""

    def __init__(
        self,
        model: Any,
        params: Optional[Union[List[np.ndarray], bytes]] = None,
        num_samples: Optional[int] = None,
        contributors: Optional[List[str]] = None,
        aditional_info: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the model."""
        self.params = params
        self.num_samples = num_samples  # type: ignore
        self.contributors = contributors  # type: ignore

    def get_parameters(self):
        """Get the model parameters."""
        return self.params

    def get_num_samples(self):
        """Get the number of samples."""
        return self.num_samples

    def build_copy(self, **kwargs):
        """Build a copy of the model."""
        return P2PFLModelMock(None, **kwargs)

    def get_contributors(self) -> List[str]:
        """Get the contributors."""
        return self.contributors


def test_avg_simple():
    """Test simple aggregation (simple arrays)."""
    models = [
        P2PFLModelMock(None, params=[np.array([1, 2, 3])], num_samples=1, contributors=["1"]),
        P2PFLModelMock(None, params=[np.array([4, 5, 6])], num_samples=1, contributors=["2"]),
        P2PFLModelMock(None, params=[np.array([7, 8, 9])], num_samples=1, contributors=["3"]),
    ]
    # New aggregator test
    aggregator = FedAvg()
    res = aggregator.aggregate(models)

    assert np.array_equal(res.get_parameters()[0], np.array([4, 5, 6]))
    assert set(res.get_contributors()) == {"1", "2", "3"}


def test_avg_complex():
    """Test complex aggregation (models)."""
    # Initial Model
    model = LightningModel(MLP(), num_samples=1, contributors=["1"])

    params = model.get_parameters()

    # Model 1
    params1 = []
    for layer in params:
        params1.append(layer + 1.0)

    # Model 2
    params2 = []
    for layer in params:
        params2.append(layer - 1.0)

    # New aggregator test
    aggregator = FedAvg()
    res = aggregator.aggregate(
        [
            model,
            LightningModel(MLP(), params=params1, num_samples=2, contributors=["2"]),
            LightningModel(MLP(), params=params2, num_samples=2, contributors=["3"]),
        ]
    )

    # Assertion: Check if the aggregated parameters are equal to the initial model's parameters
    for i, layer in enumerate(res.get_parameters()):
        assert np.allclose(layer, model.get_parameters()[i], atol=1e-7), f"Layer {i} does not match"
    assert set(res.get_contributors()) == {"1", "2", "3"}


def test_aggregator_lifecycle():
    """Test the aggregator lock."""
    aggregator = FedAvg()
    aggregator.set_nodes_to_aggregate(["node1", "node2", "node3"])

    # Try to set nodes again (should raise an exception)
    with pytest.raises(Exception) as _:
        aggregator.set_nodes_to_aggregate(["node4"])

    # Add a model
    model1 = LightningModel(MLP(), num_samples=1, contributors=["node1"])
    aggregator.add_model(model1)

    # Ensure that the previous lock, now an event is cleared (equivalent to locked)
    assert not aggregator._finish_aggregation_event.is_set()

    # Check if the model was added
    assert aggregator.get_aggregated_models() == ["node1"]

    # Add the rest of the models
    model23 = LightningModel(MLP(), num_samples=1, contributors=["node2", "node3"])
    aggregator.add_model(model23)

    # Get partial aggregation
    partial_model = aggregator.get_partial_aggregation(["node2", "node3"])
    assert all((partial_model.get_parameters()[i] == model1.get_parameters()[i]).all() for i in range(len(partial_model.get_parameters())))

    # Check if the model was added
    assert set(aggregator.get_aggregated_models()) == {"node1", "node2", "node3"}

    # Ensure that the lock is released
    t = time.time()
    aggregator.wait_and_get_aggregation(timeout=1)
    assert time.time() - t < 1

    # Check clear
    aggregator.clear()
    assert aggregator.get_aggregated_models() == []


"""
def test_median_simple():
    raise NotImplementedError

def test_median_complex():
    raise NotImplementedError
"""
