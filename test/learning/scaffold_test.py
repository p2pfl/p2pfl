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

"""Unit tests for the Scaffold."""

import copy
from typing import Any

import numpy as np
import pytest

from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class MockP2PFLModel(P2PFLModel):
    """Mock P2PFLModel for testing Scaffold aggregator."""

    def __init__(
        self,
        params: list[np.ndarray],
        num_samples: int,
        additional_info: dict[str, Any] | None = None,
        contributors: list[str] | None = None,
    ) -> None:
        """Initialize the MockP2PFLModel."""
        self._params = params
        self.num_samples = num_samples
        self.additional_info = additional_info or {}
        self._contributors = contributors or ["contributor1", "contributor2"]

    def get_parameters(self) -> list[np.ndarray]:
        """Return the model parameters."""
        return self._params

    def get_num_samples(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def add_info(self, callback: str, info: Any) -> None:
        """Add additional information to the learner state."""
        self.additional_info[callback] = info

    def get_info(self, callback: str | None = None) -> Any:
        """Get additional information from the learner state."""
        if callback is None:
            return self.additional_info
        return self.additional_info.get(callback, {})

    def build_copy(self, **kwargs) -> "MockP2PFLModel":
        """Build a copy of the model with the specified parameters."""
        return self.__class__(
            params=kwargs.get("params", copy.deepcopy(self._params)),
            num_samples=kwargs.get("num_samples", self.num_samples),
            additional_info=kwargs.get("additional_info", copy.deepcopy(self.additional_info)),
            contributors=kwargs.get("contributors", copy.deepcopy(self._contributors)),
        )

    def get_contributors(self) -> list[str]:
        """Return the contributors."""
        return self._contributors


def test_scaffold_correct_aggregation():
    """Test if scaffold aggregates delta_y_i and delta_c_i correctly."""
    aggr = Scaffold(global_lr=0.1)
    aggr.set_addr("test")

    # Initial params and mock models
    initial_global_model_params = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    aggr.global_model_params = copy.deepcopy(initial_global_model_params)
    aggr.c = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]

    model1 = MockP2PFLModel(
        params=[np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        num_samples=10,
        additional_info={
            "scaffold": {
                "delta_y_i": [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
                "delta_c_i": [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
            }
        },
        contributors=["client1"],
    )

    model2 = MockP2PFLModel(
        params=[np.array([2.0, 2.0]), np.array([2.0, 2.0])],
        num_samples=20,
        additional_info={
            "scaffold": {
                "delta_y_i": [np.array([2.0, 2.0]), np.array([2.0, 2.0])],
                "delta_c_i": [np.array([2.0, 2.0]), np.array([2.0, 2.0])],
            }
        },
        contributors=["client2"],
    )

    # Expected values
    total_samples = model1.get_num_samples() + model2.get_num_samples()  # 10 + 20 = 30
    expected_delta_y_i = [
        (1.0 * 10 + 2.0 * 20) / total_samples * aggr.global_lr,  # (1 * 10 + 2 * 20) / 30 * 0.1 = 0.16
        (1.0 * 10 + 2.0 * 20) / total_samples * aggr.global_lr,
    ]
    expected_global_model_params = [
        initial_global_model_params[0] + expected_delta_y_i[0],  # 0.0 + 0.16 = 0.16
        initial_global_model_params[1] + expected_delta_y_i[1],
    ]

    # Aggregate and check correspondence
    aggregated_model = aggr.aggregate([model1, model2])

    # check correct global model updates
    for aggr_param, expected_param in zip(aggregated_model.get_parameters(), expected_global_model_params, strict=False):
        assert np.allclose(aggr_param, expected_param, atol=1e-7), "Aggregated model parameters do not match"

    # check correct contributors
    expected_contributors = ["client1", "client2"]
    assert set(aggregated_model.get_contributors()) == set(expected_contributors), "Contributors do not match"

    # check correct global control variates updates
    expected_c = [
        (1.0 + 2.0) / 2,  # (1 + 2) / 2 = 1.5
        (1.0 + 2.0) / 2,
    ]
    for global_c, expected in zip(aggr.c, expected_c, strict=False):
        assert np.allclose(global_c, expected, atol=1e-7), "Global control variates do not match"


def test_scaffold_no_models():
    """Test that aggregating with no models raising proper exception."""
    aggr = Scaffold()
    aggr.set_addr("test")
    with pytest.raises(NoModelsToAggregateError):
        aggr.aggregate([])


def test_scaffold_missing_info():
    """Test that aggregating with missing information raises proper exceptions."""
    aggr = Scaffold()
    aggr.set_addr("test")

    # mock models with missing delta_y and delta_c
    model_missing_delta_y = MockP2PFLModel(
        params=[np.array([1.0, 1.0])], num_samples=10, additional_info={"scaffold": {"delta_c_i": [np.array([1.0, 1.0])]}}
    )

    model_missing_delta_c = MockP2PFLModel(
        params=[np.array([1.0, 1.0])], num_samples=10, additional_info={"scaffold": {"delta_y_i": [np.array([1.0, 1.0])]}}
    )

    # check that missing delta_y raises ValueError
    with pytest.raises(ValueError):
        aggr.aggregate([model_missing_delta_y])

    with pytest.raises(ValueError):
        aggr.aggregate([model_missing_delta_c])
