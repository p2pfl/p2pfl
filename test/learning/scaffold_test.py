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

"""Unit tests for the ScaffoldAggregator."""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError
from p2pfl.learning.aggregators.scaffold import ScaffoldAggregator
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class MockP2PFLModel(P2PFLModel):
    """Mock P2PFLModel for testing the ScaffoldAggregator."""

    def __init__(
        self,
        params: List[np.ndarray],
        num_samples: int,
        additional_info: Optional[Dict[str, Any]] = None,
        contributors: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the MockP2PFLModel.

        Args:
            params (List[np.ndarray]): The model parameters.
            num_samples (int): The number of samples.
            additional_info (Optional[Dict[str, Any]]): Additional information.
            contributors (Optional[List[str]]): List of contributors.

        """
        self._params = params
        self.num_samples = num_samples
        self.additional_info = additional_info or {}
        self._contributors = contributors or ["contributor1", "contributor2"]

    def get_parameters(self) -> List[np.ndarray]:
        """Return the model parameters."""
        return self._params

    def get_num_samples(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def get_info(self, key: str) -> Any:
        """Return additional information."""
        return self.additional_info[key]

    def build_copy(self, params: List[np.ndarray], num_samples: int, contributors: List[str]) -> "MockP2PFLModel":
        """
        Build a copy of the model with the specified parameters.

        Args:
            params (List[np.ndarray]): The model parameters.
            num_samples (int): The number of samples.
            contributors (List[str]): List of contributors.

        """
        return MockP2PFLModel(
            params=params, num_samples=num_samples, additional_info=self.additional_info.copy(), contributors=contributors
        )

    def add_info(self, key: str, value: Any) -> None:
        """Add additional information."""
        self.additional_info[key] = value

    def get_contributors(self) -> List[str]:
        """Return the contributors."""
        return self._contributors


def test_aggregate_with_valid_models() -> None:
    """Test the aggregation of valid models with correct expected parameters and control variates."""
    # Define two mock models with specific delta_y_i and delta_c_i
    model1 = MockP2PFLModel(
        params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        num_samples=10,
        additional_info={
            "scaffold": {
                "delta_y_i": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
                "delta_c_i": [np.array([0.01, 0.02]), np.array([0.03, 0.04])],
            }
        },
        contributors=["contributor1", "contributor2"],
    )
    model2 = MockP2PFLModel(
        params=[np.array([2.0, 3.0]), np.array([4.0, 5.0])],
        num_samples=20,
        additional_info={
            "scaffold": {
                "delta_y_i": [np.array([0.2, 0.3]), np.array([0.4, 0.5])],
                "delta_c_i": [np.array([0.02, 0.03]), np.array([0.04, 0.05])],
            }
        },
        contributors=["contributor1", "contributor2"],
    )

    raise ValueError("No se hasta que punto esto cumple su objetivo, no converge y está fallando")

    # Initialize the ScaffoldAggregator with a global learning rate
    aggregator = ScaffoldAggregator(node_name="test_node", global_lr=0.1)

    # Perform aggregation
    aggregated_model = aggregator.aggregate([model1, model2])

    # Calculate expected aggregated parameters based on the aggregation logic
    # For each layer:
    # aggregated_param = (sum(delta_y_i * num_samples) / total_samples) * global_lr

    total_samples = model1.get_num_samples() + model2.get_num_samples()

    # Layer 1
    delta_y_layer1_model1 = model1.get_info("delta_y_i")[0]  # [0.1, 0.2]
    delta_y_layer1_model2 = model2.get_info("delta_y_i")[0]  # [0.2, 0.3]
    accum_y_layer1 = (delta_y_layer1_model1 * model1.get_num_samples() + delta_y_layer1_model2 * model2.get_num_samples()) / total_samples
    expected_param_layer1 = accum_y_layer1 * aggregator.global_lr  # Multiply by global_lr

    # Layer 2
    delta_y_layer2_model1 = model1.get_info("delta_y_i")[1]  # [0.3, 0.4]
    delta_y_layer2_model2 = model2.get_info("delta_y_i")[1]  # [0.4, 0.5]
    accum_y_layer2 = (delta_y_layer2_model1 * model1.get_num_samples() + delta_y_layer2_model2 * model2.get_num_samples()) / total_samples
    expected_param_layer2 = accum_y_layer2 * aggregator.global_lr  # Multiply by global_lr

    expected_params = [expected_param_layer1, expected_param_layer2]

    # Assert that the aggregated parameters are close to the expected values
    for param, expected in zip(aggregated_model.get_parameters(), expected_params):
        assert np.allclose(param, expected), f"Expected {expected}, but got {param}"

    # Calculate expected global control variates
    # accum_c = sum(delta_c_i) / num_models
    accum_c_layer1 = (model1.get_info("delta_c_i")[0] + model2.get_info("delta_c_i")[0]) / 2
    accum_c_layer2 = (model1.get_info("delta_c_i")[1] + model2.get_info("delta_c_i")[1]) / 2
    expected_global_c = [accum_c_layer1, accum_c_layer2]

    # Assert that the aggregator's 'c' parameter is updated correctly
    for c, expected in zip(aggregator.c, expected_global_c):
        assert np.allclose(c, expected), f"Expected global_c {expected}, but got {c}"

    # Optionally, verify contributors
    expected_contributors = ["contributor1", "contributor2", "contributor1", "contributor2"]
    assert (
        aggregated_model.get_contributors() == expected_contributors
    ), f"Expected contributors {expected_contributors}, but got {aggregated_model.get_contributors()}"


def test_aggregate_with_no_models() -> None:
    """Test that aggregating with no models raises the appropriate error."""
    aggregator = ScaffoldAggregator(node_name="test_node")
    with pytest.raises(NoModelsToAggregateError):
        aggregator.aggregate([])


def test_aggregate_with_missing_required_info() -> None:
    """Test that aggregating a model missing required info keys raises a ValueError."""
    model = MockP2PFLModel(
        params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        num_samples=10,
        additional_info={
            "scaffold": {
                "delta_c_i": [np.array([0.02, 0.03]), np.array([0.04, 0.05])],
            }
        },
        contributors=["contributor1", "contributor2"],
    )

    aggregator = ScaffoldAggregator(node_name="test_node")
    with pytest.raises(ValueError):
        aggregator.aggregate([model])


def test_additional_parameters_communication() -> None:
    """Test that the aggregator correctly updates the global control variates 'c' based on the models' delta_c_i."""
    # Define two mock models with specific delta_c_i
    model1 = MockP2PFLModel(
        params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        num_samples=10,
        additional_info={
            "scaffold": {
                "delta_y_i": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
                "delta_c_i": [np.array([0.01, 0.02]), np.array([0.03, 0.04])],
            }
        },
        contributors=["contributor1", "contributor2"],
    )
    model2 = MockP2PFLModel(
        params=[np.array([2.0, 3.0]), np.array([4.0, 5.0])],
        num_samples=20,
        additional_info={
            "delta_y_i": [np.array([0.2, 0.3]), np.array([0.4, 0.5])],
            "delta_c_i": [np.array([0.02, 0.03]), np.array([0.04, 0.05])],
        },
        contributors=["contributor1", "contributor2"],
    )

    raise ValueError("No se hasta que punto esto cumple su objetivo, no converge y está fallando")

    # Initialize the aggregator
    aggregator = ScaffoldAggregator(node_name="test_node", global_lr=0.1)

    # Perform aggregation
    aggregated_model = aggregator.aggregate([model1, model2])

    # Calculate total samples
    total_samples = model1.get_num_samples() + model2.get_num_samples()

    # Calculate expected global control variate 'c'
    accum_c_layer1 = (model1.get_info("delta_c_i")[0] + model2.get_info("delta_c_i")[0]) / 2
    accum_c_layer2 = (model1.get_info("delta_c_i")[1] + model2.get_info("delta_c_i")[1]) / 2
    expected_global_c = [accum_c_layer1, accum_c_layer2]

    # Verify that aggregator's 'c' parameter is updated correctly
    for actual_c, expected_c in zip(aggregator.c, expected_global_c):
        assert np.allclose(actual_c, expected_c), f"Aggregator 'c' parameter not updated correctly. Expected {expected_c}, got {actual_c}"

    # Calculate expected aggregated parameters based on the aggregation logic
    # For each layer:
    # aggregated_param = (sum(delta_y_i * num_samples) / total_samples) * global_lr

    # Layer 1
    delta_y_layer1_model1 = model1.get_info("delta_y_i")[0]  # [0.1, 0.2]
    delta_y_layer1_model2 = model2.get_info("delta_y_i")[0]  # [0.2, 0.3]
    accum_y_layer1 = (delta_y_layer1_model1 * model1.get_num_samples() + delta_y_layer1_model2 * model2.get_num_samples()) / total_samples
    expected_param_layer1 = accum_y_layer1 * aggregator.global_lr

    # Layer 2
    delta_y_layer2_model1 = model1.get_info("delta_y_i")[1]  # [0.3, 0.4]
    delta_y_layer2_model2 = model2.get_info("delta_y_i")[1]  # [0.4, 0.5]
    accum_y_layer2 = (delta_y_layer2_model1 * model1.get_num_samples() + delta_y_layer2_model2 * model2.get_num_samples()) / total_samples
    expected_param_layer2 = accum_y_layer2 * aggregator.global_lr

    expected_params = [expected_param_layer1, expected_param_layer2]

    # Verify that aggregated model parameters are correct
    for actual_param, expected_param in zip(aggregated_model.get_parameters(), expected_params):
        assert np.allclose(actual_param, expected_param), (
            f"Aggregated model parameters not updated correctly. " f"Expected {expected_param}, got {actual_param}"
        )

    # Optionally, verify contributors
    expected_contributors = ["contributor1", "contributor2", "contributor1", "contributor2"]
    assert (
        aggregated_model.get_contributors() == expected_contributors
    ), f"Expected contributors {expected_contributors}, but got {aggregated_model.get_contributors()}"
