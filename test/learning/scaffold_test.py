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

"""
Unit tests for the ScaffoldAggregator.
"""

import pytest
import numpy as np
from p2pfl.learning.aggregators.scaffold import ScaffoldAggregator  # Adjust import as needed
from p2pfl.learning import P2PFLModel  # Adjust import as needed
from p2pfl.learning.aggregators.aggregator import NoModelsToAggregateError  # Adjust import as needed

class MockP2PFLModel(P2PFLModel):
    def __init__(self, params, _num_samples, additional_info=None):
        self._params = params
        self._num_samples = _num_samples
        self.additional_info = additional_info or {}
    
    def get_parameters(self):
        return self._params
    
    def get_num_samples(self):
        return self._num_samples
    
    def get_info(self, key):
        return self.additional_info[key]
    
    def build_copy(self, params, num_samples, contributors):
        return MockP2PFLModel(params, num_samples, self.additional_info)
    
    def add_info(self, key, value):
        self.additional_info[key] = value
    
    def get_contributors(self):
        return ["contributor1", "contributor2"]

def test_aggregate_with_valid_models():
    model1 = MockP2PFLModel(
        params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        num_samples=10,
        additional_info={'delta_y_i': [np.array([0.1, 0.2]), np.array([0.3, 0.4])], 'delta_c_i': [np.array([0.01, 0.02]), np.array([0.03, 0.04])]}
    )
    model2 = MockP2PFLModel(
        params=[np.array([2.0, 3.0]), np.array([4.0, 5.0])],
        num_samples=20,
        additional_info={'delta_y_i': [np.array([0.2, 0.3]), np.array([0.4, 0.5])], 'delta_c_i': [np.array([0.02, 0.03]), np.array([0.04, 0.05])]}
    )
    
    aggregator = ScaffoldAggregator(node_name="test_node", global_lr=0.1)
    aggregated_model = aggregator.aggregate([model1, model2])
    
    expected_params = [np.array([0.05, 0.08]), np.array([0.13, 0.18])]
    for param, expected in zip(aggregated_model.get_parameters(), expected_params):
        assert np.allclose(param, expected)
    
    expected_global_c = [np.array([0.015, 0.025]), np.array([0.035, 0.045])]
    for c, expected in zip(aggregator.c, expected_global_c):
        assert np.allclose(c, expected)

def test_aggregate_with_no_models():
    aggregator = ScaffoldAggregator(node_name="test_node")
    with pytest.raises(NoModelsToAggregateError):
        aggregator.aggregate([])

def test_aggregate_with_missing_required_info():
    model = MockP2PFLModel(
        params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        num_samples=10,
        additional_info={'delta_y_i': [np.array([0.1, 0.2]), np.array([0.3, 0.4])]}  # Missing 'delta_c_i'
    )
    
    aggregator = ScaffoldAggregator(node_name="test_node")
    with pytest.raises(ValueError):
        aggregator.aggregate([model])


def test_additional_parameters_communication():
    # Set up mock models with additional_info
    model1 = MockP2PFLModel(
        params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        _num_samples=10,
        additional_info={
            'delta_y_i': [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
            'delta_c_i': [np.array([0.01, 0.02]), np.array([0.03, 0.04])]
        }
    )
    model2 = MockP2PFLModel(
        params=[np.array([2.0, 3.0]), np.array([4.0, 5.0])],
        _num_samples=20,
        additional_info={
            'delta_y_i': [np.array([0.2, 0.3]), np.array([0.4, 0.5])],
            'delta_c_i': [np.array([0.02, 0.03]), np.array([0.04, 0.05])]
        }
    )

    # Initialize the aggregator
    aggregator = ScaffoldAggregator(node_name="test_node", global_lr=0.1)

    # Perform aggregation
    aggregated_model = aggregator.aggregate([model1, model2])

    # Calculate total samples
    total_samples = model1._num_samples + model2._num_samples

    # Calculate expected global control variate 'c'
    expected_global_c = [
        (model1.additional_info['delta_c_i'][0] * model1._num_samples + model2.additional_info['delta_c_i'][0] * model2._num_samples) / total_samples,
        (model1.additional_info['delta_c_i'][1] * model1._num_samples + model2.additional_info['delta_c_i'][1] * model2._num_samples) / total_samples
    ]

    # Verify that aggregator's 'c' parameter is updated correctly
    for actual_c, expected_c in zip(aggregator.c, expected_global_c):
        assert np.allclose(actual_c, expected_c), "Aggregator 'c' parameter not updated correctly."

    # Calculate expected aggregated parameters
    expected_params = [
        (model1.params[0] * model1._num_samples + model2.params[0] * model2._num_samples) / total_samples,
        (model1.params[1] * model1._num_samples + model2.params[1] * model2._num_samples) / total_samples
    ]

    # Verify that aggregated model parameters are correct
    for actual_param, expected_param in zip(aggregated_model.get_parameters(), expected_params):
        assert np.allclose(actual_param, expected_param), "Aggregated model parameters not updated correctly."