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

"""Tests for compression module."""

import pickle
import zlib

import numpy as np
import pytest

from p2pfl.learning.compression.lra_strategy import LowRankApproximation
from p2pfl.learning.compression.manager import CompressionManager
from p2pfl.learning.compression.quantization_strategy import PTQuantization
from p2pfl.learning.compression.topk_strategy import TopKSparsification
from p2pfl.learning.compression.zlib_strategy import ZlibCompressor

###
# Test Strategies
###


@pytest.mark.parametrize("dtype", [np.float16, np.int8])
def test_quantization(dtype: np.dtype):
    """
    Test the post-training quantization compression technique.

    Args:
        dtype: The desired data type for the compressed parameters.

    """
    technique = PTQuantization()
    original_params = [np.random.randn(10, 10) for i in range(3)]

    compressed_parameters, technique_params = technique.apply_strategy(original_params, dtype=dtype)
    assert compressed_parameters[0].dtype == dtype

    decompressed_parameters = technique.reverse_strategy(compressed_parameters, technique_params)
    assert decompressed_parameters[0].dtype == original_params[0].dtype
    assert decompressed_parameters[0].shape == original_params[0].shape


@pytest.mark.parametrize("sample_k", [0.1, 0.5, 1.0])
def test_topk_sparsification(sample_k: float):
    """
    Test the TopK sparsification strategy.

    Args:
        dummy_payload: Payload to test.
        sample_k : k to test.

    """
    technique = TopKSparsification()
    original_params = [np.random.randn(10, 10) for i in range(3)]
    total_original_size = sum(layer.size for layer in original_params)

    compressed_parameters, technique_params = technique.apply_strategy(original_params, k=sample_k)
    total_compressed_size = sum(layer.size for layer in compressed_parameters)
    assert "topk_sparse_metadata" in technique_params, "Missing metadata on compressed model"
    assert total_compressed_size <= total_original_size, "compression resulted in more parameters than the original model"
    if sample_k != 1.0:
        assert total_compressed_size < total_original_size, "compression did not remove any parameters"

    decompressed_parameters = technique.reverse_strategy(compressed_parameters, technique_params)
    total_decompressed_size = sum(layer.size for layer in decompressed_parameters)
    assert total_decompressed_size == total_original_size
    for orig, decomp in zip(original_params, decompressed_parameters):
        assert orig.shape == decomp.shape, "Decompressed shape does not match original"


@pytest.mark.parametrize("level", [1, 5])
def test_zlib(level: int):
    """
    Test Zlib compression algorithm.

    Args:
        dummy_binary_payload: Payload to test.
        level: zlib level of compression.

    """
    technique = ZlibCompressor()
    original_bytes = pickle.dumps("LUIS PERUANO UUUUUUUUUUUUUUUUUUUUUUU!!!!!!! Y HECTOR NO HACE NADA")
    compressed_bytes = technique.apply_strategy(original_bytes, level=level)
    assert len(original_bytes) > len(compressed_bytes), "compression resulted in more bytes than the original model"
    decompressed_bytes = technique.reverse_strategy(compressed_bytes)
    assert decompressed_bytes == original_bytes


@pytest.mark.parametrize("threshold", [0.5, 0.7])
def test_lowrank(threshold: float):
    """
    Test LowRank compression algorithm.

    Args:
        dummy_payload: Payload to test.
        threshold: Percentage between 0 and 1 of the energy to preserve.

    """
    technique = LowRankApproximation()
    original_params = [np.random.randn(10, 10) for i in range(3)]
    compressed_parameters, technique_params = technique.apply_strategy(original_params, threshold=threshold)
    assert "lowrank_compressed_state" in technique_params, "Missing compression metadata"
    assert sum(layer.size for layer in compressed_parameters) < sum(
        layer.size for layer in original_params
    ), "compression resulted in more parameters than the original model"

    decompressed_parameters = technique.reverse_strategy(compressed_parameters, technique_params)
    total_original = sum(layer.size for layer in original_params)
    total_decompressed = sum(layer.size for layer in decompressed_parameters)
    assert total_original == total_decompressed, "Number of elements not matching after reverse strategy."

    tol = 0.05
    for orig, decomp in zip(original_params, decompressed_parameters):
        if orig.ndim == 2:
            # relative error to compressed layers, expected ~= 1 - threshold
            energy_total = np.sum(np.linalg.svd(orig, full_matrices=False)[1] ** 2)
            error = np.linalg.norm(orig - decomp, ord="fro") ** 2 / energy_total
            assert error <= (1 - threshold + tol), f"Relative error {error:.3f} exceeds allowed limit for threshold {threshold}"
        else:
            np.testing.assert_array_equal(orig, decomp, err_msg="Non-compressed layer has changed.")


###
# Manager test
###


def test_manager():
    """Test the compression manager."""
    manager = CompressionManager()
    original_params = [np.random.randn(10, 10) for i in range(3)]
    original_add_info = {"dummy": "info"}

    # Test no techniques
    compressed_data = manager.apply(original_params, original_add_info, {})
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] is None
    assert np.array_equal(pickle.loads(deserialized_data["bytes"])["params"], original_params)
    assert pickle.loads(deserialized_data["bytes"])["additional_info"]["applied_techniques"] == []
    decompressed_params, decompressed_add_info = manager.reverse(compressed_data)
    assert np.array_equal(original_params, decompressed_params)
    assert decompressed_add_info == original_add_info

    # Test with techniques
    # (use different techniques and parameters, check that they are applied in order and ensure that the results are close to the original)
    techniques = {
        "topk": {"k": 0.5},
        "zlib": {"level": 5},
        "low_rank": {"threshold": 0.7},
    }
    compressed_data = manager.apply(original_params, {}, techniques)
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] == "zlib"
    decompressed_bytes = zlib.decompress(deserialized_data["bytes"])
    assert "params" in pickle.loads(decompressed_bytes)
    assert "additional_info" in pickle.loads(decompressed_bytes)
    assert "applied_techniques" in pickle.loads(decompressed_bytes)["additional_info"]
    assert len(pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"]) == 2
    assert pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"][0][0] == "topk"
    assert pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"][1][0] == "low_rank"
    # decompressed_params, _ = manager.reverse(compressed_data)
    # assert np.allclose(original_params[0], decompressed_params[0], atol=1e-2)
    # TODO: Check cast to float16 and int8 and ensure that the allclose is on all the tests
