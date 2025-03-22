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

import copy
import pickle

import numpy as np
import pytest

from p2pfl.learning.compression.lra_strategy import LowRankApproximation
from p2pfl.learning.compression.manager import CompressionManager
from p2pfl.learning.compression.quantization_strategy import PTQuantization
from p2pfl.learning.compression.topk_strategy import TopKSparsification
from p2pfl.learning.compression.zlib_strategy import ZlibCompressor


@pytest.fixture
def dummy_data():
    """Create dummy data to test."""
    return {
        "header" : {},
        "payload": {
            "params": [np.random.randn(10,10) for i in range(3)],
            "additional_info" : {}
        }
    }

@pytest.fixture
def dummy_payload():
    """Create dummy payload to test."""
    return {
            "params": [np.random.randn(10,10) for i in range(3)],
            "additional_info" : {}
        }

@pytest.fixture
def dummy_binary_payload():
    """Create dummy binray payload to test."""
    params = [np.random.randn(10,10) for i in range(3)]
    params_binary = pickle.dumps(params)
    return {
        "params": params_binary,
        "additional_info" : {}
    }

# =====
# Test Strategies
# =====

@pytest.mark.parametrize("dtype", [np.float16, np.int8])
def test_quantization(dummy_payload: dict, dtype: np.dtype):
    """
    Test the post-training quantization compression technique.

    Args:
        dummy_payload: The input payload.
        dtype: The desired data type for the compressed parameters.

    """
    technique = PTQuantization()
    proc_payload = technique.apply_strategy(copy.deepcopy(dummy_payload),
                                            dtype=dtype)
    assert proc_payload["params"][0].dtype == dtype
    reversed_payload = technique.reverse_strategy(proc_payload)
    assert reversed_payload["params"][0].dtype == dummy_payload["params"][0].dtype
    # TODO: check equal shapes

@pytest.mark.parametrize("sample_k", [0.1, 0.5, 1.0])
def test_topk_sparsification(dummy_payload: dict, sample_k: float):
    """
    Test the TopK sparsification strategy.

    Args:
        dummy_payload: Payload to test.
        sample_k : k to test.

    """
    technique = TopKSparsification()

    total_original_size = sum(layer.size for layer in dummy_payload["params"])

    compressed = technique.apply_strategy(
        dummy_payload.copy(),
        k = sample_k
    )
    total_compressed_size = sum(layer.size for layer in compressed["params"])
    assert "topk_sparse_metadata" in compressed["additional_info"], "Missing metadata on compressed model"
    assert total_compressed_size <= total_original_size, "compression resulted in more parameters than the original model"
    if sample_k != 1.0:
        assert total_compressed_size < total_original_size, "compression did not remove any parameters"

    decompressed = technique.reverse_strategy(compressed)
    assert "topk_sparse_metadata" not in decompressed["additional_info"], "Metadata not removed correctly after reversing the strategy"
    total_decompressed_size = sum(layer.size for layer in decompressed["params"])
    assert total_decompressed_size == total_original_size
    for orig, decomp in zip(dummy_payload["params"], decompressed["params"]):
        assert orig.shape == decomp.shape, "Decompressed shape does not match original"

@pytest.mark.parametrize("level", [1, 5])
def test_zlib(dummy_binary_payload: bytes, level:int):
    """
    Test Zlib compression algorithm.

    Args:
        dummy_binary_payload: Payload to test.
        level: zlib level of compression.

    """
    technique = ZlibCompressor()
    compressed = technique.apply_strategy(copy.copy(dummy_binary_payload), level)
    assert len(compressed) <= len(dummy_binary_payload), "compression resulted in more bytes than the original model"
    decompressed = technique.reverse_strategy(compressed)
    assert compressed == decompressed

@pytest.mark.parametrize("threshold", [0.5, 0.7])
def test_lowrank(dummy_payload: dict, threshold: float):
    """
    Test LowRank compression algorithm.

    Args:
        dummy_payload: Payload to test.
        threshold: Percentage between 0 and 1 of the energy to preserve.

    """
    technique = LowRankApproximation()
    compressed_payload = technique.apply_strategy(copy.deepcopy(dummy_payload), threshold=threshold)
    assert "lowrank_compressed_state" in compressed_payload["additional_info"], "Missing compression metadata"

    decompressed_payload = technique.reverse_strategy(copy.deepcopy(compressed_payload))
    total_original = sum(layer.size for layer in dummy_payload["params"])
    total_decompressed = sum(layer.size for layer in decompressed_payload["params"])
    assert total_original == total_decompressed, "Number of elements not matching after reverse strategy."

    tol = 0.05
    for orig, decomp in zip(dummy_payload["params"], decompressed_payload["params"]):
        if orig.ndim == 2:
            # relative error to compressed layers, expected ~= 1 - threshold
            energy_total = np.sum(np.linalg.svd(orig, full_matrices=False)[1] ** 2)
            error = np.linalg.norm(orig - decomp, ord='fro')**2 / energy_total
            assert error <= (1 - threshold + tol), (
                f"Relative error {error:.3f} exceeds allowed limit for threshold {threshold}"
            )
        else:
            np.testing.assert_array_equal(orig, decomp, err_msg="Non-compressed layer has changed.")

    assert "lowrank_compressed_state" not in decompressed_payload["additional_info"], "Strategy metadata still present."


# =====
# Manager test
# =====orig.shape == decomp.shape,
class DummyStrategy:
    """Dummy compression strategy for testing."""

    def apply_strategy(self, payload, **kwargs):
        """Apply strategy."""
        payload["additional_info"]["dummy_applied"] = True
        return payload

    def reverse_strategy(self, payload):
        """Reverse strategy."""
        payload["additional_info"].pop("dummy_applied", None)
        return payload

    def get_category(self):
        """Get category."""
        return "compressor"


def test_manager(monkeypatch, dummy_data):
    """Test the compression manager."""
    manager = CompressionManager()

    # dummy registry and monkeypatching manager with it
    dummy_registry = {
        "dummy": lambda: DummyStrategy()
    }
    monkeypatch.setattr(
        target=CompressionManager,
        name="get_registry",
        value= lambda: dummy_registry)

    # check with a dummy technique that compressor works
    techniques = {"dummy": {}}
    compressed_data = manager.apply(
        data= dummy_data.copy(),
        techniques=techniques
        )
    deserialized_data = pickle.loads(compressed_data)

    assert "dummy" in deserialized_data["header"]["applied_techniques"]
    payload_deserialized = pickle.loads(deserialized_data["payload"])
    assert "dummy_applied" in payload_deserialized["additional_info"]

    decompressed_payload = manager.reverse(data=deserialized_data)
    assert "dummy_applied" not in decompressed_payload["additional_info"]

    original_params = dummy_data["payload"]["params"]
    for orig, recov in zip(original_params, decompressed_payload["params"]):
        np.testing.assert_equal(orig.shape, recov.shape)
