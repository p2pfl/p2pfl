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
from p2pfl.learning.compression.lzma_strategy import LZMACompressor
from p2pfl.learning.compression.manager import CompressionManager
from p2pfl.learning.compression.quantization_strategy import PTQuantization
from p2pfl.learning.compression.topk_strategy import TopKSparsification
from p2pfl.learning.compression.zlib_strategy import ZlibCompressor

####
# PTQuantization
####


@pytest.fixture
def ptq_compressor():
    """Return an instance of PTQuantization."""
    return PTQuantization()


@pytest.mark.parametrize(
    "shape",
    [
        (10,),  # 1D tensor
        (5, 5),  # 2D tensor
        (3, 4, 5),  # 3D tensor
    ],
)
def test_ptq_shapes(ptq_compressor, shape):
    """Test that quantization preserves shape."""
    original = np.random.randn(*shape).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="float16")
    assert quantized[0].shape == original.shape

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert dequantized[0].shape == original.shape


@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "float32",  # Float types
        "int8",
        "uint8",  # Integer types
    ],
)
def test_ptq_dtypes(ptq_compressor, dtype):
    """Test quantization with different dtypes."""
    original = np.random.randn(10, 10).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype=dtype)
    assert quantized[0].dtype == np.dtype(dtype)

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert dequantized[0].dtype == original.dtype


@pytest.mark.parametrize("scheme", ["symmetric", "asymmetric"])
def test_ptq_schemes(ptq_compressor, scheme):
    """Test different quantization schemes."""
    original = np.random.randn(10, 10).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8", scheme=scheme)
    assert info["ptq_scheme"] == scheme

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    # Check reconstruction quality (should be close but not identical)
    assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "granularity,channel_axis",
    [
        ("per_tensor", 0),
        ("per_channel", 0),
        ("per_channel", 1),
    ],
)
def test_ptq_granularity(ptq_compressor, granularity, channel_axis):
    """Test per-tensor and per-channel quantization."""
    original = np.random.randn(5, 5).astype(np.float32)
    params = [original]

    # Skip invalid combinations
    if granularity == "per_channel" and channel_axis >= original.ndim:
        pytest.skip("Invalid channel_axis for tensor shape")

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8", granularity=granularity, channel_axis=channel_axis)

    assert info["ptq_granularity"] == granularity

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "value_range",
    [
        (-1.0, 1.0),  # Symmetric around zero
        (0.0, 1.0),  # Positive only
        (-10.0, -5.0),  # Negative only
    ],
)
def test_ptq_value_ranges(ptq_compressor, value_range):
    """Test quantization with different value ranges."""
    min_val, max_val = value_range
    # Create tensor with specific range
    original = np.random.uniform(min_val, max_val, size=(10, 10)).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8")

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    # Check reconstruction quality
    assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


def test_ptq_multiple_tensors(ptq_compressor):
    """Test quantization of multiple tensors at once."""
    original1 = np.random.randn(5, 5).astype(np.float32)
    original2 = np.random.randn(3, 7).astype(np.float32)
    params = [original1, original2]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8")
    assert len(quantized) == 2

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert len(dequantized) == 2
    assert np.allclose(dequantized[0], original1, rtol=0.1, atol=0.1)
    assert np.allclose(dequantized[1], original2, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "dtype,scheme",
    [
        ("int8", "symmetric"),
        ("uint8", "asymmetric"),
    ],
)
def test_ptq_constant_tensors(ptq_compressor, dtype, scheme):
    """Test quantization of tensors with constant values."""
    # Test cases:
    constant_cases = [
        np.zeros((5, 5), dtype=np.float32),  # All zeros
        np.ones((5, 5), dtype=np.float32),  # All ones
        np.full((5, 5), 10.0, dtype=np.float32),  # All 10s
    ]

    for original in constant_cases:
        # Apply quantization
        quantized, info = ptq_compressor.apply_strategy([original], dtype=dtype, scheme=scheme)

        # Test dequantization
        dequantized = ptq_compressor.reverse_strategy(quantized, info)

        # For constant tensors, we should get exactly the same value back
        # (within a small tolerance due to rounding)
        assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "invalid_param",
    [
        {"dtype": "complex64"},  # Invalid dtype
        {"scheme": "invalid_scheme"},  # Invalid scheme
        {"granularity": "invalid_granularity"},  # Invalid granularity
    ],
)
def test_ptq_invalid_params(ptq_compressor, invalid_param):
    """Test that invalid parameters raise appropriate errors."""
    original = np.random.randn(5, 5).astype(np.float32)

    with pytest.raises(ValueError):
        kwargs = {"dtype": "int8"}
        kwargs.update(invalid_param)
        ptq_compressor.apply_strategy([original], **kwargs)


def test_ptq_empty_params(ptq_compressor):
    """Test handling of empty parameter list."""
    with pytest.raises(ValueError):
        ptq_compressor.apply_strategy([], dtype="float16")


def test_ptq_missing_info(ptq_compressor):
    """Test error handling when required info keys are missing."""
    original = np.random.randn(5, 5).astype(np.float32)
    quantized, info = ptq_compressor.apply_strategy([original], dtype="int8")

    # Remove a required key
    bad_info = info.copy()
    del bad_info["ptq_scales"]

    with pytest.raises(ValueError):
        ptq_compressor.reverse_strategy(quantized, bad_info)


def test_ptq_compression_ratio(ptq_compressor):
    """Test compression ratio for float32 to int8 quantization."""
    original = np.random.randn(100, 100).astype(np.float32)

    # Measure original size
    original_bytes = original.nbytes

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy([original], dtype="int8")
    quantized_bytes = quantized[0].nbytes

    # Calculate compression ratio
    compression_ratio = original_bytes / quantized_bytes

    # Expected ratio for float32 to int8 should be around 4
    assert 3.5 <= compression_ratio <= 4.5


###
# TopK
###


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
    for orig, decomp in zip(original_params, decompressed_parameters, strict=False):
        assert orig.shape == decomp.shape, "Decompressed shape does not match original"


###
# LoRa
###


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
    for orig, decomp in zip(original_params, decompressed_parameters, strict=False):
        if orig.ndim == 2:
            # relative error to compressed layers, expected ~= 1 - threshold
            energy_total = np.sum(np.linalg.svd(orig, full_matrices=False)[1] ** 2)
            error = np.linalg.norm(orig - decomp, ord="fro") ** 2 / energy_total
            assert error <= (1 - threshold + tol), f"Relative error {error:.3f} exceeds allowed limit for threshold {threshold}"
        else:
            np.testing.assert_array_equal(orig, decomp, err_msg="Non-compressed layer has changed.")


###
# ZLIB
###


@pytest.mark.parametrize("level", [1, 5])
def test_zlib(level: int):
    """
    Test Zlib compression algorithm.

    Args:
        level: zlib level of compression.

    """
    technique = ZlibCompressor()
    original_bytes = pickle.dumps("LUIS PERUANO UUUUUUUUUUUUUUUUUUUUUUUUUU!!!!!!! Y HECTOR NO HACE NADA")
    compressed_bytes = technique.apply_strategy(original_bytes, level=level)
    assert len(original_bytes) > len(compressed_bytes), "compression resulted in more bytes than the original model"
    decompressed_bytes = technique.reverse_strategy(compressed_bytes)
    assert decompressed_bytes == original_bytes


@pytest.mark.parametrize("preset", [1, 5, 9])
def test_lzma(preset: int):
    """
    Test Zlib compression algorithm.

    Args:
        preset: LZMA level of compression.

    """
    technique = LZMACompressor()
    original_bytes = pickle.dumps("ABC " * 1000)
    compressed_bytes = technique.apply_strategy(original_bytes, preset=preset)
    assert len(original_bytes) > len(compressed_bytes), "compression resulted in more bytes than the original model"
    decompressed_bytes = technique.reverse_strategy(compressed_bytes)
    assert decompressed_bytes == original_bytes


###
# Manager tests
###


@pytest.fixture
def compression_manager() -> CompressionManager:
    """Fixture to create a new compression manager instance."""
    return CompressionManager()


def test_manager_multiple_byte_compressors(compression_manager: CompressionManager):
    """Test that only one byte compressor is allowed."""
    techniques = {"zlib": {"level": 5}, "lzma": {"preset": 5}}
    with pytest.raises(ValueError):
        _ = compression_manager.apply([np.random.randn(10, 10) for i in range(3)], {"dummy": "info"}, techniques)


def test_manager_unknown_strategy(compression_manager: CompressionManager):
    """Test that an unknown strategy raises an error."""
    techniques = {"unknown": {}}
    with pytest.raises(ValueError):
        _ = compression_manager.apply([np.random.randn(10, 10) for i in range(3)], {"dummy": "info"}, techniques)


def test_manager_no_techniques(compression_manager: CompressionManager):
    """Test that an empty dictionary of techniques raises an error."""
    original_params = [np.random.randn(10, 10) for i in range(3)]
    original_add_info = {}
    compressed_data = compression_manager.apply(original_params, original_add_info, {})
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] is None
    assert np.array_equal(pickle.loads(deserialized_data["bytes"])["params"], original_params)
    assert pickle.loads(deserialized_data["bytes"])["additional_info"]["applied_techniques"] == []
    decompressed_params, decompressed_add_info = compression_manager.reverse(compressed_data)
    assert np.array_equal(original_params, decompressed_params)
    assert decompressed_add_info == original_add_info


def test_manager_only_compressor(compression_manager: CompressionManager):
    """Test only compressor (loseless)."""
    original_params = [np.random.randn(10, 10) for i in range(3)]

    techniques = {"zlib": {"level : 5"}}
    compressed_data = compression_manager.apply(original_params, {}, techniques)
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] == "zlib"

    decompressed_bytes = zlib.decompress(deserialized_data["bytes"])
    assert "params" in pickle.loads(decompressed_bytes)
    assert "additional_info" in pickle.loads(decompressed_bytes)
    assert "applied_techniques" in pickle.loads(decompressed_bytes)["additional_info"]
    assert len(pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"]) == 0
    decompressed_params, _ = compression_manager.reverse(compressed_data)
    assert np.allclose(original_params[0], decompressed_params[0], atol=1e-2)


def test_manager_multiple_techniques(compression_manager: CompressionManager):
    """Test the manager with multiple techniques."""
    original_params = [np.random.randn(10, 10) for i in range(3)]
    techniques = {
        "topk": {"k": 0.5},
        "zlib": {"level": 5},
        "low_rank": {"threshold": 0.7},
    }

    compressed_data = compression_manager.apply(original_params, {}, techniques)
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] == "zlib"

    decompressed_bytes = zlib.decompress(deserialized_data["bytes"])
    assert "params" in pickle.loads(decompressed_bytes)
    assert "additional_info" in pickle.loads(decompressed_bytes)
    assert "applied_techniques" in pickle.loads(decompressed_bytes)["additional_info"]
    assert len(pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"]) == 2
    assert pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"][0][0] == "topk"
    assert pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"][1][0] == "low_rank"

    decompressed_params, decompressed_info = compression_manager.reverse(compressed_data)
    for orig, decomp in zip(original_params, decompressed_params, strict=False):
        assert orig.shape == decomp.shape
    assert len(decompressed_params) == len(original_params)


def test_additional_info_preservation(compression_manager: CompressionManager):
    """Test that techniques don't remove additional info from other processes."""
    original_params = [np.random.randn(10, 10)]
    additional_info = {"test_key": "test_value"}
    registry = compression_manager.get_registry()

    assert registry, "The compression registry should not be empty."
    for technique_name in registry:
        compressed_data = compression_manager.apply(original_params, additional_info, {technique_name: {}})
        _, decompressed_info = compression_manager.reverse(compressed_data)
        assert decompressed_info == additional_info, f"Additional info lost for technique '{technique_name}'"
