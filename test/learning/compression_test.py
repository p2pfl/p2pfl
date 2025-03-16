import copy

import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionTechnique
from p2pfl.learning.compressors.low_rank import LowRankApproximation
from p2pfl.learning.compressors.quantization import PTQuantization
from p2pfl.learning.compressors.topk_sparsification import TopKSparsification
from p2pfl.learning.compressors.zlib import ZlibCompressor


def test_quantization(payload: dict, dtype: np.dtype):
    """
    Test the post-training quantization compression technique.

    Args:
        payload: The input payload.
        dtype: The desired data type for the compressed parameters.

    """
    technique = PTQuantization()
    proc_payload = technique.apply_strategy(copy.deepcopy(payload),
                                            dtype=dtype)
    assert proc_payload[0].dtype == dtype
    reversed_payload = technique.reverse_strategy(proc_payload)
    assert reversed_payload.dtype == payload["params"].dtype

def test_topk_sparsification(sample_payload: dict, sample_k: float):
    """
    Test the TopK sparsification strategy.

    Args:
        sample_payload: Payload to test.
        sample_k : k to test.

    """
    technique = TopKSparsification()

    total_original_size = sum(layer.size for layer in sample_payload["params"])

    compressed = technique.apply_strategy(
        sample_payload.copy(),
        k = sample_k
    )
    total_compressed_size = sum(layer.size for layer in compressed["params"])
    assert "topk_sparse_metadata" in compressed["additional_info"], "Missing metadata on compressed model"
    assert total_compressed_size <= total_original_size, "Compression resulted in more parameters than the original model"
    if sample_k != 1.0:
        assert total_compressed_size < total_original_size, "Compression did not remove any parameters"

    decompressed = technique.reverse_strategy(compressed)
    assert "topk_sparse_metadata" not in decompressed["additional_info"], "Metadata not removed correctly after reversing the strategy"
    total_decompressed_size = sum(layer.size for layer in decompressed["params"])
    assert total_decompressed_size == total_original_size
    for orig, decomp in zip(sample_payload["params"], decompressed["params"]):
        assert orig.shape == decomp.shape, "Decompressed shape does not match original"

def test_zlib(binary_payload: bytes, level:int=1):
    """
    Test Zlib compression algorithm.

    Args:
        binary_payload: Payload to test.
        level: zlib level of compression.

    """
    technique = ZlibCompressor()
    compressed = technique.apply_strategy(copy.copy(binary_payload), level)
    assert len(compressed) <= len(binary_payload), "Compression resulted in more bytes than the original model"
    decompressed = technique.reverse_strategy(compressed)
    assert compressed == decompressed

def test_lowrank(payload: dict, threshold: float):
    """
    Test LowRank compression algorithm.

    Args:
        payload: Payload to test.
        threshold: Percentage between 0 and 1 of the energy to preserve.

    """
    technique = LowRankApproximation()
    compressed_payload = technique.apply_strategy(copy.deepcopy(payload), threshold=threshold)
    assert "compressed_state" in compressed_payload["additional_info"], "Falta la metadata de compresión"

    decompressed_params = technique.reverse_strategy(copy.deepcopy(compressed_payload))
    total_original = sum(layer.size for layer in payload["params"])
    total_decompressed = sum(layer.size for layer in decompressed_params)
    assert total_original == total_decompressed, "Number of elements not matching after reverse strategy."

    for orig, decomp in zip(payload["params"], decompressed_params):
        if orig.ndim == 2:
            # relative error to compressed layers, expected ~= 1 - threshold
            energy_total = np.sum(np.linalg.svd(orig, full_matrices=False)[1] ** 2)
            error = np.linalg.norm(orig - decomp, ord='fro')**2 / energy_total
            np.testing.assert_allclose(error, 1 - 0.95, rtol=0.05, err_msg="Relative error about tolerance for compressed layer")
        else:
            np.testing.assert_array_equal(orig, decomp, err_msg="Not compressed layer has changed during the process.")

    assert "lowrank_compressed_state" not in compressed_payload["additional_info"], "Strategy metadata still present."

def test_strategies():
    pass

def test_manager():
    pass