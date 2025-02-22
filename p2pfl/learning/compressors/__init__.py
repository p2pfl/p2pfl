"""P2PFL compressions."""
from .low_rank import LowRankApproximation
from .quantization import PTQuantization
from .topk_sparsification import TopKSparsification
from .zlib import ZlibCompressor

COMPRESSION_REGISTRY = {
    "ptq": PTQuantization,
    "zlib": ZlibCompressor,
    "topk": TopKSparsification,
    "low_rank": LowRankApproximation
}
