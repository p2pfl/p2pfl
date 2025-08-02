"""P2PFL compressions."""

from .dp_strategy import DifferentialPrivacyCompressor, LocalDPCompressor
from .lra_strategy import LowRankApproximation
from .lzma_strategy import LZMACompressor
from .quantization_strategy import PTQuantization
from .topk_strategy import TopKSparsification
from .zlib_strategy import ZlibCompressor

# All strategies need to be registered for the manager.
COMPRESSION_STRATEGIES_REGISTRY = {
    "ptq": PTQuantization,
    "topk": TopKSparsification,
    "low_rank": LowRankApproximation,
    "zlib": ZlibCompressor,
    "lzma": LZMACompressor,
    "dp": DifferentialPrivacyCompressor,
    "local_dp": LocalDPCompressor,
}
