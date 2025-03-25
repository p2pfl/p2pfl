"""P2PFL compressions."""

from .lra_strategy import LowRankApproximation
from .quantization_strategy import PTQuantization
from .topk_strategy import TopKSparsification
from .zlib_strategy import ZlibCompressor

# All strategies need to be registered for the manager.
COMPRESSION_STRATEGIES_REGISTRY = {
    "ptq": PTQuantization,
    "topk": TopKSparsification,
    "low_rank": LowRankApproximation,
    "zlib": ZlibCompressor,
}
