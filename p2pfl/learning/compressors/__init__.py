"""P2PFL compressions."""
from .quantization import PTQuantization
from .topk_sparsification import TopKSparsification
from .zlib import ZlibCompressor

COMPRESSION_REGISTRY = {
    "ptq": PTQuantization,
    "zlib": ZlibCompressor,
    "topk": TopKSparsification,
}

# dynamically generated bitmask :)
COMPRESSION_BITMASK = {name: 1 << i for i, name in enumerate(COMPRESSION_REGISTRY)}

# COMPRESSION_BITMASK = {
#     "ptq": 0b00000001,
#     "zlib": 0b00000010,
#     "topk": 0b00000100,
# }
