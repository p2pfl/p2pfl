"""
Package for the learning frameworks in P2PFL.

This package contains all the integrations with the different learning frameworks.
"""

from enum import Enum

###
#   Framework Enum
###


class Framework(Enum):
    """Enum for the different learning frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    FLAX = "flax"
