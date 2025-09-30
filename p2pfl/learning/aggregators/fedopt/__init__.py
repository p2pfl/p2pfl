"""
FedOpt family of aggregators for P2PFL.

This package provides implementations of the FedOpt family of federated optimization algorithms.
See individual modules for detailed documentation.
"""

from p2pfl.learning.aggregators.fedopt.base import FedOptBase  # noqa: F401
from p2pfl.learning.aggregators.fedopt.fedadagrad import FedAdagrad  # noqa: F401
from p2pfl.learning.aggregators.fedopt.fedadam import FedAdam  # noqa: F401
from p2pfl.learning.aggregators.fedopt.fedyogi import FedYogi  # noqa: F401

__all__ = ["FedOptBase", "FedAdagrad", "FedAdam", "FedYogi"]
