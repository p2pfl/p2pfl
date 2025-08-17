"""Aggregation algorithms for P2PFL."""

# Import FedOpt family from subfolder for backward compatibility
from p2pfl.learning.aggregators.fedopt import FedAdagrad, FedAdam, FedOptBase, FedYogi

__all__ = ["FedOptBase", "FedAdagrad", "FedAdam", "FedYogi"]
