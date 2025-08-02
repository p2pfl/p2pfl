#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Utils."""

import random

import numpy as np

from p2pfl.learning.frameworks import Framework

"""
Module to set seeds of all components of the p2pfl system.
"""


def set_seed(seed: int | None = 666, framework: Framework | str | None = None) -> None:
    """
    Set the seed for random number generators in Python's `random`, NumPy, PyTorch, JAX, and TensorFlow.

    Handles cases where PyTorch, JAX, or TensorFlow might not be installed.

    Args:
        seed: The seed value to use. Defaults to 666.
        framework: The framework to seed. Defaults to all frameworks. Options: pytorch, flax, tensorflow.

    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    if isinstance(framework, str):
        framework = Framework[framework.upper()]

    if framework is None or framework == Framework.PYTORCH:
        try:
            import torch

            # PyTorch seeding
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
        except ImportError:
            print("PyTorch not found. Skipping PyTorch seeding.")

    if framework is None or framework == Framework.FLAX:
        try:
            import jax
            import jax.random

            # JAX seeding
            jax.random.PRNGKey(seed)
            print("Warning: JAX seeding may not be fully supported.")
        except ImportError:
            print("JAX not found. Skipping JAX seeding.")

    if framework is None or framework == Framework.TENSORFLOW:
        try:
            import tensorflow as tf

            # TensorFlow seeding
            tf.random.set_seed(seed)
        except ImportError:
            print("TensorFlow not found. Skipping TensorFlow seeding.")
