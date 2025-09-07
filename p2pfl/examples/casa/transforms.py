#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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

"""
CASA transforms for P2PFL datasets.

This module contains transformation functions for CASA datasets.
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def casa_transforms(examples):
    """Transform CASA dataset columns to features and labels arrays."""
    # Extract feature columns (columns 0-35)
    feature_cols = [str(i) for i in range(36)]
    features = []
    for col in feature_cols:
        if col in examples:
            features.append(examples[col])

    # Stack features and transpose to get shape (batch_size, n_features)
    X = np.array(features).T

    # Reshape for LSTM input: add time dimension (batch_size, 1, n_features)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Extract label columns (columns 36-45)
    label_cols = [str(i) for i in range(36, 46)]
    labels = []
    for col in label_cols:
        if col in examples:
            labels.append(examples[col])

    # Stack labels and transpose to get shape (batch_size, n_labels)
    Y = np.array(labels).T

    # Convert one-hot encoded labels to sparse format (class indices)
    # If labels are one-hot encoded, convert to class indices
    if Y.shape[1] == 10:  # Check if it's one-hot encoded (10 classes)
        Y = np.argmax(Y, axis=1)

    return {"features": X, "labels": Y}


def get_casa_transforms() -> dict[str, Callable[[Any], dict[str, Any]]]:
    """Export transforms for CASA dataset."""
    return {"train": casa_transforms, "test": casa_transforms}
