#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Keras dataset export strategy."""

from typing import Any

import numpy as np
from datasets import Dataset  # type: ignore

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy
from p2pfl.settings import Settings


class KerasExportStrategy(DataExportStrategy):
    """Export strategy for TensorFlow/Keras datasets."""

    @staticmethod
    def export(
        data: Dataset,
        batch_size: int | None = None,
        **kwargs,
    ) -> tuple[Any, Any, int]:
        """
        Export the data as numpy arrays for Keras.

        Returns a (features, labels, batch_size) tuple. Keras model.fit() and
        model.evaluate() accept numpy arrays directly, which avoids
        HuggingFace's to_tf_dataset() that deadlocks on macOS in Ray workers.

        Args:
            data: The Hugging Face Dataset to export.
            batch_size: The batch size for training/evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of (features, labels, batch_size).

        """
        if not batch_size:
            batch_size = Settings.training.DEFAULT_BATCH_SIZE

        # Get the columns
        columns = list(data[0].keys())[:-1]
        label_cols = list(data[0].keys())[-1:]

        # Convert to numpy — avoids to_tf_dataset() which deadlocks on macOS
        # in Ray workers due to TF internal thread pool conflicts.
        features = {col: np.array(data[col]) for col in columns}
        labels = {col: np.array(data[col]) for col in label_cols}

        # Unwrap single-column dicts
        if len(features) == 1:
            features = next(iter(features.values()))
        if len(labels) == 1:
            labels = next(iter(labels.values()))

        return features, labels, batch_size
