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

import numpy as np
import tensorflow as tf  # type: ignore
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
    ) -> tf.data.Dataset:
        """
        Export the data as a TensorFlow Dataset.

        Converts through numpy arrays to avoid HuggingFace's ``to_tf_dataset()``
        streaming pipeline, which deadlocks under asyncio when p2pfl's gossip
        and heartbeat tasks are active.

        Args:
            data: The Hugging Face Dataset to export. Transforms should already be applied to the dataset via set_transform.
            batch_size: The batch size for the TensorFlow Dataset.
            **kwargs: Additional keyword arguments.

        Returns:
            A TensorFlow Dataset.

        """
        if not batch_size:
            batch_size = Settings.training.DEFAULT_BATCH_SIZE

        keys = list(data[0].keys())
        feature_cols = keys[:-1]
        label_col = keys[-1]

        features = [np.array(data[col], dtype=np.float32) for col in feature_cols]
        x = features[0] if len(features) == 1 else np.concatenate(features, axis=-1)
        y = np.array(data[label_col], dtype=np.int64)

        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
