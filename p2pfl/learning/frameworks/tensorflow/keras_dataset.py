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

        Args:
            data: The Hugging Face Dataset to export. Transforms should already be applied to the dataset via set_transform.
            batch_size: The batch size for the TensorFlow Dataset.
            seed: The seed for the TensorFlow Dataset.
            **kwargs: Additional keyword arguments.

        Returns:
            A TensorFlow Dataset.

        """
        if not batch_size:
            batch_size = Settings.training.DEFAULT_BATCH_SIZE

        # Get the columns
        columns = list(data[0].keys())[:-1]
        label_cols = list(data[0].keys())[-1:]

        print(
            f"Getting columns by order: {columns}, label_cols: {label_cols}. "
            "If need different ones, implement your own KerasExportStrategy or a custom transform."
        )

        # Export Keras dataset
        return data.to_tf_dataset(
            batch_size=batch_size,
            columns=columns,
            label_cols=label_cols,
        )
