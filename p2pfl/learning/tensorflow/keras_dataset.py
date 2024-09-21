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

from typing import Callable, List, Optional

import tensorflow as tf
from datasets import Dataset  # type: ignore

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy


class KerasExportStrategy(DataExportStrategy):
    """Export strategy for TensorFlow/Keras datasets."""

    @staticmethod
    def export(
        data: Dataset,
        transforms: Optional[Callable] = None,
        batch_size: int = 1,
        columns: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> tf.data.Dataset:
        """
        Export the data as a TensorFlow Dataset.

        Args:
            data: The Hugging Face Dataset to export.
            transforms: Optional transformations to apply (not implemented yet).
            batch_size: The batch size for the TensorFlow Dataset.
            seed: The seed for the TensorFlow Dataset.
            columns: The columns to include in the TensorFlow Dataset.
            label_cols: The columns to use as labels.
            **kwargs: Additional keyword arguments.

        Returns:
            A TensorFlow Dataset.

        """
        if label_cols is None:
            label_cols = ["label"]
        if columns is None:
            columns = ["image"]
        if transforms is not None:
            raise NotImplementedError("Transforms are not yet supported for KerasExportStrategy.")

        # Export Keras dataset
        return data.to_tf_dataset(
            batch_size=batch_size,
            columns=columns,
            label_cols=label_cols,
        )
