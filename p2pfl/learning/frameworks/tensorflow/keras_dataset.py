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
from datasets import Dataset  # type: ignore

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy


class KerasExportStrategy(DataExportStrategy):
    """Export strategy for TensorFlow/Keras datasets."""

    @staticmethod
    def export(
        data: Dataset,
        _batch_size: int | None = None,
        **_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Export as ``(x, y)`` numpy arrays — avoids pyarrow/TF import-order deadlock in Keras 3."""
        keys = list(data[0].keys())
        feature_cols = keys[:-1]
        label_col = keys[-1]

        features = [np.array(data[col], dtype=np.float32) for col in feature_cols]
        x = features[0] if len(features) == 1 else np.concatenate(features, axis=-1)
        y = np.array(data[label_col], dtype=np.int64)

        return x, y
