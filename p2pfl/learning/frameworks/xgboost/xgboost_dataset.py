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

"""XGBoost DMatrix export integration."""


import numpy as np
from datasets import Dataset

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy


class XGBoostExportStrategy(DataExportStrategy):
    """Export strategy for XGBoost."""

    @staticmethod
    def export(
        data: Dataset, train: bool = True, label_key: str = None, feature_keys: list[str] | None = None, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Export dataset to numpy arrays for XGBoost.

        Args:
            data: The dataset to export.
            train: Whether this is training data.
            label_key: The key for the label column.
            feature_keys: List of feature column keys.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (features, labels) as numpy arrays.

        """
        # Convert to pandas and then numpy
        df = data.to_pandas()
        if label_key is None:
            label_key = df.columns[-1] if label_key is None else label_key
        keys = feature_keys or [c for c in df.columns if c != label_key]
        X = df[keys].to_numpy()
        y = df[label_key].to_numpy()
        return X, y
