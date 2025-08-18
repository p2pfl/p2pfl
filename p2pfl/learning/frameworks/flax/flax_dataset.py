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

"""Flax Dataset export strategy."""

from collections.abc import Generator
from typing import Any

import jax.numpy as jnp
from datasets import Dataset  # type: ignore
from torch.utils.data import DataLoader

from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy


class FlaxExportStrategy(DataExportStrategy):
    """Export strategy for JAX/Flax datasets."""

    @staticmethod
    def export(
        data: Dataset,
        batch_size: int | None = None,
        num_workers: int = 0,
        **kwargs,
    ) -> Generator[tuple[jnp.ndarray, jnp.ndarray], Any, None]:
        """
        Export the data using the JAX/Flax strategy.

        Args:
            data: The data to export. Transforms should already be applied to the dataset via set_transform.
            batch_size: The batch size to use for the exported data.
            num_workers: The number of workers to use for the exported
            kwargs: Additional keyword arguments.

        Returns:
            The exported data.

        """
        # TODO: HARDCODEADO A MNIST?

        # TODO: fix dataloader .with_format(type="jax", ...) with custom collate_fn
        torch_loader = DataLoader(data.with_format(type="torch", output_all_columns=True), batch_size=batch_size, num_workers=num_workers)

        def jax_batch_generator() -> Generator[tuple[jnp.ndarray, jnp.ndarray], Any, None]:
            for batch in torch_loader:
                features = jnp.array(batch["image"].numpy())
                labels = jnp.array(batch["label"].numpy())
                yield features, labels

        return jax_batch_generator()
