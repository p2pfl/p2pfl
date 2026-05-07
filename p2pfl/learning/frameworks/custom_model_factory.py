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

"""Custom model factory."""

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


class CustomModelFactory:
    """Factory for creating learners based on the model framework."""

    @classmethod
    def create_model(cls, type: str, model: P2PFLModel) -> P2PFLModel:
        """
        Create a custom model.

        Args:
            type: The type of model.
            model: The model.

        Returns:
            The custom model.

        """
        framework = model.get_framework()
        if framework == "tensorflow":
            from p2pfl.learning.frameworks.tensorflow.custom_models.custom_model_factory import KerasCustomModelFactory

            return KerasCustomModelFactory.create_model(type, model)
        elif framework == "pytorch":
            from p2pfl.learning.frameworks.pytorch.custom_models.custom_model_factory import LightningCustomModelFactory

            return LightningCustomModelFactory.create_model(type, model)
        else:
            logger.error("CustomModelFactory", f"Unsupported framework: {framework}")
            raise ValueError(f"Unsupported framework: {framework}")
