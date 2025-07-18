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

"""P2PFLCallback factory."""

from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


class LearnerFactory:
    """Factory for creating learners based on the model framework."""

    @classmethod
    def create_learner(cls, model: P2PFLModel) -> type[Learner]:
        """
        Create a learner based on the model framework.

        Args:
            model: The model to encapsulate.

        """
        framework = model.get_framework()
        if framework == Framework.PYTORCH.value:
            from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

            return LightningLearner
        elif framework == Framework.TENSORFLOW.value:
            from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner

            return KerasLearner
        elif framework == Framework.FLAX.value:
            from p2pfl.learning.frameworks.flax.flax_learner import FlaxLearner

            return FlaxLearner
        else:
            logger.error("LearnerFactory", f"Unsupported framework: {framework}")
            raise ValueError(f"Unsupported framework: {framework}")
