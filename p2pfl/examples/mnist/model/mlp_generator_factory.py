#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
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

"""Factory of functions to create MLPs for MNIST."""

import contextlib
from enum import Enum, auto
from typing import Callable

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_flax import model_build_fn as model_build_fn_flax


class ModelType(Enum):
    """Enum for the supported model types."""

    TENSORFLOW = auto()
    PYTORCH = auto()
    FLAX = auto()


def get_model_builder(model_type: ModelType) -> Callable:
    """Get the model builder function for the given model type."""
    match model_type:
        case ModelType.TENSORFLOW:
            return model_build_fn_tensorflow
        case ModelType.PYTORCH:
            return model_build_fn_pytorch
        case ModelType.FLAX:
            return model_build_fn_flax
        case _:
            raise ValueError("Invalid model type")
