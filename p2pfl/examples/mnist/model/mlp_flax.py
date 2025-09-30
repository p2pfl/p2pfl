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

"""Simple MLP on PyTorch Lightning for MNIST."""

from flax import linen as nn

from p2pfl.learning.frameworks.flax.flax_model import FlaxModel

# from p2pfl.learning.frameworks.flax.flax_model import FlaxModel

####
# Example MLP in Flax
####


class MLP(nn.Module):
    """Multilayer Perceptron (MLP) for MNIST classification using Flax."""

    hidden_sizes: tuple[int, int] = (256, 128)
    out_channels: int = 10

    @nn.compact
    def __call__(self, x):
        """
        Define the forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input tensor, expected to be a flattened MNIST image,
                             or a batch of images with shape (batch_size, image_size).

        Returns:
            jnp.ndarray: The output logits of the MLP, with shape (batch_size, out_channels).
                         These represent the unnormalized scores for each class.

        """
        x = x.reshape((1, -1))
        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(features=size)(x))
        x = nn.Dense(features=self.out_channels)(x)
        return x


# Export P2PFL model
def model_build_fn(*args, **kwargs) -> FlaxModel:
    """Export the model build function."""
    raise NotImplementedError("This function is not implemented yet, FLAX is not enought mature")
