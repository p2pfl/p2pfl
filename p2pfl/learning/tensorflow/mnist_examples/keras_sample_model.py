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

"""Keras sample model."""

# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.metrics import SparseCategoricalAccuracy


# class MLP(tf.keras.Model):
#     """
#     Multilayer Perceptron (MLP) to solve MNIST with Keras.
#     """

#     def __init__(self, out_channels: int = 10, **kwargs) -> None:
#         super(MLP, self).__init__(**kwargs)
#         self.flatten = Flatten()
#         self.l1 = Dense(256, activation="relu")
#         self.l2 = Dense(128, activation="relu")
#         self.l3 = Dense(out_channels)
#         self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
#         self.metric = SparseCategoricalAccuracy()

#     def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
#         x = self.flatten(inputs)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         return x


# class CNN(tf.keras.Model):
#     """
#     Convolutional Neural Network (CNN) to solve MNIST with Keras.
#     """

#     def __init__(self, out_channels: int = 10, **kwargs) -> None:
#         super(CNN, self).__init__(**kwargs)
#         self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
#         self.flatten = Flatten()
#         self.d1 = Dense(128, activation="relu")
#         self.d2 = Dense(out_channels)
#         self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
#         self.metric = SparseCategoricalAccuracy()

#     def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
#         x = self.conv1(inputs)
#         x = self.flatten(x)
#         x = self.d1(x)
#         x = self.d2(x)
#         return x
