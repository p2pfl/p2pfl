import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


class MLP(tf.keras.Model):
    """
    Multilayer Perceptron (MLP) to solve MNIST with Keras.
    """

    def __init__(self, out_channels: int = 10, **kwargs) -> None:
        super(MLP, self).__init__(**kwargs)
        self.flatten = Flatten()
        self.l1 = Dense(256, activation="relu")
        self.l2 = Dense(128, activation="relu")
        self.l3 = Dense(out_channels)
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.metric = SparseCategoricalAccuracy()

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x = self.flatten(inputs)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
