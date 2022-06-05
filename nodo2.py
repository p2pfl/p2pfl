from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from collections import OrderedDict
import torch
import time

node = Node(MLP(),MnistFederatedDM(),port=7888)
node.start()