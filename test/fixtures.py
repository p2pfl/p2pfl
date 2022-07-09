import pytest
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node

@pytest.fixture
def two_nodes():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(MLP(),MnistFederatedDM())
    n1.start()
    n2.start()

    yield n1,n2

    n1.stop()
    n2.stop()

@pytest.fixture
def four_nodes():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(MLP(),MnistFederatedDM())
    n3 = Node(MLP(),MnistFederatedDM())
    n4 = Node(MLP(),MnistFederatedDM())
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    return (n1,n2,n3,n4)