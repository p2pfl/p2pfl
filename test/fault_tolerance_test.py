from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import pytest
import time

@pytest.fixture
def two_nodes():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(MLP(),MnistFederatedDM())
    n1.start()
    n2.start()

    yield n1,n2

    n1.stop()
    n2.stop()

def test_bad_binary_model(two_nodes):
    n1, n2 = two_nodes
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) 

    # Start Learning
    n1.set_start_learning(rounds=2,epochs=0)
 
    # Adding noise to the buffer
    for _ in range(2000):
        n1.neightboors[0].param_bufffer += "noise".encode("utf-8")
        
# Modelo incompatible
def __test_wrong_model():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(CNN(),MnistFederatedDM())
    n1.start()
    n2.start()
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) 
    
    n1.set_start_learning(rounds=2,epochs=0)

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in [n1,n2]]:
            finish = finish and f

        if finish:
            break

    n1.stop()
    n2.stop()
