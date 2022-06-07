from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import pytest
import time
import torch
from test.fixtures import two_nodes, four_nodes

def test_interrupt_train():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(MLP(),MnistFederatedDM())
    n1.start()
    n2.start()

    # Conexión
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors)==1

    # Desconexión
    n2.neightboors[0].stop()
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors)== 0

    n1.stop()
    n2.stop()
    
test_interrupt_train()