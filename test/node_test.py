from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import pytest
import time
import torch
from test.fixtures import two_nodes, four_nodes

########################
#    Tests Learning    #
########################

@pytest.mark.parametrize('x',[(2,1),(2,2)]) 
def test_convergence(x):
    n,r = x

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port, full=True)
        time.sleep(0.1)

    # Check if they are connected
    time.sleep(3)     
    for node in nodes:
        assert len(node.get_neighbors()) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=r,epochs=0)

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    # Validate that the models obtained are equal
    model = None
    first = True
    for node in nodes:
        if first:
            model = node.learner.get_parameters()
            first = False
        else:
            for layer in model:
                a = torch.round(model[layer], decimals=2)
                b = torch.round(node.learner.get_parameters()[layer], decimals=2)
                assert torch.eq(a, b).all()

    # Close
    for node in nodes:
        node.stop()
        time.sleep(.1) # Wait because of asincronity


def test_interrupt_train(two_nodes):
    if __name__ == '__main__': # To avoid creating new process when current has not finished its bootstrapping phase
        n1, n2 = two_nodes
        n1.connect_to(n2.host,n2.port, full=True)

        time.sleep(1) # Wait because of asincronity

        n1.set_start_learning(100,100)

        time.sleep(1) # Wait because of asincronity

        n1.set_stop_learning()

        while n1.round is not None and n2.round is not None:
            time.sleep(0.1)

        

# TO IMPLEMET WHEN THE TOPOLOGY WAS NOT FULLY CONNECTED
def test_connect_while_training(four_nodes):
    n1,n2,n3,n4 = four_nodes
    
    # Connect Nodes (unless the n4)
    n1.connect_to(n2.host,n2.port, full=True)
    n3.connect_to(n1.host,n1.port, full=True)
    time.sleep(0.1) #Esperar por la asincronía

    # Start Learning
    n1.set_start_learning(2,1)
    time.sleep(4)

    # Try to connect
    assert n1.connect_to(n4.host,n4.port, full=True) == None    
    n4.connect_to(n1.host,n1.port,full=False)
    time.sleep(1)
    assert n4.get_neighbors() == [] 

    for n in four_nodes:
        n.stop()
        time.sleep(0.1)
    