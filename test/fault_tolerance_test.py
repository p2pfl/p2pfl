from numpy import full
from p2pfl.settings import Settings
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import pytest
import time
from test.fixtures import two_nodes, four_nodes
    


    

    
# QUEDA COLGADO ALGÚN THREAD
@pytest.mark.parametrize('n',[2,4]) 
def test_node_down_on_learning(n):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port, full=True)
        time.sleep(1)

    # Check if they are connected
    for node in nodes:
        assert len(node.get_neighbors()) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=2,epochs=0)

    # Stopping node
    time.sleep(0.3)
    nodes[1].stop()
    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    for node in nodes:
        node.stop()

def __test_abrupt_connection_down_on_learning(two_nodes):
    pass

def test_bad_binary_model(two_nodes):
    n1, n2 = two_nodes
    n1.connect_to(n2.host,n2.port, full=True)
    time.sleep(0.1) 

    # Start Learning
    n1.set_start_learning(rounds=2,epochs=0)
 
    # Adding noise to the buffer
    for _ in range(200):
        n1.get_neighbors()[0]._NodeConnection__param_bufffer += "noise".encode("utf-8")
    
    while not n1.round is None and not n2.round is None:
        time.sleep(0.1)

    # Wait agregation thread of node 2 -> test still running if we don't wait
    time.sleep(Settings.AGREGATION_TIMEOUT+2)
        
# Modelo incompatible
def test_wrong_model():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(CNN(),MnistFederatedDM())
    
    n1.start()
    n2.start()
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) 
    
    n1.set_start_learning(rounds=2,epochs=0)
    time.sleep(0.1)

    # Wait 4 results
    while not n1.round is None and not n2.round is None:
        time.sleep(0.1)

    n1.stop()
    n2.stop()
