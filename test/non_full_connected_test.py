import time
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
from test.fixtures import four_nodes

def test_ring_network_learning(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Ring network
    n1.connect_to(n2.host,n2.port, full=False)
    n2.connect_to(n3.host,n3.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)
    n4.connect_to(n1.host,n1.port, full=False)

    __wait_connection(four_nodes)
    
    # Verify topology 
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == len(n3.get_neighbors()) == len(n4.get_neighbors()) == 2

    __test_learning((n1,n2,n3,n4))

    # Verify topology after learning
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == len(n3.get_neighbors()) == len(n4.get_neighbors()) == 2

    # Stop Nodes
    [n.stop() for n in four_nodes]
        

def test_star_network_learning(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Star Network
    n1.connect_to(n4.host,n4.port, full=False)
    n2.connect_to(n4.host,n4.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)

    __wait_connection(four_nodes)

    # Verify topology after learning
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == len(n3.get_neighbors()) == 1
    assert len(n4.get_neighbors()) == 3

    __test_learning((n1,n2,n3,n4))

    # Verify topology after learning
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == len(n3.get_neighbors()) == 1
    assert len(n4.get_neighbors()) == 3

    # Stop Nodes
    [n.stop() for n in four_nodes]

def test_line_network_learning(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Star Network
    n1.connect_to(n2.host,n2.port, full=False)
    n2.connect_to(n3.host,n3.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)

    __wait_connection(four_nodes)

    # Verify topology
    assert len(n1.get_neighbors()) == len(n4.get_neighbors()) == 1
    assert len(n2.get_neighbors()) == len(n3.get_neighbors()) == 2

    __test_learning((n1,n2,n3,n4))

    # Verify topology after learning
    assert len(n1.get_neighbors()) == len(n4.get_neighbors()) == 1
    assert len(n2.get_neighbors()) == len(n3.get_neighbors()) == 2

    # Stop Nodes
    [n.stop() for n in four_nodes]

#######################
#    AUX FUNCTIONS    #
#######################

def __wait_connection(nodes):
    n1, n2, n3, n4 = nodes
    acum = 0
    while len(n1.get_network_nodes()) != 4 or len(n2.get_network_nodes()) != 4 or len(n3.get_network_nodes()) !=  4 or len(n4.get_network_nodes()) != 4:
        begin = time.time()
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > 6:
            assert False

def __test_learning(nodes):
    n1, n2, n3, n4 = nodes
    n1.set_start_learning(rounds=2,epochs=0)  
    time.sleep(1)  
    while any([n.round is not None for n in nodes]):
        time.sleep(0.1)

    # Verify network nodes
    assert len(n1.get_network_nodes()) == len(n2.get_network_nodes()) == len(n3.get_network_nodes()) == len(n4.get_network_nodes()) == 4
