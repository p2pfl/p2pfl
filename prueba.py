from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time



import pytest
from p2pfl.base_node import BaseNode
from p2pfl.communication_protocol import CommunicationProtocol
import time
import pytest
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
from p2pfl.settings import Settings

def test_node_down_on_learning(n):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM(sub_id=i, number_sub=n),simulation=True)
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port, full=True)
        time.sleep(1)

    # Check if they are connected
#    time.sleep(1)
    for node in nodes:
        assert len(node.get_neighbors()) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=4,epochs=2)

    # Stopping node
    #nodes[1].stop()
    
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



def four_nodes():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(MLP(),MnistFederatedDM())
    n3 = Node(MLP(),MnistFederatedDM())
    n4 = Node(MLP(),MnistFederatedDM())
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    return n1,n2,n3,n4


def test_gossip_heartbeat():
    nodes = four_nodes()
    n1, n2, n3, n4 = nodes
    n1.connect_to(n2.host,n2.port, full=False)
    n2.connect_to(n3.host,n3.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)

    acum = 0
    while len(n1.heartbeater.get_nodes()) != 4 or len(n2.heartbeater.get_nodes()) != 4 or len(n3.heartbeater.get_nodes()) != 4 or len(n4.heartbeater.get_nodes()) != 4:
        begin = time.time()
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > 6:
            assert False

    for n in nodes:
        print(n.get_name(), " ->" ,n.heartbeater.get_nodes())

    n1.set_start_learning(rounds=10,epochs=2)    

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        print([node.round is None for node in nodes])
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    for node in nodes:
        node.stop()



def test_encrypted_convergence(x):
    n,r = x

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM(), simulation=False)
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port, full=True)
        time.sleep(0.1)
        
    # Overhead at handshake because of the encryption
    time.sleep(3)

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

    # Close
    for node in nodes:
        node.stop()
        time.sleep(.1) # Wait because of asincronity

########################################################################################################################################################################################################

import time
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
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
    while len(n1.heartbeater.get_nodes()) != 4 or len(n2.heartbeater.get_nodes()) != 4 or len(n3.heartbeater.get_nodes()) !=  4 or len(n4.heartbeater.get_nodes()) != 4:
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
    assert len(n1.heartbeater.get_nodes()) == len(n2.heartbeater.get_nodes()) == len(n3.heartbeater.get_nodes()) == len(n4.heartbeater.get_nodes()) == 4

########################################################################################################################################################################################################

if __name__ == '__main__':

    for _ in range(50):
        #test_gossip_heartbeat()
        test_node_down_on_learning(5)
        #test_encrypted_convergence((2,1))
        #print("\n\n\n\n\n")
        break
        """
        test_ring_network_learning(four_nodes())
        print("\n\n\n\n\n")
        test_line_network_learning(four_nodes())
        print("\n\n\n\n\n")
        test_star_network_learning(four_nodes())
        print("\n\n\n\n\n")
        """
