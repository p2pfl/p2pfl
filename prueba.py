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
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port)
        time.sleep(1)

    # Check if they are connected
#    time.sleep(1)
    for node in nodes:
        assert len(node.neightboors) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=4,epochs=0)

    # Stopping node
    #nodes[1].stop()
    
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

    time.sleep(2) # Wait for asincronity    
    print(len(n1.heartbeater.get_nodes()) , len(n2.heartbeater.get_nodes()), len(n3.heartbeater.get_nodes()) , len(n4.heartbeater.get_nodes()))

    assert len(n1.heartbeater.get_nodes()) == len(n2.heartbeater.get_nodes()) == len(n3.heartbeater.get_nodes()) == len(n4.heartbeater.get_nodes()) == 3

    n1.set_start_learning(rounds=2,epochs=0)    


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




if __name__ == '__main__':

    for _ in range(50):
        test_gossip_heartbeat()
        #test_node_down_on_learning(5)
        print("\n\n\n\n\n")
        #break

