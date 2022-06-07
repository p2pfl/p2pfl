from p2pfl.const import HEARTBEAT_FREC, TIEMOUT
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import pytest
import time

nodes = []
        
def test_node_down_on_learning(n):

    # Node Creation
    
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port)
        time.sleep(0.5)

    # Check if they are connected
    for i in range(len(nodes)-1):
        node=nodes[i]
        print("node {} is connected to {}".format(i,len(node.neightboors)))
        assert len(node.neightboors) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=2,epochs=0)

    # Stopping node
    nodes[1].stop()

  
    # Wait 4 results
    while True:
        time.sleep(1)
        
        for node in nodes:
            print(node.agredator.models.keys())
        
        
        finish = True
        x = [node.round is None for node in nodes]
        print(x)
        y = [len(nc.param_bufffer) for nc in node.neightboors for node in nodes]
        print(y)
        y = [nc.tmp for nc in node.neightboors for node in nodes]
        print(y)
        for f in x:
            finish = finish and f

        if finish:
            break


    return nodes

while True:
    nodes = []
    test_node_down_on_learning(5)

    for node in nodes:
        print("----------------------STOP------------------------------")
        node.stop()