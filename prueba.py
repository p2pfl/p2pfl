from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time

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
    nodes[0].set_start_learning(rounds=2,epochs=0)

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


if __name__ == '__main__':
    for _ in range(50):
        test_node_down_on_learning(5)
        print("\n\n\n\n\n")

