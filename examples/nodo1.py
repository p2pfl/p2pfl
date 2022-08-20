from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import time
import sys

if __name__ == '__main__':
        
    while True:

        if len(sys.argv) != 5:
            print("Usage: python3 nodo1.py <self_host> <self_port> <other_node_host> <other_node_port>")
            sys.exit(1)

        node = Node(MLP(),MnistFederatedDM(),host=sys.argv[1],port=sys.argv[2])
        node.start()

        node.connect_to(sys.argv[3],sys.argv[4])
        time.sleep(4)
        
        node.set_start_learning(rounds=2,epochs=1)

        # Wait 4 results
        
        while True:
            time.sleep(1)

            if node.round is None:
                break

        node.stop()

