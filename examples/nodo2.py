from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import sys

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 nodo2.py <self_host> <self_port>")
        sys.exit(1)

    node = Node(MLP(),MnistFederatedDM(),host=sys.argv[1],port=sys.argv[2])
    node.start()    