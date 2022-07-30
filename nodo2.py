from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP

if __name__ == '__main__':
            
    node = Node(MLP(),MnistFederatedDM(),host="192.168.1.52",port=6666)
    node.start()
    #node.set_start_learning(rounds=2,epochs=1)
    