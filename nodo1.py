from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from collections import OrderedDict
import torch
import time
import threading

if __name__ == '__main__':
        
    while True:

        node = Node(MLP(),MnistFederatedDM())
        node.start()

        #node.connect_to("localhost",6666)
        time.sleep(0.1)
        
        node.set_start_learning(rounds=2,epochs=1)

        # Wait 4 results
        
        while True:
            time.sleep(1)

            if node.round is None:
                break

        node.stop()

