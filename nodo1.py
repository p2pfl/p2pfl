from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from collections import OrderedDict
import torch
import time
import threading

while True:
    for thread in threading.enumerate(): 
        print(thread.name)
    print("\n")

    node = Node(CNN(),MnistFederatedDM())
    #node.start()


    for thread in threading.enumerate(): 
        print(thread.name)
    print("\n")

    node.connect_to("localhost",6666)
    time.sleep(0.1)

    node.set_start_learning(rounds=2,epochs=0)

    # Wait 4 results
    while True:
        time.sleep(1)

        if node.round is None:
            break

    node.stop()

    break