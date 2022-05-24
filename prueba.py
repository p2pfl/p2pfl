from p2pfl.node import Node
from p2pfl.learning.model import MyNodeLearning
from p2pfl.agregator import FedAvg    
from collections import OrderedDict
import torch
import time

def test_convergence(x):
    n,r = x

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(host="localhost")
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port)
        time.sleep(0.1)

    # Check if they are connected
    for node in nodes:
        assert len(node.neightboors) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=r)

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f
            
        if finish:
            break


    # Validamos Modelos obtenidos sean iguales
    model = None
    first = True
    for node in nodes:
        if first:
            model = node.learner.get_parameters()
            first = False
        else:
            for layer in model:
                a = torch.round(model[layer], decimals=2)
                b = torch.round(node.learner.get_parameters()[layer], decimals=2)
                assert torch.eq(a, b).all()

    # Cerrar
    for node in nodes:
        node.stop()
        time.sleep(.2) #Esperar por la asincronía
        
        
test_convergence((4,3))



