from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import pytest
import time
import torch
from test.fixtures import two_nodes, four_nodes

###########################
#  Tests Infraestructure  #
###########################

def test_node_paring(two_nodes):
    n1, n2 = two_nodes

    # Conexión
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors)==1

    # Desconexión
    n2.neightboors[0].stop()
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors)== 0


def test_full_connected(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Conexión n1 n2
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors)==1

    # Conexión n3 n1
    n3.connect_to(n1.host,n1.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == len(n3.neightboors) == 2

    # Conexión n4 n1
    n4.connect_to(n1.host,n1.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == len(n3.neightboors) == len(n4.neightboors) == 3

    # Desconexión n1
    n1.stop()
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n2.neightboors) == len(n3.neightboors) == len(n4.neightboors) == 2

    # Desconexión n2
    n2.stop()
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n3.neightboors) == len(n4.neightboors) == 1

    # Desconexión n3
    n3.stop()
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n4.neightboors) == 0

    # Desconexión n4
    n4.stop()

def test_multimsg(two_nodes):
    n1, n2 = two_nodes

    # Conexión
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) 

    n1.broadcast(CommunicationProtocol.build_beat_msg() + CommunicationProtocol.build_beat_msg())
    time.sleep(0.1) 
    assert n2.neightboors[0].errors == 0

    n1.broadcast(CommunicationProtocol.build_beat_msg() + CommunicationProtocol.build_stop_msg())
    time.sleep(0.1) 
    assert len(n2.neightboors) == 0

def test_bad_msg(two_nodes):
    n1, n2 = two_nodes

    # Conexión
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) 

    # Ante 1 error se detendrá la conexión (actualmente según los parátros 1 error es suficiente)
    n1.broadcast(b"saludos Enrique y Dani")
    time.sleep(0.1) 
    assert len(n2.neightboors) == 0


########################
#    Tests Learning    #
########################

#parametrizar, metiendo num rondas y num nodos :)
@pytest.mark.parametrize('x',[(2,1),(2,2)]) 
def test_convergence(x):
    n,r = x

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM())
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
    nodes[0].set_start_learning(rounds=r,epochs=0)

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
        time.sleep(.1) #Esperar por la asincronía


def test_interrupt_train(two_nodes):
    if __name__ == '__main__': # To avoid creating new process when current has not finished its bootstrapping phase
        n1, n2 = two_nodes
        n1.connect_to(n2.host,n2.port)

        time.sleep(1) #Esperar por la asincronía

        n1.set_start_learning(100,100)

        time.sleep(1) #Esperar por la asincronía

        n1.set_stop_learning()

        while n1.round is not None and n2.round is not None:
            print(n1.round,n2.round)
            time.sleep(0.1)