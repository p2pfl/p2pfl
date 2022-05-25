from p2pfl.node import Node
import pytest
import time
import torch

@pytest.fixture
def two_nodes():
    n1 = Node("localhost")
    n2 = Node("localhost")
    n1.start()
    n2.start()

    yield n1,n2

    n1.stop()
    n2.stop()

@pytest.fixture
def four_nodes():
    n1 = Node("localhost")
    n2 = Node("localhost")
    n3 = Node("localhost")
    n4 = Node("localhost")
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    return n1,n2,n3,n4

#
#
# PROBLEMA CON LA SALIDA DEL TEST -> SE QUEDAN CONEXIONES ABIERTAS, REVISARLO
#
#

###########
#  Tests  #
###########


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

"""
#parametrizar, metiendo num rondas y num nodos :)
@pytest.mark.parametrize('x',[(2,1),(2,2),(4,1),(4,2)]) 
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
"""