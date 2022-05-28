from p2pfl.const import HEARTBEAT_FREC, TIEMOUT
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


def test_node_abrupt_down(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Conexión n1 n2
    n1.connect_to(n2.host,n2.port)

    # Conexión n3 n1
    n3.connect_to(n1.host,n1.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == len(n3.neightboors) == 2

    # Conexión n4 n1
    n4.connect_to(n1.host,n1.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == len(n3.neightboors) == len(n4.neightboors) == 3

    # Desconexión n4 abruptamente (socket closed)
    #
    #    (n4): será consciente de que la comunicación con n1 se ha perdido cuando haga uso del socket (heartbeat)
    #   (otros) nuevamente el uso del socket (heartbeat) detectará que la conexión ha sido rechazada por el nodo
    for con in n4.neightboors:
        con.socket.close() #provocamos un bad file descriptor
    time.sleep(HEARTBEAT_FREC+0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == len(n3.neightboors) == 2
    n4.stop()
    
    # Desconexión n3 abruptamente (deja de enviar heartbeat)
    n3.heartbeater.stop()
    time.sleep(TIEMOUT+0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == 1
    n3.stop()
    

    # Desconexión n2 y n1
    n2.stop()
    n1.stop()


#------------------------------------------
#   REVISAR INTERRUPCIONES MÁS A FONDO -> (cuando el learner aun no está corriendo pero el proceso está siendo llamado)
#------------------------------------------
def test_interrupt_train():
    n1 = Node("localhost")
    n1.start()
    n1.set_start_learning(99999,99999)

    time.sleep(1) #Esperar por la asincronía

    n1.set_stop_learning()
    n1.stop()
    

def test_interrupt_train2(two_nodes):
    n1, n2 = two_nodes
    n1.connect_to(n2.host,n2.port)

    time.sleep(1) #Esperar por la asincronía

    n1.set_start_learning(99999,99999)

    time.sleep(1) #Esperar por la asincronía

    n2.set_stop_learning()
    
    time.sleep(1) #Esperar por la asincronía
    
    assert n1.round is None
    assert n2.round is None

###################
#  Tests Learning #
###################

# Esto tengo que mirar bien como validarlo
def test_bad_binary_model():
    # cascará pickle con la desserialización del modelo
    assert False

# Modelo incompatible
def test_wrong_model():
    # cascar pytorch
    assert False

#parametrizar, metiendo num rondas y num nodos :)
@pytest.mark.parametrize('x',[(2,1),(2,2)]) 
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
