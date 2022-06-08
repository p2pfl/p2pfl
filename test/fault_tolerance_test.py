from p2pfl.const import HEARTBEAT_FREC, SOCKET_TIEMOUT
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import pytest
import time
from test.fixtures import two_nodes, four_nodes

    
def __test_node_abrupt_down(four_nodes):
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

    # Desconexión n4 abruptamente (socket closed)å
    #    (n4): será consciente de que la comunicación con n1 se ha perdido cuando haga uso del socket (heartbeat)
    #   (otros) nuevamente el uso del socket (heartbeat) detectará que la conexión ha sido rechazada por el nodo
    for con in n4.neightboors:
        con.socket.close() #provocamos un bad file descriptor
    time.sleep(HEARTBEAT_FREC+0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == len(n3.neightboors) == 2
    n4.stop()
    
    # Desconexión n3 abruptamente (deja de enviar heartbeat)
    n3.heartbeater.stop()
    time.sleep(SOCKET_TIEMOUT+0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors) == 1
    n3.stop()
    

    # Desconexión n2 y n1
    n2.stop()
    n1.stop()

    
# QUEDA COLGADO ALGÚN THREAD
@pytest.mark.parametrize('n',[2,4]) 
def test_node_down_on_learning(n):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port)
        time.sleep(1)

    # Check if they are connected
    for node in nodes:
        assert len(node.neightboors) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=2,epochs=0)

    # Stopping node
    nodes[1].stop()
    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    for node in nodes:
        node.stop()

def __test_abrupt_connection_down_on_learning(two_nodes):
    pass

# por acabar
def test_bad_binary_model(two_nodes):
    n1, n2 = two_nodes
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) 

    # Start Learning
    n1.set_start_learning(rounds=2,epochs=0)
 
    # Adding noise to the buffer
    for _ in range(200):
        n1.neightboors[0].param_bufffer += "noise".encode("utf-8")
    
    while not n1.round is None and not n2.round is None:
        time.sleep(0.1)
        
# Modelo incompatible
def test_wrong_model():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(CNN(),MnistFederatedDM())
    
    n1.start()
    n2.start()
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) 
    
    #n1.set_start_learning(rounds=2,epochs=0)

    # Wait 4 results
    while not n1.round is None and not n2.round is None:
        time.sleep(0.1)

    n1.stop()
    n2.stop()
    