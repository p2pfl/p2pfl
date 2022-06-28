import pytest
from p2pfl.base_node import BaseNode
from p2pfl.communication_protocol import CommunicationProtocol
import time


@pytest.fixture
def two_nodes():
    n1 = BaseNode()
    n2 = BaseNode()
    n1.start()
    n2.start()

    yield n1,n2

    n1.stop()
    n2.stop()

@pytest.fixture
def four_nodes():
    n1 = BaseNode()
    n2 = BaseNode()
    n3 = BaseNode()
    n4 = BaseNode()
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    return n1,n2,n3,n4
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

