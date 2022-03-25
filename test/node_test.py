from p2pfl.node import Node
import pytest
import time

@pytest.fixture
def nodes():
    n1 = Node("localhost")
    n2 = Node("localhost")
    n1.start()
    n2.start()

    yield n1,n2

    n1.stop()
    n2.stop()


def test_node_paring(nodes):
    n1, n2 = nodes

    # Conexión
    n1.connect_to(n2.host,n2.port)
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors)==1

    # Desconexión
    n2.neightboors[0].stop()
    time.sleep(0.1) #Esperar por la asincronía
    assert len(n1.neightboors) == len(n2.neightboors)== 0

