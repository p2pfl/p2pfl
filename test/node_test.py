from p2pfl import Node
import pytest
import time

def init_nodes():
    #Init nodes
    #poner puerto libre alto ranom
    n1 = Node("localhost",6661)
    n2 = Node("localhost",6668)
    n1.start()
    n2.start()
    return n1,n2

def stop_nodes(n1,n2):
    n1.stop()
    n2.stop()

def test_node_creatrion():
    n1, n2 = init_nodes()

    assert len(n1.neightboors) == len(n2.neightboors)==1

    stop_nodes(n1,n2)

