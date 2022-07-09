import time
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node

def test_ring_network_learning():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(MLP(),MnistFederatedDM())
    n3 = Node(MLP(),MnistFederatedDM())
    n4 = Node(MLP(),MnistFederatedDM())
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    # RED ANILLO
    n1.connect_to(n2.host,n2.port, full=False)
    n2.connect_to(n3.host,n3.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)
    n4.connect_to(n1.host,n1.port, full=False)
    
    __test_learning((n1,n2,n3,n4))

    n1.stop()
    n2.stop()
    n3.stop()
    n4.stop()

def test_star_network_learning():
    assert False

def __test_learning(nodes):
    n1, n2, n3, n4 = nodes

    acum = 0
    while len(n1.heartbeater.get_nodes()) != len(n2.heartbeater.get_nodes()) != len(n3.heartbeater.get_nodes()) != len(n4.heartbeater.get_nodes()) != 4:
        begin = time.time()
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > 6:
            assert False

    time.sleep(1)

    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == len(n3.get_neighbors()) == len(n4.get_neighbors()) == 2

    n1.set_start_learning(rounds=2,epochs=0)  

    time.sleep(1)  

    while all([n.round is not None for n in nodes]):
        time.sleep(0.1)

    time.sleep(1)  

    # Verify network nodes
    assert len(n1.heartbeater.get_nodes()) == len(n2.heartbeater.get_nodes()) == len(n3.heartbeater.get_nodes()) == len(n4.heartbeater.get_nodes()) == 4

    # Verify node neighbors
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == len(n3.get_neighbors()) == len(n4.get_neighbors()) == 2
