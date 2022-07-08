import time
from p2pfl.settings import Settings
from test.fixtures import two_nodes, four_nodes

######################
#    Tests Gossip    #
######################

# tendr'iamos que hacer un disconnect la verdad


def test_gossip_learning(four_nodes):
    n1, n2, n3, n4 = four_nodes

    n1.connect_to(n2.host,n2.port, full=False)
    n2.connect_to(n3.host,n3.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)

    acum = 0
    while len(n1.heartbeater.get_nodes()) != len(n2.heartbeater.get_nodes()) != len(n3.heartbeater.get_nodes()) != len(n4.heartbeater.get_nodes()) != 3:
        begin = time.time()
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > 6:
            assert False

    n1.set_start_learning(rounds=2,epochs=0)    

    while all([n.round is not None for n in four_nodes]):
        time.sleep(0.1)

    for n in four_nodes:
        n.stop()
        time.sleep(0.1)