import time
from p2pfl.settings import Settings
from test.fixtures import two_nodes, four_nodes

######################
#    Tests Gossip    #
######################

# tendr'iamos que hacer un disconnect la verdad

def test_gossip_heartbeat(four_nodes):
    n1, n2, n3, n4 = four_nodes

    n1.connect_to(n2.host,n2.port, full=False)
    n2.connect_to(n3.host,n3.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)
    time.sleep(1) 

    assert len(n1.heartbeater.get_nodes()) == len(n2.heartbeater.get_nodes()) == len(n3.heartbeater.get_nodes()) == len(n4.heartbeater.get_nodes()) == 3

    n1.stop()
    time.sleep(Settings.NODE_TIMEOUT*3/2) 

    assert len(n2.heartbeater.get_nodes()) == len(n3.heartbeater.get_nodes()) == len(n4.heartbeater.get_nodes()) == 2

    # Stop Nodes
    n2.stop()
    n3.stop()
    n4.stop()

def test_gossip_heartbeat(four_nodes):
    n1, n2, n3, n4 = four_nodes

    n1.connect_to(n2.host,n2.port, full=False)
    n2.connect_to(n3.host,n3.port, full=False)
    n3.connect_to(n4.host,n4.port, full=False)

    time.sleep(2) # Wait for asincronity    

    n1.set_start_learning(rounds=2,epochs=0)    

    while all([n1.round is not None for n in [n1,n2,n3,n4]]):
        time.sleep(0.1)

    for n in [n1,n2,n3,n4]:
        n.stop()
        time.sleep(0.1)