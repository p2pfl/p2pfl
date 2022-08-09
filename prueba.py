from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time

def mnist_execution(n,start,simulation,conntect_to=None, iid=True):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM(sub_id=i, number_sub=n, iid=iid),host="192.168.1.62",simulation=simulation)
        node.start()
        nodes.append(node)
    
    # Connect other network
    if conntect_to is not None:
        nodes[0].connect_to(conntect_to[0],conntect_to[1])

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port, full=True)
        time.sleep(1)

    print("Esperamos")
    time.sleep(5)
    print("Empezamos")
    for n in nodes:
        print(len(n.get_neighbors()))
        print(len(n.get_network_nodes()))

    # Start Learning
    if start:
        nodes[0].set_start_learning(rounds=10,epochs=2)
    else:
        time.sleep(20)

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



def four_nodes():
    n1 = Node(MLP(),MnistFederatedDM())
    n2 = Node(MLP(),MnistFederatedDM())
    n3 = Node(MLP(),MnistFederatedDM())
    n4 = Node(MLP(),MnistFederatedDM())
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    return n1,n2,n3,n4

if __name__ == '__main__':

    for _ in range(50):
        mnist_execution(1,True,True)
        break
 
