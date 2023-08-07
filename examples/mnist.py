from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time

def mnist_execution(n,r,start,simulation,conntect_to=None, iid=True):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM(sub_id=i, number_sub=n, iid=iid),simulation=simulation)
        node.start()
        nodes.append(node)
    
    # Connect other network
    if conntect_to is not None:
        nodes[0].connect_to(conntect_to[0],conntect_to[1])

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port, full=True)
        time.sleep(1)

    time.sleep(5)
    print("Starting...")
    
    for n in nodes:
        print(len(n.get_neighbors()))
        print(len(n.get_network_nodes()))

    # Start Learning
    if start:
        nodes[0].set_start_learning(rounds=r,epochs=1)
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


if __name__ == '__main__':
    # Record time
    start_time = time.time()
    mnist_execution(10,50,True,True)
    # Print time in minutes
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
 
