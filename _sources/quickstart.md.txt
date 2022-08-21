# Quickstart

**P2P Federated Learning (p2pfl)** is a library focused on the easy creation of federated learning systems.

## PyTorch Quickstart

In this tutorial we will learn how to train a **Multilayer Perceptron** on **MNIST** using **p2pfl** and PyTorch. We will use a adapted version of the MNIST dataset to federated envairoments.

Two nodes will be uded, one of these nodes will only start and the other will start, connect and start learning.

For the node that only starts (``node1.py``), with a couple of lines everything will be configured:

```python
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP

if __name__ == '__main__':
    node = Node(MLP(),MnistFederatedDM(sub_id=0,number_sub=2),host="127.0.0.1",port=6666)
    node.start()    
```

For the other node (``node2.py``) a few more lines will be needed:

```python
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import time
import sys

if __name__ == '__main__':

    node = Node(MLP(),MnistFederatedDM(sub_id=1,number_sub=2),host="127.0.0.1")
    node.start()

    node.connect_to("127.0.0.1",6666)
    time.sleep(4)
        
    node.set_start_learning(rounds=2,epochs=1)

    # Wait 4 results
        
    while True:
        time.sleep(1)

        if node.round is None:
            break

    node.stop()
```

Now, for executing the experiment, ``node1.py`` will be executed first, after that, execute ``node2.py``.

For more information see the [documentation](https://pguijas.github.io/federated_learning_p2p/documentation.html).

### Other examples

This and othe examples can be found [here](https://github.com/pguijas/federated_learning_p2p/tree/main/examples/).

## TensorFlow Quickstart

``` {note}
   not implemented yet
```
