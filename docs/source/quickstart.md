# Quickstart

**P2P Federated Learning (p2pfl)** is a library focused on the easy creation of federated learning systems.

## PyTorch Quickstart

In this tutorial, we will learn how to train a **Multilayer Perceptron** on **MNIST** using **p2pfl** and PyTorch. We will use an adapted version of the MNIST dataset to federated environments.

Two nodes will be used, one of these nodes will only start and the other will start, connect and start learning.

The node that only starts (``node1.py``) will be configured with just a couple of lines:

```python
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 nodo1.py <self_port>")
        sys.exit(1)

    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=0, number_sub=2),
        port=int(sys.argv[1]),
    )
    node.start()

    input("Press any key to stop\n")
```

The other node (``node2.py``) will need a few more lines:

```python
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import time
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 nodo2.py <self_port> <other_node_port>")
        sys.exit(1)

    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=1, number_sub=2),
        port=int(sys.argv[1]),
    )
    node.start()

    node.connect(f"127.0.0.1:{sys.argv[2]}")
    time.sleep(4)

    node.set_start_learning(rounds=2, epochs=1)

    # Wait 4 results

    while True:
        time.sleep(1)

        if node.round is None:
            break

    node.stop()
```

Now, to execute the experiment, ``node1.py`` will be executed first, after that, execute ``node2.py``.

For more information see the [documentation](https://pguijas.github.io/federated_learning_p2p/documentation.html).

### Other examples

This and other examples can be found [here](https://github.com/pguijas/federated_learning_p2p/tree/main/examples/).

## TensorFlow Quickstart

``` {note}
   not implemented yet
```
