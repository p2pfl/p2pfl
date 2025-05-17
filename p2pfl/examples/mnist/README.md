![GitHub Logo](../../../other/logo.png)

# P2PFL - MNIST Example

This folder contains an example of using P2PFL with the MNIST dataset. The MNIST dataset is a collection of handwritten digits commonly used for training image processing systems.

## How to run

> Be sure that p2pfl is installed

### Running `mnist.py`

```sh
python p2pfl/examples/mnist/minst.py
```

### Running `minst.yaml`

```sh
python -m p2pfl run p2pfl/examples/mnist/minst.yaml
# or
python -m p2pfl run mnist
```

### Running `node1.py` and then `node1.py`

```sh
# First
python p2pfl/examples/mnist/node1.py --port 6666
# Then
python p2pfl/examples/mnist/node2.py --port 6666
```
