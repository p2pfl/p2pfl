![GitHub Logo](logo.png)

# P2PFL - Federated Learning over p2p networks

[![GitHub license](https://img.shields.io/github/license/pguijas/federated_learning_p2p)](https://github.com/pguijas/federated_learning_p2p/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/pguijas/federated_learning_p2p)](https://img.shields.io/github/issues/pguijas/federated_learning_p2p)
![GitHub forks](https://img.shields.io/github/forks/pguijas/federated_learning_p2p)
![GitHub forks](https://img.shields.io/github/stars/pguijas/federated_learning_p2p)

p2pfl is a decentralized federated learning library, it allows creating basic federated learning systems on p2p networks using gossip protocols.

See documentation [here](https://pguijas.github.io/federated_learning_p2p/).

See memory and future work [here](memoria.pdf)

## Installation

To install the library, you can simply run:

```bash
pip install p2pfl
```

## Important

Carefully with number of open files at high scale experiments.

If fails, try to change the number of open files. `ulimit -n {VALUE}`
