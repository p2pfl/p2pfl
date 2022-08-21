# federated_learning_p2p

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
