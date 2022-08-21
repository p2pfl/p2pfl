# P2P Federated Learning

**P2P Federated Learning (p2pfl)** is a decentralized federated learning library, it allows creating basic federated learning systems on p2p networks using gossip protocols.

<!---``` 
sphinx-apidoc -F -o tmp ../p2pfl
```
-->

## Future work

- [ ] Add new aggregation algoritms
- [ ] Secure Aggregation
- [ ] Iterative terminal for node gestion
- [ ] Connect nodes during training process
- [ ] Gossip loop optimization to send models (too much sendings when a node fails)
- [ ] Tolerance to incomplete reception of messages
- [ ] Tests with a high number of nodes in deploy
- [ ] Comparison of node settings when learning starts
- [ ] Associate votes to rounds
- [ ] Network Login Authentication
- [ ] Study about attacks that could be done in the network
- [ ] Add support for dockerized installation

## Index

```{eval-rst}
.. toctree::
   :maxdepth: 4

   installation
   quickstart
   documentation

```
