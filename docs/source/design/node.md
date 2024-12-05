# 🏛️ Node

The `Node` class is the central component of the P2PFL library, representing a single participant in a decentralized federated learning network. It encapsulates the functionalities required for a node to participate in the collaborative training of machine learning models.  Each `Node` manages its own dataset, model, learning process, and communication with its peers.

Como se ha comentado en el diseño global de la librería, este componente actúa como un central hub con los diferentes componentes involucrados en el federated learning process. En esencia, delega lógica y ofreciendo una interfaz sencilla para el usuario.

## 💪 Main functionality

- DELEGA
- En esencia, el nodo permite lo siguiente:
    - Estado compartido
    - Protocolo comunicacion
        - Neighborhood management
        - Node Management (servicer loop)
    - Learning Setters
        - learning, model, data
    - Network Learning Management
        - StartLearningCommand
        - learning_workflow
- Merge de cosas....
    


Local Learning
    
> asiasa
- **Model Initialization**:  Sets up the machine learning model to be trained.  The model must implement the `P2PFLModel` interface.
- **Dataset Handling**:  Manages the local dataset used for training.  The dataset must implement the `P2PFLDataset` interface.
- **Communication**:  Handles communication with other nodes in the network using a specified communication protocol.
- **Learning Management**:  Orchestrates the local training process and the aggregation of model updates.
- **State Management**:  Maintains the state of the node, including its current status, round information, and neighbor connections.


## ⚙️ Usage




## Node Lifecycle

### Starting the Node
To start the node's services:

```python
node.start(wait=False)
```

- `wait`: If `True`, blocks the process until the node stops.

### Stopping the Node
To stop the node and its communication services:

```python
node.stop()
```

# A partir de aqui esto no me gusta, muy disperso, no creo que aporete mucho más que leer las firmas de node, ns....
## Initialization

> no aporta nada

To create a `Node`:

```python
node = Node(
    model=MLP(),
    data=MnistFederatedDM(),
    address="127.0.0.1"
)
```


## Neighbor Management

### Connect to Another Node
Link the Node to a specific peer:

```python
node.connect("192.168.1.10:666")
```

- `addr`: Address of the target node.

### Get Neighbors
Retrieve a list of connected peers:
```python
neighbors = node.get_neighbors(only_direct=True)
```

- `only_direct`: If `True`, returns only direct neighbors.

### Disconnect from a Node
Remove a specific peer connection:

```python
node.disconnect("192.168.1.10:666")
```

- `addr`: Address of the target node.

## Learning Workflow
### Starting Learning
Start distributed learning across the network:

```python
node.set_start_learning(rounds=3, epochs=2)
```

- `rounds`: Number of training rounds.
- `epochs`: Number of epochs per round.

### Stopping Learning
Stop the learning process in the network:

```python
node.set_stop_learning()
```

## Node State Overview
> ESTO DEBERIA TENER SU PROPIA SECCION
The `NodeState` keeps track of important information, such as:

- `Status`: Current activity (e.g., Idle, Learning).
- `Round Information`: Current and total rounds of the experiment.
- `Neighbors`: Tracks peer connections and their statuses.
- `Locks and Events`: Manage synchronization during training.
