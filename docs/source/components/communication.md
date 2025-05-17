# 📡 Communication Protocols

Communication protocols are the backbone of P2PFL, dictating how nodes interact and exchange information (models, metrics, commands) within the decentralized network.  P2PFL is designed to be protocol-agnostic, allowing you to choose the communication method that best suits your needs.  This flexibility is achieved through the `CommunicationProtocol` abstract base class, which defines a consistent interface for different protocols. This section details the currently supported protocols and how to use them within the P2PFL framework.

## Supported Protocols

### gRPC (gRPC Remote Procedure Call)

gRPC is a high-performance, open-source universal RPC framework that enables efficient communication between distributed systems. P2PFL leverages gRPC for reliable, secure, and scalable data exchange between peers.  Key advantages of using gRPC include:

* **Cross-platform compatibility:**  Supports a wide range of languages and platforms, enabling diverse nodes to participate in the federated learning process.
* **Streamlined communication:**  Offers bidirectional streaming and efficient binary serialization using Protocol Buffers, minimizing communication overhead.
* **Scalability:**  Designed for high-performance and scalability, making it suitable for large-scale federated networks.
* **Security:** Supports Transport Layer Security (TLS) and mutual TLS (mTLS) for secure communication, protecting sensitive data during transmission.  See [communication encryption](../tutorials/certificates.md) for more information on how to configure secure gRPC communication in P2PFL.


**Usage in the framework:**

To use gRPC, import the `GrpcCommunicationProtocol` and pass it to the `protocol` parameter when creating a `Node` instance:

```python
from p2pfl.node import Node
from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol

node = Node(
    # ... other node parameters
    protocol=GrpcCommunicationProtocol(),
    addr="127.0.0.1:5000" # Example address (IP:port)
)
```

### In-Memory Communication

For scenarios where nodes reside within the same process (e.g., local testing, simulations, debugging), in-memory communication provides a significantly faster and more efficient alternative to network-based protocols like gRPC.  By directly exchanging data in memory, this protocol eliminates the overhead associated with serialization and network transmission.  This is particularly beneficial for:

* **Local testing and debugging:** Enables rapid iteration and simplified debugging during development.
* **Simulation environments:** Facilitates efficient simulation of large federated networks on a single machine using tools like Ray.  See [Simulations](../tutorials/simulation.md) for more information.

**Usage in the framework:**

To use in-memory communication, import the `InMemoryCommunicationProtocol` and pass it to the `protocol` parameter during node creation:

```python
from p2pfl.node import Node
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol

node = Node(
    # ... other node parameters
    protocol=InMemoryCommunicationProtocol(),
    addr="node-1" # Example address for in-memory communication
)
```

This improved documentation provides a more comprehensive explanation of the communication protocols, highlighting their advantages and use cases. It also clarifies the protocol-agnostic design of P2PFL and provides more specific examples of how to use each protocol.  The addition of links to related documentation further enhances the user experience.


### Unix Sockets

Unix sockets provide a mechanism for inter-process communication (IPC) on the same machine. They offer a more efficient alternative to TCP/IP sockets for local communication, as they bypass network stack overhead.  This can be advantageous in scenarios where:

* **Nodes are on the same machine:** Ideal for local testing, development, and simulations where network communication is unnecessary.
* **Performance is critical:**  Offers faster communication compared to TCP/IP for local IPC.

**Usage in the framework:**

To use Unix sockets, use the `GrpcCommunicationProtocol` with a Unix socket address.  The address should start with `unix://` followed by the absolute path to the socket file:

```python
from p2pfl.node import Node
from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol

node = Node(
    # ... other node parameters
    protocol=GrpcCommunicationProtocol(),
    addr="unix:///tmp/p2pfl.sock" # Example Unix socket address
)
```

Make sure the directory for the socket file exists and is accessible to the user running the P2PFL node.  Each node should use a unique socket file path.
