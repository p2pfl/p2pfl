# 📡 Communication Protocols

The communication protocols define how the various nodes interact and exchange data. These protocols are essential for facilitating communication between peers in the network. This section provides an overview of the supported protocols in the current version of the framework.

## Supported Protocols

1. **gRPC (gRPC Remote Procedure Call):**
   gRPC is a high-performance, open-source universal RPC framework that enables efficient communication between distributed systems. In this framework, gRPC is used for reliable, secure, and scalable data exchange between peers. Its key features include:
   - **Cross-platform compatibility:** Supports multiple languages and platforms.
   - **Streamlined communication:** Enables bidirectional streaming and efficient binary serialization using Protocol Buffers.
   - **Scalability:** Optimized for large-scale federated networks.

    **Usage in the framework:**
    Pass the class `GrpcCommunicationProtocol` to the parameter `protocol` when creating the Node.
    ```python
    node = Node(
        ...
        protocol=GrpcCommunicationProtocol,
    )
    ```

2. **In-Memory Communication:**
   For setups where peers are running within the same process or closely connected systems, in-memory communication offers a lightweight and efficient alternative to network-based communication. This protocol reduces overhead by bypassing serialization and network stacks, making it ideal for:
   - **Local testing and debugging:** Enables rapid iteration during development.
   - **Simulation environments:** Facilitates scenarios where multiple peers are simulated on a single machine.

    **Usage in the framework:**
    Pass the class `InMemoryCommunicationProtocol` to the parameter `protocol` when creating the Node.
    ```python
    node = Node(
        ...
        protocol=InMemoryCommunicationProtocol,
    )
    ```

🌟 Ready? **You can view next**: > [Workflows](docs-workflows.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-workflows.md">Workflows</a> </div>