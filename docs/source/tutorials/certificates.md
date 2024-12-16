# 🛡️ Communication Encryption with Mutual TLS (mTLS)

P2PFL prioritizes secure communication between nodes in the decentralized federated learning network. To achieve this, it leverages **mutual TLS (mTLS)**, a robust security protocol that ensures both confidentiality and authentication.

## Why Mutual TLS (mTLS)?

In traditional TLS, only the server authenticates itself to the client. However, in a true peer-to-peer (P2P) environment like P2PFL, there's no fixed server-client relationship. Each node acts as both a client and a server, initiating and receiving connections. Requiring only one side to present a certificate would leave the network vulnerable to unauthorized participants and impersonation attacks.

mTLS addresses this by requiring **both** peers to present digital certificates, verifying each other's identities before establishing a connection. This bi-directional authentication is crucial for:

*   **Security:** Prevents unauthorized nodes from joining the network and mitigates impersonation attacks.
*   **Trust:** Builds trust between nodes by ensuring they are communicating with verified participants.
*   **Privacy:** Encrypts communication, protecting sensitive data like model updates during transmission.

## mTLS in P2PFL with gRPC

P2PFL uses **gRPC**, a high-performance RPC framework, for communication between nodes. gRPC seamlessly integrates with mTLS, handling the certificate exchange and verification process transparently.

In P2PFL, each node possesses two certificates signed by a trusted **Root Certificate Authority (CA)**:

*   **Server Certificate:** Presented when the node acts as a gRPC server, receiving connections.
*   **Client Certificate:** Presented when the node acts as a gRPC client, initiating connections.

When two nodes establish a connection, the following steps occur:

1. **Handshake:** The gRPC server presents its server certificate to the client.
2. **Client Verification:** The client verifies the server's certificate against its trusted root CA. If the certificate is invalid or not trusted, the connection is refused.
3. **Client Authentication:** The gRPC client presents its client certificate to the server.
4. **Server Verification:** The server verifies the client's certificate against the trusted root CA. If the certificate is invalid or not trusted, the connection is refused.
5. **Secure Connection:** If both certificates are valid and trusted, a secure, encrypted connection is established.

## Certificate Structure and Generation

P2PFL employs a two-tiered certificate structure:

*   **Root Certificate Authority (CA):** The foundation of trust. The CA's public certificate is distributed to all nodes, enabling them to verify other nodes' certificates. The CA's private key **must be kept secure** as it's used to sign node certificates.
*   **Node Certificates:** Each node has a server certificate and a client certificate, both signed by the root CA.

> **Warning:** The example configuration files provided with P2PFL are for testing purposes only and **should not be used in production environments**. System administrators should manage certificate generation and distribution using appropriate security measures.

### Certificate Generation Procedure

1. **Root CA Generation:** Generate the root CA's public certificate (`CA.crt`) and private key (`CA.key`).
2. **Node Certificate Generation:** For each node, generate the following, ensuring they are signed by the root CA:
    *   `server.crt`: The node's public key certificate for server functionality.
    *   `server.key`: The corresponding private key for the server certificate.
    *   `client.crt`: The node's public key certificate for client functionality.
    *   `client.key`: The corresponding private key for the client certificate.

    > **Important:** Ensure that the Subject Alternative Names (SANs) in these certificates accurately reflect the IP addresses or domain names of each node.

3. **Certificate Distribution:** Securely distribute the generated certificates to each node. Each node needs its `server.key`, `client.key`, and the root CA's public certificate (`CA.crt`).

4. **Node Configuration:** Configure each node to use the appropriate certificates and keys by specifying their file paths in the node's settings.

## Enabling mTLS in P2PFL

To enable mTLS in your P2PFL setup, you need to set the `USE_SSL` flag to `True` in your `Settings` and provide the paths to your generated certificates:

```python
from p2pfl.settings import Settings

Settings.USE_SSL = True
Settings.CA_CRT = "path/to/your/ca.crt"
Settings.SERVER_CRT = "path/to/your/server.crt"
Settings.SERVER_KEY = "path/to/your/server.key"
Settings.CLIENT_CRT = "path/to/your/client.crt"
Settings.CLIENT_KEY = "path/to/your/client.key"
```

By following these steps, you can secure your P2PFL network with mTLS, ensuring that only authenticated nodes can participate in the federated learning process.
