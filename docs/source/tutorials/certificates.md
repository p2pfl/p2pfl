# 🛡️ Communication Encryption with Mutual TLS

P2PFL prioritizes secure communication between nodes in the decentralized federated learning network. To achieve this, it leverages **mutual TLS (mTLS)**, a robust security protocol that ensures both confidentiality and authentication.

## Why Mutual TLS?

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

In this section we will review the steps we consider necessary for the generation of the TLS certificates for P2PFL.

1. **Root CA Generation:** Generate the root CA's public certificate (`CA.crt`) and private key (`CA.key`).

    These files are critical for ensuring the integrity and authenticity of certificates issued for nodes within the P2PFL network. They should be created in a secure environment, ideally using a tool such as OpenSSL, and the private key must be stored in a secure location, as its compromise could undermine the security of the entire certificate chain.

    For P2PFL certificate generation we used `openssl`. First, we will generate the private key as `ca.key` using RSA algorithm and AES-256 encryption. Note that we used a password (`CA_PASS`) to encrypt the private key. 

    ```bash
    openssl genpkey -algorithm RSA -out ca.key -aes256 -pass pass:${CA_PASS}
    ```

    Once we generated the CA private key, we can create a self-signed X.509 certificate with `openssl` with our previously generated private key. For this step, we can pre-fill some configuration information (`openssl.cnf`) for the certificate such as such as the organization, common name and extensions. For this example we will generate our certificate with SHA256 hashing algorithm and with a validity period of 1024 days. Note that we still need to have access to the `CA_PASS` (as every time we need to sign some certificate).

    ```bash
    openssl genpkey -algorithm RSA -out ca.key -aes256 -pass pass:${CA_PASS}
    ```

2. **Node Certificate Generation:** For each node, generate the following, ensuring they are signed by the root CA:
    *   `server.crt`: The node's public key certificate for server functionality.
    *   `server.key`: The corresponding private key for the server certificate.
    *   `client.crt`: The node's public key certificate for client functionality.
    *   `client.key`: The corresponding private key for the client certificate.

    To create client and server keys and certificates, we can also use `openssl`. For this case, certificate generation is exactly the same for the server and the client, so wi will use `node` as a variable that can be either a client or a server. We will, again, start by generating the private key using the RSA algorithm and reusing the password of the CA for simplicity:

    ```bash
    openssl genpkey -algorithm RSA -out node.key -pass pass:${CA_PASS}
    ```

    After we created our private key, we can then generate a Certificate Signing Request, which is basically a request for a digital certificate.

    ```bash
    openssl req -new -key node.key -out node.csr -config openssl.cnf
    ```

    Now, in the CA, we can sign the certificate signing request resulting in the creation of a signed X.509 certificate using the SHA256 algorithm. Also note that we provided a configuration file `node.cnf` which specifies Subject Alternative Names to use in the cerrtificate. This can be re-utilized for both server and client.

    ```bash
    openssl x509 -req -in node.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out node.crt -days 500 -sha256 -extfile node.cnf -extensions v3_req -passin pass:${CA_PASS}
    ```

    > **Important:** Ensure that the Subject Alternative Names (SANs) in these certificates accurately reflect the IP addresses or domain names of each node.

3. **Certificate Distribution:** Securely distribute the generated certificates to each node. Each node needs its `server.key`, `client.key`, and the root CA's public certificate (`CA.crt`).

    However, certificate distribution is not something that our library provides out-of-the-box. What was shown here is a simplified version of the process. In real-world environments, certificate distribution can become far more complex due to several key factors:

    * Distributing private keys securely is fundamental. In practice, this requires secure transfer protocols or the use of hardware security modules (HSMs) for key management.
    * As the number of nodes in the network grows, manually managing and distributing certificates becomes increasingly challenging. This requires a centralized or automated approach to manage the issuance, distribution, renewal, and revocation.

4. **Node Configuration:** Configure each node to use the appropriate certificates and keys by specifying their file paths in the node's settings.

    To enable mTLS in your P2PFL setup, you need to set the `USE_SSL` flag to `True` in your `Settings` and provide the paths to your generated certificates:

    ```python
    from p2pfl.settings import Settings
    
    Settings.ssl.USE_SSL = True
    Settings.ssl.CA_CRT = "path/to/your/ca.crt"
    Settings.ssl.SERVER_CRT = "path/to/your/server.crt"
    Settings.ssl.SERVER_KEY = "path/to/your/server.key"
    Settings.ssl.CLIENT_CRT = "path/to/your/client.crt"
    Settings.ssl.CLIENT_KEY = "path/to/your/client.key"
    ```
    
    By following these steps, you can secure your P2PFL network with mTLS, ensuring that only authenticated nodes can participate in the federated learning process.
