# 💳 Certificates

P2PFL offers simple configuration files for certificate generation used in our CI pipeline to enable mutual TLS (mTLS) for the gRPC communications between pairs.

## 1. Overview

P2PFL utilizes a two-tiered certificate structure:

1. **Root Certificate Authority (CA):** The root CA acts as the trust anchor for the entire network.

2. **Node Certificates:** Each node in the P2PFL network receives two certificates (both server and client) signed by the Root CA. These certificates identify the nodes and allow them to authenticate themselves to their peers.

## 2. Root Certificate

The root certificate is the foundation of trust. It comprises two crucial files:

* The public certificate of the Root CA. It's distributed to all nodes in the network so they can verify the authenticity of other nodes' certificates.

* The private key of the Root CA.  **It must be kept secure and confidential.**  This key is used to sign the node certificates and should *never* be shared. The loss or theft of this key compromises the entire network's security.

## 3. Node Certificates

Each node requires two certificates, one for acting as a server and another for acting as a client when communicating with peers. These certificates, both signed by the root CA, enable mutual authentication, ensuring that each node can verify the identity of its peers and establish trust within the network.

Analogous to the previous example, each node possesses two public key certificates, one designated for the server role and the other for the client role, utilized for authentication by peer nodes. Correspondingly, each node also maintains two private keys, associated with their respective certificates, employed for digitally signing and decrypting communications of both client and server roles.

## 4. Certificate Generation

The certificate generation procedure comprises the following steps:

1. **Root CA Generation**: The first step in the process is the generation of the public certificate (`CA.crt`) and private key (`CA.key`) files, constituting the root CA.

2. **Node Certificate Generation**: For each node within the network, the following certificates and keys must be generated and appropriately signed by the aforementioned Root CA:

    * `server.crt`: The node's public key certificate for server functionality.
    * `server.key`: The corresponding private key for the server certificate.
    * `client.crt`: The node's public key certificate for client functionality.
    * `client.key`: The corresponding private key for the client certificate.

    It is imperative to ensure that the Subject Alternative Names (SANs) within these certificates accurately reflect the respective IP addresses or domain names associated with each node.

3. **Certificate Distribution**: Following generation, the generated certificates must be securely disseminated to each node. Each node requires its corresponding private keys (`server.key` and `client.key`) and the root CA's public certificate (`CA.crt`).

4. **Node Configuration**: The generated certificates and keys must be securely deployed and configured within each node's application settings. This typically involves specifying the file paths to the respective certificate and key files. Ensure appropriate access controls are in place to protect the private keys.

---

**WARNING**: Using the example configuration files in a production environment is strongly discouraged and could compromise the security of the enviroment. System administrators should manage the certificate generation and distribution process using appropriate security measures for the specific scenario.

---


🌟 Ready? **You can view next**: > [mTLS](docs-tls.md)
