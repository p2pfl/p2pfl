# 🛡 Mutual TLS (mTLS)

This document explains how Mutual TLS (mTLS) is used in P2PFL to secure gRPC communication between peers.

## 1. Overview

P2PFL leverages mutual TLS (mTLS) to ensure secure and authenticated communication between peers over gRPC in a decentralized network. Unlike traditional TLS, where only the server authenticates itself to the client, which is fundamentally unsuitable for true peer-to-peer networks. In a P2P environment, there isn't a fixed server and client relationship; each peer acts as both a client and a server, initiating and receiving connections. Therefore, requiring only one side to present a certificate leaves the network vulnerable to unauthorized participants. mTLS addresses this by requiring both peers to present digital certificates, verifying each other's identities before establishing a connection. This bi-directional authentication significantly enhances P2P network security, preventing unauthorized peers from joining and mitigating impersonation attacks where a malicious node might try to pose as a legitimate participant. This mutual verification is crucial for building trust and ensuring secure communication within the decentralized structure of a P2P network.

## 2. mTLS and gRPC

gRPC, a high-performance RPC framework, seamlessly integrates with mTLS. When mTLS is enabled, gRPC clients and servers exchange certificates during the initial handshake. The gRPC framework handles the certificate verification process transparently, ensuring that only authenticated nodes can communicate.

## 3. How mTLS Works in P2PFL

In P2PFL, each node acts as both a gRPC client and server, communicating with other nodes in the network. Therefore, each node possesses two certificates:

* **Server Certificate:** Presented when the node acts as a gRPC server.
* **Client Certificate:** Presented when the node acts as a gRPC client.

When a gRPC connection is initiated between two P2PFL nodes:

1. The gRPC server presents its server certificate to the client.
2. The client verifies the presented server certificate against the trusted root CA. If the certificate is not present, or is not valid, the communication is refused.
3. The gRPC client presents its client certificate to the server.
4. The server verifies the presented client certificate against the trusted root CA. If the certificate is not present, or is not valid, the communication is refused.

If both certificates are valid and trusted, the connection is established, and secure communication can proceed. 

