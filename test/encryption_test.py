import time
import pytest
import torch

from p2pfl.node import Node
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.encrypter import RSACipher
from p2pfl.encrypter import AESCipher
from p2pfl.learning.pytorch.lightninglearner import LightningLearner
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP

#############################
#    RSA Encryption Test    #
#############################

def test_rsa_encription_decription1():
    rsa = RSACipher()
    rsa.load_pair_public_key(rsa.serialize_key())
    message = "Hello World!".encode("utf-8")
    encrypted_message = rsa.encrypt(message)
    decrypted_message = rsa.decrypt(encrypted_message)
    assert message == decrypted_message

def test_rsa_encription_decription2():
    cipher1 = RSACipher()
    cipher2 = RSACipher()
    cipher1.load_pair_public_key(cipher2.serialize_key())
    cipher2.load_pair_public_key(cipher1.serialize_key())
    # Exchange messages and check if they are the same
    message = "Buenas tardes Dani y Enrique!".encode("utf-8")
    encrypted_message1 = cipher1.encrypt(message)
    encrypted_message2 = cipher2.encrypt(message)
    decrypted_message1 = cipher2.decrypt(encrypted_message1)
    decrypted_message2 = cipher1.decrypt(encrypted_message2)
    assert message == decrypted_message1
    assert message == decrypted_message2

#############################
#    AES Encryption Test    #
#############################

def test_aes_encription_decription1():
    cipher = AESCipher()
    message="zzZZZZ!"
    encoded_mesage = cipher.add_padding(message.encode("utf-8"))
    encrypted_message = cipher.encrypt(encoded_mesage)
    decrypted_message = cipher.decrypt(encrypted_message)
    decoded_message = " ".join(decrypted_message.decode("utf-8").split())
    assert message == decoded_message

def test_aes_encription_decription2():
    cipher1 = AESCipher()
    cipher2 = AESCipher(key=cipher1.get_key())
    message="balb l    ablab alb  a lbalabla     bal"
    encoded_mesage = cipher1.add_padding(message.encode("utf-8"))
    encrypted_message = cipher1.encrypt(encoded_mesage)
    decrypted_message = cipher2.decrypt(encrypted_message)
    decoded_message = decrypted_message.decode("utf-8")
    assert message.split() == decoded_message.split()

def test_aes_encription_decription_model():
    cipher = AESCipher()
    nl = LightningLearner(MLP(), None)
    encoded_parameters = nl.encode_parameters()

    messages = CommunicationProtocol.build_params_msg(encoded_parameters)

    encrypted_messages = [cipher.encrypt(cipher.add_padding(x)) for x in messages]
    decrypted_messages = [cipher.decrypt(x) for x in encrypted_messages]

    for i in range(len(messages)):
        assert messages[i] == decrypted_messages[i]

#############################
#    Node Encrypted Test    #
#############################

@pytest.mark.parametrize('x',[(2,1),(2,2)]) 
def test_encrypted_convergence(x):
    n,r = x

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(),MnistFederatedDM(), simulation=False)
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes)-1):
        nodes[i+1].connect_to(nodes[i].host,nodes[i].port)
        time.sleep(0.1)
        
    # Overhead at handshake because of the encryption
    time.sleep(3)

    # Check if they are connected
    time.sleep(3)     
    for node in nodes:
        assert len(node.neightboors) == n-1

    # Start Learning
    nodes[0].set_start_learning(rounds=r,epochs=0)

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    # Validate that the models obtained are equal
    model = None
    first = True
    for node in nodes:
        if first:
            model = node.learner.get_parameters()
            first = False
        else:
            for layer in model:
                a = torch.round(model[layer], decimals=2)
                b = torch.round(node.learner.get_parameters()[layer], decimals=2)
                assert torch.eq(a, b).all()

    # Close
    for node in nodes:
        node.stop()
        time.sleep(.1) # Wait because of asincronity
