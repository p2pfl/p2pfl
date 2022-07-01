from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.encrypter import RSACipher
from p2pfl.encrypter import AESCipher
from p2pfl.learning.pytorch.lightninglearner import LightningLearner
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
    encrypted_message = cipher.encrypt(cipher.encode_with_padding(message))
    decrypted_message = cipher.decode_without_padding(cipher.decrypt(encrypted_message))
    assert message == decrypted_message

def test_aes_encription_decription2():
    cipher1 = AESCipher()
    cipher2 = AESCipher(key=cipher1.get_key())
    message="balblablabalbalbalablabal"
    encrypted_message = cipher1.encrypt(cipher1.encode_with_padding(message))
    decrypted_message = cipher2.decode_without_padding(cipher2.decrypt(encrypted_message))
    assert message == decrypted_message

def test_aes_encription_decription_model():
    cipher = AESCipher()
    nl = LightningLearner(MLP(), None)
    encoded_parameters = nl.encode_parameters()

    messages = CommunicationProtocol.build_params_msg(encoded_parameters)

    encrypted_messages = [cipher.encrypt(x) for x in messages]
    decrypted_messages = [cipher.decrypt(x) for x in encrypted_messages]

    for i in range(len(messages)):
        assert messages[i] == decrypted_messages[i]