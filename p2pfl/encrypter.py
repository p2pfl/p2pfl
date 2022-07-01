from Crypto import Random
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes



class Encrypter:
    """
    Class with methods to encrypt and decrypt messages.
    """

    def encrypt(message, key):
        pass

    def decrypt(message, key):      
        pass




class RSACipher(Encrypter):
    """
    Class with methods to encrypt and decrypt messages using RSA encryption.
    """

    def __init__(self, gen_keys=True):
        random_generator = Random.new().read
        self.private_key = RSA.generate(1024, random_generator)
        self.public_key = self.private_key.publickey()
        self.pair_public_key = None

    def load_pair_public_key(self, key):
        """
        Loads the public key of the pair.

        Args:
            key: The key to use to decrypt the message encoded at base64.
        """
        self.pair_public_key = RSA.importKey(base64.b64decode(key))

    def serialize_key(self):
        """
        Serializes a RSA public key.
        
        Returns:
            key: (str) The serialized key encoded at base64.
        """
        return base64.b64encode(self.public_key.exportKey("DER"))

    def encrypt(self, message):
        """
        Encrypts a message using RSA. Its encripted with the public key of the pair.
        
        Args:
            message: (str) The message to encrypt.

        Returns:
            message: (str) The encrypted message.
        """
        cipher = PKCS1_OAEP.new(self.pair_public_key)
        return base64.b64encode(cipher.encrypt(message))

    def decrypt(self, message):
        """
        Decrypts a message using RSA. Its decripted with the private key.

        Args:
            message: (str) The message to decrypt.

        Returns:
            message: (str) The decrypted message.
        """
        cipher = PKCS1_OAEP.new(self.private_key)
        return cipher.decrypt(base64.b64decode(message))

#
#
# REVISAR MODOS AES Y DOCUMENTAR
#
#

class AESCipher(Encrypter):
    """
    Class with methods to encrypt and decrypt messages using AES encryption.
    """
    
    def __init__(self, key=None): 
        self.bs = AES.block_size
        self.key = key
        if key is None:
            self.key = get_random_bytes(16) # 256 bits
        self.cipher = AES.new(self.key, AES.MODE_ECB)


    def encrypt(self, raw):
        return self.cipher.encrypt(raw)

    def decrypt(self, enc):
        return self.cipher.decrypt(enc)
    
    def add_padding(self, msg):
        """
        Add padding to a encoded UTF-8 text. Adds " " charactets (1 byte) to fill the rest of the block.

        Args:
            msg: (str) The encoded text.
        
        Returns:
            msg: (str) The encoded text with padding.
        """
        # Calculate the number of bytes needed to fill the block
        bytes_left = self.bs - len(msg) % self.bs
        if bytes_left == self.bs:
            bytes_left = 0

        # Add padding
        pading_char = " ".encode("utf-8")
        for i in range(bytes_left):
            msg += pading_char

        return msg 

    def get_key(self):
        return self.key

    def get_block_size():
        """
        Returns:
            block_size: (int) The length of the block in bytes.
        """
        return AES.block_size

    def key_len():
        """ 
        Returns:
            key_len: (int) The length of the key in bytes.
        """
        return 256