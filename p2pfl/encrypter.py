from Crypto import Random
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

class Encrypter:
    """
    Class with methods to encrypt and decrypt messages.
    """

    def encrypt(self, message):
        pass

    def decrypt(self, message):      
        pass

    def get_key(self):
        pass
    
###############################
#    Asymmetric Encryption    #
###############################

class RSACipher(Encrypter):
    """
    Class with methods to encrypt and decrypt messages using RSA asymetric encryption.
    """

    def __init__(self):
        random_generator = Random.new().read
        self.__private_key = RSA.generate(1024, random_generator)
        self.__public_key = self.__private_key.publickey()
        self.__pair_public_key = None

    def encrypt(self, message):
        """
        Encrypts a message using RSA. Message is encrypted using the public key of the pair (the other node key).
        
        Args:
            message: (bytes) The message to encrypt.

        Returns:
            message: (bytes) The encrypted message.
        """
        cipher = PKCS1_OAEP.new(self.__pair_public_key)
        return base64.b64encode(cipher.encrypt(message))

    def decrypt(self, message):
        """
        Decrypts a message using RSA. Message is decripted using the private key. 

        Args:
            message: (bytes) The message to decrypt.

        Returns:
            message: (bytes) The decrypted message.
        """
        cipher = PKCS1_OAEP.new(self.__private_key)
        return cipher.decrypt(base64.b64decode(message))

    def load_pair_public_key(self, key):
        """
        Loads the public key of the pair (other node).

        Args:
            key: The key to use to decrypt the message encoded at base64.
        """
        self.__pair_public_key = RSA.importKey(base64.b64decode(key))

    def get_key(self):
        """
        Get the serialized RSA public key.
        
        Returns:
            key: The serialized key encoded at base64.
        """
        return base64.b64encode(self.__public_key.exportKey("DER"))

##############################
#    Symmetric Encryption    #
##############################

class AESCipher(Encrypter):
    """
    Class with methods to encrypt and decrypt messages using AES symetric encryption.
    """
    
    def __init__(self, key=None): 
        self.bs = AES.block_size
        self.key = key
        if key is None:
            self.key = get_random_bytes(16) # 256 bits
        self.cipher = AES.new(self.key, AES.MODE_ECB)

    def encrypt(self, message):
        """
        Encrypts a message using AES. Message is encrypted using the shared key.
        Keep in mind that AES uses a block cipher, so the message sould be filled with padding.

        Args:
            message: (str) The message to encrypt.

        Returns:
            message: (str) The encrypted message. 
        """
        return self.cipher.encrypt(message)

    def decrypt(self, message):
        """
        Decrypts a message using AES. Message is decripted using the shared key. 
        Keep in mind that AES uses a block cipher, so the message can be a filled message with padding.

        Args:
            message: (bytes) The message to decrypt.

        Returns:
            message: (bytes) The decrypted message.
        """
        return self.cipher.decrypt(message)
    
    def add_padding(self, msg):
        """
        Add padding to a encoded UTF-8 text. Adds " " charactets (1 byte) to fill the rest of the block.
        Careful: for this case the filling content don't affect to the meaning of messages.

        Args:
            msg: (bytes) The encoded text.
        
        Returns:
            msg: (bytes) The encoded text with padding.
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
        """
        Get the shared RSA key.
        
        Returns:
            key: The shared key.
        """
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