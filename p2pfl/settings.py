"""
Module to define constants for the p2pfl system.
"""

###################
# Global Settings #
###################

import logging
from p2pfl.encrypter import AESCipher


class Settings():
    """
    Class to define global settings for the p2pfl system.
    """

    BUFFER_SIZE = 8192 # Try to avoid model sobresegmentation
    rest = BUFFER_SIZE % AESCipher.get_block_size()
    if rest != 0:
        new_value = BUFFER_SIZE + AESCipher.get_block_size() - rest
        logging.error("Changing buffer size to %d. %d is incompatible with the AES block size.", BUFFER_SIZE, new_value)
        BUFFER_SIZE = new_value
    SOCKET_TIEMOUT = 15
    HEARTBEAT_FREC = 5
    AGREGATION_TIEMOUT = 15
    TRAIN_SET_SIZE = 2
    AMOUNT_LAST_MESSAGES_SAVED = 100 # Used to control gossiping
    GOSSIP_FREC = 100 # X rounds per second
    GOSSIP_ROUND_SENDINGS = 100 # send X messages per round



    GOSSIP_MODEL_FREC = 5 # X times per second
    GOSSIP_MODEL_SENDS_BY_ROUND = 1