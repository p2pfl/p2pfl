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

    # BUFFER
    BUFFER_SIZE = 8192 # Try to avoid model sobresegmentation
    rest = BUFFER_SIZE % AESCipher.get_block_size()
    if rest != 0:
        new_value = BUFFER_SIZE + AESCipher.get_block_size() - rest
        logging.error("Changing buffer size to %d. %d is incompatible with the AES block size.", BUFFER_SIZE, new_value)
        BUFFER_SIZE = new_value

    # TIMEOUT'S
    NODE_TIMEOUT = 5
    TIMEOUT_WAIT_VOTE = 10
    AGREGATION_TIMEOUT = 15

    # HEARTBEATER
    HEARTBEAT_PERIOD = 2
    HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD = 1

    # FEDERATED 
    TRAIN_SET_SIZE = 10

    # GOSSIP Messages
    AMOUNT_LAST_MESSAGES_SAVED = 100 # Used to control gossip messages
    GOSSIP_MESSAGES_FREC = 100 # X rounds per second
    GOSSIP_MESSAGES_PER_ROUND = 100 # send X messages per round

    # GOSSIP Models
    GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 3 # If X rounds are equal, exit gossiping
    GOSSIP_MODELS_FREC = 4  # X times per second | A really high value will make to send duplicated models (send twice before getting a models left update) 
    GOSSIP_MODELS_PER_ROUND = 100
    FRAGMENTS_DELAY = 0.005 # Delay between fragments (to avoid buffer overflow)