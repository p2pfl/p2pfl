"""
Module to define constants for the p2pfl system.
"""

import logging
from p2pfl.encrypter import AESCipher

###################
# Global Settings #
###################

class Settings():
    """
    Class to define global settings for the p2pfl system.

    Attributes:
        BLOCK_SIZE (int): Max ammount of bytes to read from the buffer at a time. If ```BLOCK_SIZE`` is not divisible by the block size used for symetric encryption it will be rounded to the next closest value.
        
        NODE_TIMEOUT (int): Timeout (seconds) for a node to be considered dead.
        VOTE_TIMEOUT (int): Timeout (seconds) for a node to wait for a vote.
        AGREGATION_TIMEOUT (int): Timeout (seconds) for a node to wait for other models.
        
        HEARTBEAT_PERIOD (int): Period (seconds) for the node to send a heartbeat.
        HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD (int): Times by period to refresh the neighbors list.
        WAIT_HEARTBEATS_CONVERGENCE (int): Time (seconds) to wait for the heartbeats to converge.

        TRAIN_SET_SIZE (int): Size of the training set.

        AMOUNT_LAST_MESSAGES_SAVED (int): Amount of last messages saved. Used to avoid gossiping cycles.
        GOSSIP_MESSAGES_FREC (int): Frequency of gossiping messages (times by second).
        GOSSIP_MESSAGES_PER_ROUND (int): Amount of gossip messages per round.

        GOSSIP_EXIT_ON_X_EQUAL_ROUNDS (int): Amount of equal rounds to exit gossiping. Careful, a low value can cause an early stop of gossiping.
        GOSSIP_MODELS_FREC (int): Frequency of gossiping models (times by second).
        GOSSIP_MODELS_PER_ROUND (int): Amount of gossip models per round.
        FRAGMENTS_DELAY (int): Delay (seconds) to wait before sending a fragment. This is a very important value, if the node is too slow and the buffer isn't big enough, the node will send fragments too fast and the other node will not receive them.
    """

    # TCP Block Size
    BLOCK_SIZE = 8192 # Try to avoid model sobresegmentation
    rest = BLOCK_SIZE % AESCipher.get_block_size()
    if rest != 0:
        new_value = BLOCK_SIZE + AESCipher.get_block_size() - rest
        logging.info("Changing buffer size to %d. %d is incompatible with the AES block size.", BLOCK_SIZE, new_value)
        BLOCK_SIZE = new_value

    # TIMEOUT'S
    NODE_TIMEOUT = 5
    VOTE_TIMEOUT = 10
    AGREGATION_TIMEOUT = 15

    # HEARTBEATER
    HEARTBEAT_PERIOD = 2
    HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD = 1
    WAIT_HEARTBEATS_CONVERGENCE = 5

    # FEDERATED 
    TRAIN_SET_SIZE = 5

    # GOSSIP Messages
    AMOUNT_LAST_MESSAGES_SAVED = 100 
    GOSSIP_MESSAGES_FREC = 100
    GOSSIP_MESSAGES_PER_ROUND = 100

    # GOSSIP Models
    GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 9
    GOSSIP_MODELS_FREC = 2  
    GOSSIP_MODELS_PER_ROUND = 2
    FRAGMENTS_DELAY = 0.01