"""
Module to define constants for the p2pfl system.
"""

###################
# Global Settings #
###################

class Settings():
    """
    Class to define global settings for the p2pfl system.
    """

    BUFFER_SIZE = 8192 # Try to avoid model sobresegmentation
    SOCKET_TIEMOUT = 15
    HEARTBEAT_FREC = 5
    MAX_ERRORS = 1
    AGREGATION_TIEMOUT = 15
    TRAIN_SET_SIZE = 1
    GOSSIP_MODEL_FREC = 5 # X times per second
    GOSSIP_MODEL_SENDS_BY_ROUND = 1