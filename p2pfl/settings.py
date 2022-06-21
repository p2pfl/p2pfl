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

    BUFFER_SIZE = 8192 # the buffer size should be coherent with the ammount of mesaages ant the size of models
    SOCKET_TIEMOUT = 15
    HEARTBEAT_FREC = 5
    MAX_ERRORS = 1
    AGREGATION_TIEMOUT = 15
