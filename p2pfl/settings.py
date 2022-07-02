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
    GOSSIP_MESSAGES_FREC = 100 # X rounds per second
    GOSSIP_MESSAGES_PER_ROUND = 100 # send X messages per round


    FRAGMENTS_DELAY = 0.1  # en un futuro molar'ia hacer un mecanismo que se adapte din'amicamente a la velocidad de los nodos

    GOSSIP_MODELS_FREC = 1  # X times per second | A really high value will make to send duplicated models (send twice before getting a models left update) 
                            # SE PODRÍa TRATAR DE NO MANDARLO ANTES DE OBTENER RESPUESTA
    GOSSIP_MODELS_PER_ROUND = 2