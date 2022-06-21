import threading
import time
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *

#####################
#    Heartbeater    #
#####################

class Heartbeater(threading.Thread):
    """
    Thread based heartbeater that sends a beat message to all the neighbors of a node every `HEARTBEAT_FREC` seconds.

    Args:
        nodo_padre (Node): Node that use the heartbeater.
    """
    def __init__(self, nodo_padre):
        threading.Thread.__init__(self)
        self.nodo_padre = nodo_padre
        self.name = "heartbeater-" + str(self.nodo_padre.get_addr()[0]) + ":" + str(self.nodo_padre.get_addr()[1])

    def run(self):
        """
        Send a beat every HEARTBEAT_FREC seconds to all the neighbors of the node.
        """
        while not self.nodo_padre.is_stopped():
            # We do not check if the message was sent
            #   - If the model is sending, a beat is not necessary
            #   - If the connection its down timeouts will destroy connections
            self.nodo_padre.broadcast(CommunicationProtocol.build_beat_msg(), is_necesary=False)
            time.sleep(HEARTBEAT_FREC)