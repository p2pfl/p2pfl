import threading
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *

#####################
#    Heartbeater    #
#####################

class Heartbeater(threading.Thread):
    """
    Thread based heartbeater that sends a beat message to all the neighbors of a node every HEARTBEAT_FREC seconds.

    Args:
        nodo_padre (Node): Node that use the heartbeater.
    """
    def __init__(self, nodo_padre):
        threading.Thread.__init__(self)
        self.nodo_padre = nodo_padre
        self.terminate_flag = threading.Event()
        self.name = "heartbeater-" + str(self.nodo_padre.get_addr()[0]) + ":" + str(self.nodo_padre.get_addr()[1])

    def stop(self):
        """
        Stops the heartbeater.
        """
        self.terminate_flag.set()

    def run(self):
        """
        Send a beat every HEARTBEAT_FREC seconds to all the neighbors of the node.
        """
        while not self.terminate_flag.is_set():
            # We do not check if the message was sent
            #   - If the model is sending, a beat is not necessary
            #   - If the connection its down timeouts will destroy connections
            self.nodo_padre.broadcast(CommunicationProtocol.build_beat_msg(), is_necesary=False)
            self.terminate_flag.wait(HEARTBEAT_FREC)