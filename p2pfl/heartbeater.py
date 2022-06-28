import threading
import time
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable

#####################
#    Heartbeater    #
#####################

class Heartbeater(threading.Thread, Observable):
    """
    Thread based heartbeater that sends a beat message to all the neighbors of a node every `HEARTBEAT_FREC` seconds.

    Args:
        nodo_padre (Node): Node that use the heartbeater.
    """
    def __init__(self, node_name):
        Observable.__init__(self)
        threading.Thread.__init__(self, name = "heartbeater-" + node_name)
        self.terminate_flag = threading.Event()

    def run(self):
        """
        Send a beat every HEARTBEAT_FREC seconds to all the neighbors of the node.
        """
        while not self.terminate_flag.is_set():
            # We do not check if the message was sent
            #   - If the model is sending, a beat is not necessary
            #   - If the connection its down timeouts will destroy connections
            self.notify(Events.SEND_BEAT_EVENT, None)
            time.sleep(Settings.HEARTBEAT_FREC)

    def stop(self):
        self.terminate_flag.set()