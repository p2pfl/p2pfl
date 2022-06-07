import threading
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *

#####################
#    Heartbeater    #
#####################

class Heartbeater(threading.Thread):
    def __init__(self, nodo_padre):
        threading.Thread.__init__(self)
        self.nodo_padre = nodo_padre
        self.terminate_flag = threading.Event()
        self.name = "heartbeater-" + str(self.nodo_padre.get_addr()[0]) + ":" + str(self.nodo_padre.get_addr()[1])

    def stop(self):
        self.terminate_flag.set()

    def run(self):
        while not self.terminate_flag.is_set():
            # We do not check if the message was sent
            #   - If the model is sending, a beat is not necessary
            #   - If the connection its down timeouts will destroy connections
            self.nodo_padre.broadcast(CommunicationProtocol.build_beat_msg()) 
            self.terminate_flag.wait(HEARTBEAT_FREC)