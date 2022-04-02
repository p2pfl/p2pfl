import threading
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *

############################
# ESTO ES MAS UN HEARTBEAT #
############################

class Heartbeater(threading.Thread):
    def __init__(self, nodo_padre):
        threading.Thread.__init__(self)
        self.nodo_padre = nodo_padre
        self.terminate_flag = threading.Event()

    def stop(self):
        self.terminate_flag.set()

    #SE ENVIA PING QUE LUEGO RECIBIREMOS EN EL CONNECTION NODE
    def run(self):
        while not self.terminate_flag.is_set(): 
            self.nodo_padre.broadcast(CommunicationProtocol.BEAT.encode("utf-8"))
            self.terminate_flag.wait(5)