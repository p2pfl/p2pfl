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

    def run(self):
        while not self.terminate_flag.is_set():
            # No nos cercioramos de que se envíen (el socket puede estar ocupado o caido) 
            #   - Si está ocupado no hace falta enviar nada
            #   - Si está caido ya vencerá el timeout y eliminará el nodo
            self.nodo_padre.broadcast(CommunicationProtocol.build_beat_msg()) 
            self.terminate_flag.wait(HEARTBEAT_FREC)