import threading
from p2pfl.const import *

############################
# ESTO ES MAS UN HEARTBEAT #
############################

class Heartbeater(threading.Thread):
    def __init__(self, nodo_padre):
        
        threading.Thread.__init__(self)
        
        self.nodo_padre = nodo_padre
        self.terminate_flag = threading.Event()
        #self.dead_time = 30  # time to disconect from node if not pinged

    def stop(self):
        self.terminate_flag.set()

    #SE ENVIA PING QUE LUEGO RECIBIREMOS EN EL CONNECTION NODE
    def run(self):
        while not self.terminate_flag.is_set(): 
            self.nodo_padre.broadcast(BEAT.encode("utf-8"))   #cambiar mensaje de ping
            self.terminate_flag.wait(5)