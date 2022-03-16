import threading
import time


class Pinger(threading.Thread):
    def __init__(self, nodo_padre):
        
        threading.Thread.__init__(self)
        
        self.nodo_padre = nodo_padre
        self.terminate_flag = threading.Event()
        self.dead_time = 30  # time to disconect from node if not pinged

    def stop(self):
        self.terminate_flag.set()

    #SE ENVIA PING QUE LUEGO RECIBIREMOS EN EL CONNECTION NODE
    def run(self):
        while not self.terminate_flag.is_set():  
            self.nodo_padre.broadcast("ping")
            time.sleep(20)
