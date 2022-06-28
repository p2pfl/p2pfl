import time
import threading

class Gossiper(threading.Thread):
    
    # Meterlo como un observer -> Hacer del mismo modo el heartbeater

    def __init__(self, node):
        self.msgs = {}
        self.node = node
        threading.Thread.__init__(self, name=("gossiper-" + str(node.get_addr()[0]) + "-" + str(node.get_addr()[1]) ))
        self.add_lock = threading.Lock()
        self.terminate_flag = threading.Event()

    def add_messages(self, msgs, node):
        self.add_lock.acquire()
        for msg in msgs:
            self.msgs[msg] = [node]
        self.add_lock.release()

    def run(self):
        while not self.terminate_flag.is_set():

            # Seleccionar X mensajes y mandarlos a Y vecinos cada Z segundos

            self.add_lock.acquire()

            if len(self.msgs) > 0:
                for msg,nodes in self.msgs.items():
                    # Send to all the nodes except the ones that the message was already sent to
                    self.node.broadcast(msg,exc=nodes) 
                self.msgs = {}
            self.add_lock.release()
        

            time.sleep(1)

    def stop(self):
        self.terminate_flag.set()