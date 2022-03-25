import socket
import threading
import logging
from p2pfl.const import *

class NodeConnection(threading.Thread):

    def __init__(self, nodo_padre, socket, buffer):

        threading.Thread.__init__(self)

        self.terminate_flag = threading.Event()
        self.nodo_padre = nodo_padre
        self.socket = socket
        self.buffer = buffer


    def get_addr(self):
        return self.socket.getsockname()

    #Connection loop
    #   -ping periódico
    #   -procesado de peticiones
    def run(self):
        self.socket.settimeout(TIEMOUT)

        while not self.terminate_flag.is_set():
            try:
                msg = self.socket.recv(BUFFER_SIZE)
                self.__do_things(msg)
                
            except socket.timeout:
                logging.debug("{} NodeConnection Timeout".format(self.get_addr()))
                self.stop()
                break

            except Exception as e:
                logging.debug("{} Exception: ".format(self.get_addr()) + str(e))
                self.stop()
                break

        
        #Bajamos Nodo
        logging.debug("Closed connection with {}".format(self.get_addr()))
        self.nodo_padre.rm_neighbor(self)
        self.socket.close()


    def __do_things(self, action):

        if action == BEAT:
            logging.debug("Beat {}".format(self.get_addr()))

        elif action == STOP:
            self.send(EMPTY) #esto es para que se actualice el terminate flag -> mirar otra forma de hacer downs instantáneos
            self.terminate_flag.set()

        elif action == "":
            pass
            
        else:
            print("Nao Comprendo (" + action + ")")


    def send(self, data): 
        self.socket.sendall(data)

    # No es un stop instantáneo
    def stop(self):
        self.send(STOP)
        self.terminate_flag.set()

