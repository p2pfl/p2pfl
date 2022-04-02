from encodings import utf_8
from re import U
import socket
import threading
import logging
from urllib import response
from p2pfl.const import *


# METER CONTROL DE ESTADOS PARA EVITAR ATAQUES

class NodeConnection(threading.Thread):

    def __init__(self, nodo_padre, socket, addr, buffer=""):

        threading.Thread.__init__(self)

        self.terminate_flag = threading.Event()
        self.nodo_padre = nodo_padre
        self.socket = socket
        self.empty_msgs = 0
        self.addr = addr


    def get_addr(self):
        return self.addr

    def run(self):
        self.socket.settimeout(TIEMOUT)

        while not self.terminate_flag.is_set():
            try:
                msg = self.socket.recv(BUFFER_SIZE)
                self.__event_handler(msg)        
                
            except socket.timeout:
                logging.debug("{} (NodeConnection) Timeout".format(self.get_addr()))
                self.terminate_flag.set()
                break

            except Exception as e:
                logging.debug("{} (NodeConnection) Exception: ".format(self.get_addr()) + str(e))
                logging.exception(e)
                self.terminate_flag.set()
                break
        
        #Bajamos Nodo
        logging.debug("Closed connection with {}".format(self.get_addr()))
        self.nodo_padre.rm_neighbor(self)
        self.socket.close()


    def __event_handler(self,msg):
        action = msg.decode("utf-8").split()

        if len(action) > 0:

            if action[0] == BEAT:
                pass #logging.debug("Beat {}".format(self.get_addr()))

            elif action[0] == STOP:
                self.send(EMPTY.encode("utf-8")) #esto es para que se actualice el terminate flag -> mirar otra forma de hacer downs instantáneos
                self.terminate_flag.set()

            elif action[0] == CONN_TO:
                if len(action) > 2:
                    print(action[1])
                    print(action[2])

                    self.nodo_padre.connect_to(action[1], int(action[2]), full=False)
            
            else:
                print("Nao Comprendo (" + msg.decode("utf-8") + ")")

        else:
            self.empty_msgs += 1
            if self.empty_msgs > 10:
                self.terminate_flag.set()

    def send(self, data): 
        try:
            self.socket.sendall(data)
        except Exception as e:
            logging.debug("(Node Connection) Exception: " + str(e))
            logging.exception(e) 
            self.terminate_flag.set() #exit

    # No es un stop instantáneo -> ESTE STOP ES PARA INICIAR STOPS EDUCADOS (desde fuera), no para desconexiones abruptas
    def stop(self):
        self.send(STOP.encode("utf-8"))
        self.terminate_flag.set()

