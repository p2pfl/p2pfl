import socket
import threading
import logging
from urllib import response
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *


# METER CONTROL DE ESTADOS PARA EVITAR ATAQUES

class NodeConnection(threading.Thread):

    def __init__(self, nodo_padre, socket, addr):
        threading.Thread.__init__(self)
        self.terminate_flag = threading.Event()
        self.nodo_padre = nodo_padre
        self.socket = socket
        self.errors = 0
        self.addr = addr
        self.comm_protocol = CommunicationProtocol({
            CommunicationProtocol.BEAT: self.__on_beat,
            CommunicationProtocol.STOP: self.__on_stop,
            CommunicationProtocol.CONN_TO: self.__on_conn_to,
        })


    def run(self):
        self.socket.settimeout(TIEMOUT)

        while not self.terminate_flag.is_set():
            try:
                # Recive and process messages
                msg = self.socket.recv(BUFFER_SIZE).decode("utf-8")
                if not self.comm_protocol.process_message(msg):
                    self.errors += 1
                    # If we have too many errors, we stop the connection
                    if self.errors > 10:
                        self.terminate_flag.set()
                        logging.debug("Too mucho errors. {}".format(self.get_addr()))

                
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

    def get_addr(self):
        return self.addr

    def send(self, data): 
        try:
            self.socket.sendall(data)
        except Exception as e:
            logging.debug("(Node Connection) Exception: " + str(e))
            logging.exception(e) 
            self.terminate_flag.set() #exit

    def stop(self):
        self.terminate_flag.set()
        self.send(CommunicationProtocol.STOP.encode("utf-8"))

    #########################
    #       Callbacks       #
    #########################

    def __on_beat(self):
        pass #logging.debug("Beat {}".format(self.get_addr()))

    def __on_stop(self):
            self.terminate_flag.set()
            self.send(CommunicationProtocol.STOP.encode("utf-8")) #esto es para que se actualice el terminate flag -> mirar otra forma de hacer downs instantáneos

    def __on_conn_to(self,h,p):
            self.nodo_padre.connect_to(h, p, full=False)



