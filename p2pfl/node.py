import socket
import threading
import logging
import sys
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater


# https://github.com/GianisTsol/python-p2p/blob/master/pythonp2p/node.py

# Observacones:
#   - Para el traspaso de modelos, sea mejor crear un socket a parte

BUFFER_SIZE = 1024
HI_MSG = "hola"

"""
from p2pfl.node import Node

n1 = Node("localhost",6777)
n1.start()

from p2pfl.node import Node

n2 = Node("localhost",6778)
n2.start()
n2.connect_to("localhost",6777)
"""
#que extienda de thread?
class Node(threading.Thread):
    def __init__(self, host, port=0):

        threading.Thread.__init__(self)

        self.terminate_flag = threading.Event()
        self.host = host
        self.port = port

        # Setting Up Node Socket (listening)
        self.node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP
        #self.node_socket.settimeout(0.2)
        self.node_socket.bind((host, port))
        if port==0:
            self.port = self.node_socket.getsockname()[1]
        
        self.node_socket.listen(5)# no mas de 5 peticones a la cola

        # Neightboors
        self.neightboors = []

        #Loggin ? por ver
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        self.heartbeater = Heartbeater(self)
        self.heartbeater.start()


    def get_addr(self):
        return self.host,self.port

    #Objetivo: Agregar vecinos a la lista -> CREAR POSIBLES SOCKETS 
    def run(self):
        logging.info('Nodo a la escucha en {} {}'.format(self.host, self.port))
        while not self.terminate_flag.is_set(): 
            try:
                (node_socket, addr) = self.node_socket.accept()
                
                # MSG
                msg = node_socket.recv(BUFFER_SIZE).decode("UTF-8")
                splited = msg.split("\n")
                head = splited[0]
                rest = "\n".join(splited[1:])
                
                #
                # EL HI SERÁ en un futuro la encriptación
                #
                if head == HI_MSG:
                    logging.info('Conexión aceptada con {}'.format(addr))

                    nc = NodeConnection(self,node_socket,rest)
                    nc.start()
                    self.add_neighbor(nc)

                else:
                    logging.debug('Conexión rechazada con {}:{}'.format(addr,msg))
                    node_socket.close()
           
            except Exception as e:
                #revisar excepciones de timeout y configurarlas
                print(e)

        #Detenemos nodo
        logging.info('Bajando el nodo, dejando de escuchar en {} {}'.format(self.host, self.port))
        self.heartbeater.stop()
        for n in self.neightboors:
            n.stop()
        self.node_socket.close()

    def stop(self): 
        self.terminate_flag.set()
        # Enviamos mensaje al loop para evitar la espera del recv
        try:
            self.__send(self.host,self.port,b"")
        except:
            pass

    ##########################
    #     Msg management     #
    ##########################

    def __send(self, h, p, data, persist=False): 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((h, p))
        s.sendall(data)
        if persist:
            return s
        else:
            s.close()
            return None

    def connect_to(self, h, p): 
        msg=(HI_MSG + "\n").encode("utf-8")
        s = self.__send(h,p,msg,persist=True)
        
        # Agregaos el vecino
        nc = NodeConnection(self,s,"")
        nc.start()
        self.add_neighbor(nc)

    def broadcast(self, msg, exc=None):
        for n in self.neightboors:
            if n != exc:
                n.send(msg)

    #############################
    #  Neighborhood management  #
    #############################

    def get_neighbors(self):pass
    def add_neighbor(self, n):
        self.neightboors.append(n)

    def rm_neighbor(self,n):
        try:
            self.neightboors.remove(n)
        except:
            pass
  