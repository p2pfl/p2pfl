import socket
import threading
import logging
import sys
from p2pfl.const import *
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater


# https://github.com/GianisTsol/python-p2p/blob/master/pythonp2p/node.py

# Observacones:
#   - Para el traspaso de modelos, sea mejor crear un socket a parte
#   - Crear un enmascaramiento de sockets por si en algun futuro se quiere modificar

# REVISAR UN BROKEN PIPE AL CERRAR BRUSCAMENTE


############################################################################################
# FULL CONNECTED HAY QUE IMPLEMENTARLO DE FORMA QUE CUANDO SE INTRODUCE UN NODO EN LA RED, SE HACE UN BROADCAST
############################################################################################



BUFFER_SIZE = 1024

"""
from p2pfl.node import Node
n1 = Node(port=5555)
n1.start()

from p2pfl.node import Node
n2 = Node(port=6666)
n2.start()
n2.connect_to("127.0.0.1",5555)


from p2pfl.node import Node
n3 = Node(port=6779)
n3.start()
n3.connect_to("127.0.0.1",6666)
"""

class Node(threading.Thread):
    def __init__(self, host="127.0.0.1", port=0):

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
                close = False
                
                # MSG
                msg = node_socket.recv(BUFFER_SIZE).decode("UTF-8")
                splited = msg.split() #revisar cuando lo pnga en clase desacoplada para que siga el mismo formato
                #
                # HI -> ESTO ES MUY INSEGURO, PERO OREINTADO A METERLE ENCRIPTACION NO ME PREOCUPA
                #
                if len(splited) > 3:
                    if splited[0] == CONN:
                        try:
                            # Comprobamos que ip y puerto son correctos
                            source = (splited[1], int(splited[2]))
                            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s.settimeout(2)
                            result = s.connect_ex((source[0],source[1])) 
                            s.close()
                            # Agregaos el vecino
                            if result == 0:
                                logging.info('Conexión aceptada con {}'.format(source))
                                nc = NodeConnection(self,node_socket,source)
                                nc.start()
                                self.add_neighbor(nc)
                                #Notificamos a la red la conexion (ttl de 1 xq es full conectada)
                                if splited[3]=="1":
                                    self.broadcast((CONN_TO + " " + source[0] + " " + str(source[1])).encode("utf-8"),exc=[nc])

                        except Exception as e:
                            logging.exception(e)
                            self.rm_neighbor(nc)
                            close=True
                    else:
                        close=True
                else:
                        close=True
           
                if close:
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



               
    # TTL x implementar
    def broadcast(self, msg, ttl=1, exc=[]):
        for n in self.neightboors:
            if not (n in exc):
                print("Broadcasting to" + str(n.get_addr()) + ": " + msg.decode("UTF-8"))
                n.send(msg)

    #############################
    #  Neighborhood management  #
    #############################

    def get_neighbor(self, h, p):
        for n in self.neightboors:
            print(str(n.get_addr()) + str((h,p)))
            if n.get_addr() == (h,p):
                return n
        return None

    def get_neighbors(self):
        return self.neightboors

    def add_neighbor(self, n):
        self.neightboors.append(n)

    def rm_neighbor(self,n):
        try:
            self.neightboors.remove(n)
        except:
            pass
  
    def connect_to(self, h, p, full=True):
        if full:
            full = "1"
        else:
            full = "0"

        msg=(CONN + " " + str(self.host) + " " + str(self.port) + " " + full).encode("utf-8")
        s = self.__send(h,p,msg,persist=True)
        
        # Agregaos el vecino
        nc = NodeConnection(self,s,(h,p))
        nc.start()
        self.add_neighbor(nc)
      
