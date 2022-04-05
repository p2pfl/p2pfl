import socket
import threading
import logging
import sys
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater


# https://github.com/GianisTsol/python-p2p/blob/master/pythonp2p/node.py

# Observacones:
#   - Para el traspaso de modelos, sea mejor crear un socket a parte
#   - Crear un enmascaramiento de sockets por si en algun futuro se quiere modificar

# REVISAR THREADS PAR QUE NO HAYA INANICIONES

# REVISAR QUE NO SE PUEDAN CONCATENAR MENSAJES EN EL BUFFER

# Se debe tener en cuenta que si salta una excepción en una ejecución remota se da por error -> por lo tanto procurar loguear la excepción


#REVISAR TESST Y EJECUCIONES DESDE CONSOLA


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

    def __init__(self, host="127.0.0.1", port=0, model=0):
        threading.Thread.__init__(self)
        self.terminate_flag = threading.Event()
        self.host = host
        self.port = port

        # Setting Up Node Socket (listening)
        self.node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP
        self.node_socket.bind((host, port))
        self.node_socket.listen(50)# no mas de 50 peticones a la cola
        if port==0:
            self.port = self.node_socket.getsockname()[1]
        
        # Neightboors
        self.neightboors = []

        # Logging
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        # Heartbeater
        self.heartbeater = Heartbeater(self)
        self.heartbeater.start()

        # Learning
        self.model = model
        self.round = None
        self.models = []
        self.models_lock = threading.Lock()


    #Objetivo: Agregar vecinos a la lista -> CREAR POSIBLES SOCKETS 
    def run(self):
        logging.info('Nodo a la escucha en {} {}'.format(self.host, self.port))
        while not self.terminate_flag.is_set(): 
            try:
                (node_socket, addr) = self.node_socket.accept()
                msg = node_socket.recv(BUFFER_SIZE)
                
                # Process new connection
                if msg:
                    msg = msg.decode("UTF-8")
                    callback = lambda h,p,b: self.__process_new_connection(node_socket, h, p, b)
                    if not CommunicationProtocol.process_connection(self,msg,callback):
                        logging.debug('Conexión rechazada con {}:{}'.format(addr,msg))
                        node_socket.close()         
                        
            except Exception as e:
                logging.exception(e)

        #Detenemos nodo
        logging.info('Bajando el nodo, dejando de escuchar en {} {}'.format(self.host, self.port))
        self.heartbeater.stop()
        for n in self.neightboors:
            n.stop()
        self.node_socket.close()


    def __process_new_connection(self, node_socket, h, p, broadcast):
        try:
            # Comprobamos que ip y puerto son correctos
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # REVISAR CUANDO SE LE PASA UN SERVICIO QUE NO RESPONDE X TEMAS DE QUE LOS TIMEUTS TARDAN MUCHO
            s.settimeout(2)
            result = s.connect_ex((h,p)) 
            s.close()
    
            # Agregaos el vecino
            if result == 0:
                logging.info('Conexión aceptada con {}'.format((h,p)))
                nc = NodeConnection(self,node_socket,(h,p))
                nc.start()
                self.add_neighbor(nc)
                 
                if broadcast:
                    self.broadcast((CommunicationProtocol.CONN_TO + " " + h + " " + str(p)).encode("utf-8"),exc=[nc])

        except Exception as e:
            logging.exception(e)
            self.rm_neighbor(nc)
            raise e
   

    def stop(self): 
        self.terminate_flag.set()
        # Enviamos mensaje al loop para evitar la espera del recv
        try:
            self.__send(self.host,self.port,b"")
        except:
            pass

    def get_addr(self):
        return self.host,self.port

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
                n.send(msg)

    ############################
    #         Learning         #
    ############################

    def start_learning(self): #aquí tendremos que pasar el modelo
        self.round = 0
        logging.info("Broadcasting model to all clients: " + str(self.model))
        msg = CommunicationProtocol.build_data_msg(self.model)
        self.broadcast(msg.encode("utf-8"))
    
    def stop_learning(self):
        self.round = None
        logging.info("Stopping learning")
        #DETENER PROC DE ENTRENAMIENTO EN FUTURO

    #
    # aquí tenemos problemas de sincronía
    #
    def add_model(self,m):

        #plantearse mecanismo para validar quien introduce modelo

        # Agregamos modelo
        self.models_lock.acquire()
        self.models.append(m)
        logging.info("Model added (" + str(len(self.models)) + "/" + str(len(self.neightboors)) + ")")

        #Revisamos si están todos
        if len(self.models)==len(self.neightboors):
            logging.info("Comenzando la promediación")
            self.round = self.round + 1

        
        self.models_lock.release()


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

        msg=(CommunicationProtocol.CONN + " " + str(self.host) + " " + str(self.port) + " " + full).encode("utf-8")
        s = self.__send(h,p,msg,persist=True)
        
        # Agregaos el vecino
        nc = NodeConnection(self,s,(h,p))
        nc.start()
        self.add_neighbor(nc)
      


    # CAMBIAR EL NOMBRE A ESTOS 2 MÉTODOS

    def set_start_learning(self, rounds=1): #falta x implementar rondas, de momento solo se coverge a 1 y listo
        if self.round is None:
            # Como es full conected, con 1 broadcast llega
            self.broadcast((CommunicationProtocol.START_LEARNING + " " + str(rounds)).encode("utf-8"))
            self.start_learning()
        else:
            print("Learning is Running")

    def set_stop_learning(self):
        if self.round is not None:
            self.broadcast((CommunicationProtocol.STOP_LEARNING).encode("utf-8"))
            self.stop_learning()
        else:
            print("Learning is not Running")