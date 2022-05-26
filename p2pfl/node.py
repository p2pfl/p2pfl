import socket
import threading
import logging
import sys
from p2pfl.agregator import FedAvg
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *
from p2pfl.learning.data import FederatedDM
from p2pfl.learning.learner import MyNodeLearning
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater
import time
#from p2pfl.learning.model import NodeLearning

# Observacones:
#   - Para el traspaso de modelos, sea mejor crear un socket a parte
#   - Crear un enmascaramiento de sockets por si en algun futuro se quiere modificar

############################################################################################
# FULL CONNECTED HAY QUE IMPLEMENTARLO DE FORMA QUE CUANDO SE INTRODUCE UN NODO EN LA RED, SE HACE UN BROADCAST
############################################################################################

# Tener cuidado con asyncronismos, tal como está en cuanto se agreguen los modelos mete el modelo tal cual está


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

    def __init__(self, host="127.0.0.1", port=0, model=None):
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
        log_dir = str(self.host) + "_" + str(self.port)
        self.learner = MyNodeLearning(FederatedDM(), log_dir, model=model) # De MOMENTO NO USAMOS LA DATA PERO HAY QUE PONERLO
        self.round = None
        self.totalrounds = None
        self.agredator = None #esto está bien aquí? -> no, a parte hay que instanciarlos x ronda


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
    def broadcast(self, msg, ttl=1, exc=[], is_model=False):
        for n in self.neightboors:
            if not (n in exc):
                n.send(msg, is_model)

    ############################
    #         Learning         #
    ############################

    #CREAR UN THREAD DE CÖMPUTO PARA TRAINING Y AGREGACIÓN DE MODELOS

    def start_learning(self,rounds): #aquí tendremos que pasar el modelo
        self.round = 0
        self.totalrounds = rounds
        self.agredator = FedAvg(self)
        self.__train_step()

    
    def stop_learning(self):
        self.round = None
        logging.info("Stopping learning")
        # DETENET THREAD DE CÖMPUTO


    def add_model(self,m):
        # Model on bytes
        #plantearse mecanismo para validar quien introduce modelo
        self.agredator.add_model(self.learner.decode_parameters(m))

    #-----------------------------------------------------
    # Implementar un observador en condiciones
    #-----------------------------------------------------
    def on_round_finished(self):
        # no hagrá que destruir el anteroir?
        self.agredator = FedAvg(self)
        self.round = self.round + 1
        logging.info("Round {} of {} finished. ({})".format(self.round,self.totalrounds,self.get_addr()))

        if self.round < self.totalrounds:
            # Wait local model sharing processes (to avoid model sharing conflicts)
            while True:
                print("Waiting for local model sharing processes to finish")
                if self.round_models_shared:
                    break
                time.sleep(0.1)

            # Next Step
            self.__train_step()            
        else:
            self.round = None
            logging.info("Finish!!.")




    ################
    # Trainig step # 
    ################

    def __train_step(self):
        self.round_models_shared = False
        self.__train()
        self.agredator.add_model(self.learner.get_parameters())
        self.__bc_model()
        
    def __train(self):
        logging.info("Training...")
        self.learner.fit()

    def __bc_model(self):
        logging.info("Broadcasting model to all clients...")
        encoded_msgs = CommunicationProtocol.build_data_msgs(self.learner.encode_parameters())
        # Lock Neightboors Communication
        self.set_sending_model(True)
        for msg in encoded_msgs:
            self.broadcast(msg,is_model=True)

        # UnLock Neightboors Communication
        self.set_sending_model(False)

        self.round_models_shared = True

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
      


    # CAMBIAR EL NOMBRE A ESTOS 2 MÉTODOS -> confuso

    def set_start_learning(self, rounds=1): #falta x implementar rondas, de momento solo se coverge a 1 y listo
        if self.round is None:
            # Como es full conected, con 1 broadcast llega
            logging.info("Broadcasting start learning...")
            self.broadcast((CommunicationProtocol.START_LEARNING + " " + str(rounds)).encode("utf-8"))
            self.start_learning(rounds)
        else:
            print("Learning is Running")

    def set_stop_learning(self):
        if self.round is not None:
            self.broadcast((CommunicationProtocol.STOP_LEARNING).encode("utf-8"))
            self.stop_learning()
        else:
            print("Learning is not Running")

    def set_sending_model(self, flag):
        for node in self.neightboors:
            node.set_sending_model(flag)