import socket
import threading
import logging
import sys
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *
from p2pfl.learning.agregators.fedavg import FedAvg
from p2pfl.learning.pytorch.datamodules.mnist import MnistFederatedDM
from p2pfl.learning.pytorch.learners.lightninglearner import LightningLearner
from p2pfl.learning.pytorch.models.mlp import MLP
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater
import time

from p2pfl.utils.observer import Observer

# FRACCIONES -> radom o por mecanismos de votación

###################################################################################################################
# FULL CONNECTED HAY QUE IMPLEMENTARLO DE FORMA QUE CUANDO SE INTRODUCE UN NODO EN LA RED, SE HACE UN BROADCAST
###################################################################################################################

class Node(threading.Thread, Observer):

    #####################
    #     Node Init     #
    #####################

    def __init__(self, model, data, host="127.0.0.1", port=0, agregator=FedAvg):
        threading.Thread.__init__(self)
        self.terminate_flag = threading.Event()
        self.host = host
        self.port = port

        # Setting Up Node Socket (listening)
        self.node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP Socket
        self.node_socket.bind((host, port))
        self.node_socket.listen(50) # no more than 50 connections at queue
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
        self.learner = LightningLearner(model, data, log_name=log_dir) 
        self.round = None
        self.totalrounds = None
        self.agredator = agregator(self)


    def get_addr(self):
        return self.host,self.port


    ########################
    #   Main Thread Loop   #
    ########################
    
    # Start the main loop in a new thread
    def start(self):
        super().start()

    
    def stop(self): 
        self.terminate_flag.set()
        if self.round is not None:
            self.stop_learning()
        # Enviamos mensaje al loop para evitar la espera del recv
        try:
            self.__send(self.host,self.port,b"")
        except:
            pass


    # Main Thread of node.
    #   Its listening for new nodes to be added
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
                    if not CommunicationProtocol.process_connection(msg,callback):
                        logging.debug('Conexión rechazada con {}:{}'.format(addr,msg))
                        node_socket.close()         
                        
            except Exception as e:
                logging.exception(e)

        #Stop Node
        logging.info('Bajando el nodo, dejando de escuchar en {} {}'.format(self.host, self.port))
        self.heartbeater.stop()
        for n in self.neightboors:
            n.stop()
        self.node_socket.close()


    def __process_new_connection(self, node_socket, h, p, broadcast):
        try:
            # Check if ip and port are correct
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            s.settimeout(2)
            result = s.connect_ex((h,p)) 
            s.close()
    
            # Add neightboor
            if result == 0:
                logging.info('Conexión aceptada con {}'.format((h,p)))
                nc = NodeConnection(self,node_socket,(h,p))
                nc.add_observer(self)
                nc.start()
                self.add_neighbor(nc)
                 
                if broadcast:
                    self.broadcast(CommunicationProtocol.build_connect_to_msg(h,p),exc=[nc])

        except Exception as e:
            logging.exception(e)
            self.rm_neighbor(nc)
            raise e
   

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

    # Observer Pattern used to notify end of connections
    def update(self,nc):
        self.rm_neighbor(nc)
  
    # Connecto to a node
    #   - If full -> the node will be connected to the entire network
    def connect_to(self, h, p, full=True):
        if full:
            full = "1"
        else:
            full = "0"

        # Send connection request
        msg=CommunicationProtocol.build_connect_msg(self.host,self.port,full)
        s = self.__send(h,p,msg,persist=True)
        
        # Add socket to neightboors
        nc = NodeConnection(self,s,(h,p))
        nc.add_observer(self)
        nc.start()
        self.add_neighbor(nc)
      
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

    # FULL_CONNECTED -> 3th iteration |> TTL 
    def broadcast(self, msg, ttl=1, exc=[], is_model=False):
        sended=True 
        for n in self.neightboors:
            if not (n in exc):
                sended = sended and n.send(msg, is_model)
        return sended

    ############################
    #         Learning         #
    ############################

    # Start the network learning
    def set_start_learning(self, rounds=1, epochs=1): 
        # 
        # Maybe needs a lock to avoid concurrency problems
        #
        if self.round is None:
            logging.info("Broadcasting start learning...")
            self.broadcast(CommunicationProtocol.build_start_learning_msg(rounds,epochs))
            # Learning Thread
            learning_thread = threading.Thread(target=self.start_learning,args=(rounds,epochs))
            learning_thread.start()
        else:
            print("Learning is not Running")

    # Stop the network learning
    def set_stop_learning(self):
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.stop_learning()
        else:
            print("Learning is not Running")

    # Start the local learning
    def start_learning(self,rounds,epochs): #local
        self.round = 0
        self.totalrounds = rounds
        # Indicates samples that be used in the learning process
        if not self.broadcast(CommunicationProtocol.build_num_samples_msg(self.learner.get_num_samples())):
            logging.error("No se han podido enviar los números de muestras a todos los nodos")
            self.set_stop_learning()
        # Train
        self.learner.set_epochs(epochs)
        self.__train_step()

    #-------------------------------------------------------
    # REVISAR EN PROFUNDIDAD -> cuando aun no se inició el proc de learning (trainer)
    #-------------------------------------------------------
    # Stop the local learning
    def stop_learning(self): #local
        logging.info("Stopping learning")
        self.learner.interrupt_fit()
        self.round = None
        self.totalrounds = None
        self.agredator.clear()

    #-------------------------------------------------------
    # FUTURO -> validar quien introduce moedelos (llevar cuenta) |> (2 aprox)
    #-------------------------------------------------------
    def add_model(self,m,w): 
        # Check if Learning is running
        if self.round is not None:
            self.agredator.add_model(self.learner.decode_parameters(m),w) 

    def on_round_finished(self):
        if self.round is not None:
            self.round = self.round + 1
            logging.info("Round {} of {} finished. ({})".format(self.round,self.totalrounds,self.get_addr()))

            # Determine if learning is finished
            if self.round < self.totalrounds:
                # Wait local model sharing processes (to avoid model sharing conflicts)
                while True:
                    #print("Waiting for local model sharing processes to finish")
                    if not self.__is_sending_model():
                        break
                    time.sleep(0.1)

                # Next Step
                self.__train_step()            
            else:
                self.round = None
                logging.info("Finish!!.")
        else:
            logging.info("FL not running but models received")


    ################
    # Trainig step # 
    ################

    def __train_step(self):
        # Check if Learning has been interrupted
        if self.round is not None:
            self.__train()
        
        if self.round is not None:
            self.agredator.add_model(self.learner.get_parameters(), self.learner.get_num_samples())
            self.__bc_model()
       
        
    def __train(self):
        logging.info("Training...")
        self.learner.fit()

    def __bc_model(self):
        logging.info("Broadcasting model to all clients...")
        encoded_msgs = CommunicationProtocol.build_params_msg(self.learner.encode_parameters())
        # Lock Neightboors Communication
        self.__set_sending_model(True)
        # Send Fragments
        for msg in encoded_msgs:
            self.broadcast(msg,is_model=True)
        # UnLock Neightboors Communication
        self.__set_sending_model(False)

    def __set_sending_model(self, flag):
        for node in self.neightboors:
            node.set_sending_model(flag)

    def __is_sending_model(self):
        for node in self.neightboors:
            if node.is_sending_model():
                return True
        return False
