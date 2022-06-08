from concurrent.futures import thread
from distutils.log import debug
import socket
import threading
import logging
import sys
import time
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.const import *
from p2pfl.learning.agregators.fedavg import FedAvg
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.pytorch.learners.lightninglearner import LightningLearner
from p2pfl.node_connection import NodeConnection
from p2pfl.heartbeater import Heartbeater
from p2pfl.utils.observer import Events, Observer



# FRACCIONES -> radom o por mecanismos de votación

# revisar test que falla: test_node_down_on_learning

# REVISAR LO DE BLOQUEAR CUANDO SE MANDA EL MODELO

# Cambiar algunos int nones por -1 para preservar el tipo

# Models Added es resquicio, no se usa -> visualizar info

# para dejar bonito el código, podría separar el nodo en una clase genéricas (nodo) y luego en nodo para fl

#s separar en node y nodefl

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
        self.name = "node-" + self.get_addr()[0] + ":" + str(self.get_addr()[1])
        
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
        self.models_added = None
        self.agredator = agregator(self)
        self.agredator.add_observer(self)
        self.is_model_init = False
        self.__finish_wait_lock = threading.Lock()
        self.__finish_agregation_lock = threading.Lock()
        self.__finish_agregation_lock.acquire()
        
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
                        logging.debug('({}) Conexión rechazada con {}:{}'.format(self.get_addr(),addr,msg))
                        node_socket.close()         
                        
            except Exception as e:
                logging.exception(e)

        #Stop Node
        logging.info('Bajando el nodo, dejando de escuchar en {} {} y desconectándose de {} nodos'.format(self.host, self.port, len(self.neightboors))) 
        # Al realizar esta copia evitamor errores de concurrencia al recorrer la misma, puesto que sabemos que se van a eliminar los nodos de la misma
        nei_copy_list = self.neightboors.copy()
        for n in nei_copy_list:
            n.stop()
        self.heartbeater.stop()
        self.node_socket.close()


    def __process_new_connection(self, node_socket, h, p, broadcast):
        try:
            # Check if connection with the node already exist
            if self.get_neighbor(h,p) == None:

                # Check if ip and port are correct
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
                s.settimeout(2)
                result = s.connect_ex((h,p)) 
                s.close()
        
                # Add neightboor
                if result == 0:
                    logging.info('({}) Conexión aceptada con {}'.format(self.get_addr(),(h,p)))
                    nc = NodeConnection(self,node_socket,(h,p))
                    nc.add_observer(self)
                    nc.start()
                    self.add_neighbor(nc)
                    
                    if broadcast:
                        self.broadcast(CommunicationProtocol.build_connect_to_msg(h,p),exc=[nc])
            else:
                node_socket.close()

        except Exception as e:
            logging.exception(e)
            node_socket.close()
            self.rm_neighbor(nc)
            raise e
   

    #############################
    #  Neighborhood management  #
    #############################

    def get_neighbor(self, h, p):
        for n in self.neightboors:
            #print(str(n.get_addr()) + str((h,p)))
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
    def update(self,event,obj):
        if event == Events.END_CONNECTION:
            self.rm_neighbor(obj)
            self.agredator.check_and_run_agregation()
        elif event == Events.NODE_READY_EVENT:
            # Try to unlock to check if all nodes are ready (on_finish_round (agregator_thread))
            try:
                self.__finish_wait_lock.release()
            except:
                pass
        elif event == Events.AGREGATION_FINISHED:
            try:
                self.__finish_agregation_lock.release()
            except:
                pass

    # Connecto to a node
    #   - If full -> the node will be connected to the entire network
    def connect_to(self, h, p, full=True):
        if full:
            full = "1"
        else:
            full = "0"
            
        # Check if connection with the node already exist
        if self.get_neighbor(h,p) == None:

            # Send connection request
            msg=CommunicationProtocol.build_connect_msg(self.host,self.port,full)
            s = self.__send(h,p,msg,persist=True)
            
            # Add socket to neightboors
            nc = NodeConnection(self,s,(h,p))
            nc.add_observer(self)
            nc.start()
            self.add_neighbor(nc)
        
        else:
            logging.info('El nodo ya se encuentra conectado con {}:{}'.format(h,p))

    def disconnect_from(self, h, p):
        self.get_neighbor(h,p).stop()
      
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

    def set_model(self, model):
        self.learner.set_model(model)

    def set_data(self, data):
        self.learner.set_data(data)

    # Start the network learning
    #
    # Asegurarse de que se envien mensajes que el broadcast no se asegura
    #
    def set_start_learning(self, rounds=1, epochs=1): 
        # 
        # Maybe needs a lock to avoid concurrency problems
        #
        if self.round is None:
            # Start Learning
            logging.info("({}) Broadcasting start learning...".format(self.get_addr()))
            self.broadcast(CommunicationProtocol.build_start_learning_msg(rounds,epochs))
            # Initialize model
            logging.info("({}) Sending Initial Model Weights".format(self.get_addr()))
            self.is_model_init = True
            self.__bc_model()
            # Learning Thread
            learning_thread = threading.Thread(target=self.start_learning,args=(rounds,epochs))
            learning_thread.name = "learning_thread-" + self.get_addr()[0] + ":" + str(self.get_addr()[1])
            learning_thread.start()
        else:
            logging.debug("({}) Learning already started".format(self.get_addr()))


    # Stop the network learning
    def set_stop_learning(self):
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.stop_learning()
        else:
            logging.debug("({}) Learning already stopped".format(self.get_addr()))

    # Start the local learning
    def start_learning(self,rounds,epochs): #local
        self.round = 0
        self.totalrounds = rounds

        #esto de aqui es una apaño de los malos
        if not self.is_model_init:
            logging.info("({}) Initialicing Model Weights".format(self.get_addr()))
            while not self.is_model_init:
               time.sleep(0.1)

        # Indicates samples that be used in the learning process
        logging.info("({}) Broadcasting Number of Samples...".format(self.get_addr()))
        #esto ya no hará falta -> ahora multilee -> tb nos perjudica porque si ya se está mandado el modelo, va a promediarlo x 0
        if not self.broadcast(CommunicationProtocol.build_num_samples_msg(self.learner.get_num_samples())):
            logging.error("({}) No se han podido enviar los números de muestras a todos los nodos".format(self.get_addr()))
            self.set_stop_learning()
        
        # Train
        self.learner.set_epochs(epochs)
        self.__train_step()

    #-------------------------------------------------------
    # REVISAR EN PROFUNDIDAD -> cuando aun no se inició el proc de learning (trainer)
    #-------------------------------------------------------
    # Stop the local learning
    def stop_learning(self): #local
        logging.info("({}) Stopping learning".format(self.get_addr()))
        self.learner.interrupt_fit()
        self.round = None
        self.totalrounds = None
        self.agredator.clear()

    #-------------------------------------------------------
    # FUTURO -> validar quien introduce moedelos (llevar cuenta) |> (2 aprox)
    #-------------------------------------------------------
    #
    # POR ACABAR, CONTROLADO ERROR PERO NO SE HACE NADA
    #
    # NO ES LO MISMO UNA EXCEPCION DE PICKE QUE UNA DE PYTORCH
    #
    # Traza nodo que generó modelo
    #
    #-------------------------------------------------------
    # DEJARLO AQUI O METERLO EN EL COMANDO? -> dejar código bonito luego
    def add_model(self,node,m,w): 
        #print("({}) Adding model from {}".format(self.get_addr(),node))
        # Check if Learning is running
        if self.round is not None:
            try:
                if self.is_model_init:
                    # Add model to agregator
                    self.agredator.add_model(node,self.learner.decode_parameters(m),w) 
                else:
                    # Initialize model
                    self.is_model_init = True
                    logging.info("({}) Model initialized".format(self.get_addr()))
                    self.learner.set_parameters(self.learner.decode_parameters(m))
            
            except DecodingParamsError as e:
                # Bajamos el nodo
                logging.error("({}) Error decoding parameters".format(self.get_addr()))
                # temporal
                # append m in a file
                with open('paramserror.log','a') as f:
                    f.write(str(m))
                    f.write("\n\n\n")
                self.stop()

            except ModelNotMatchingError as e:
                # Bajamos el nodo
                logging.error("({}) Models not matching.".format(self.get_addr()))
                self.stop()
                    
            except Exception as e:
                # Bajamos el nodo
                self.stop()
                raise(e)
        else: 
            logging.error("({}) Tried to add a model while learning is not running".format(self.get_addr()))

    
    # on finish tb podríamos eliminarlo por puros eventos -> 
    def on_round_finished(self):
        try:
            if self.round is not None:
                self.models_added = 0 # hardcoded for now -----------------------------------------------------------------------------------------------------------
                # Determine if learning is finished
                print("{} < {}".format(self.round,self.totalrounds))
                if self.round < self.totalrounds:  # comportamientos de sincornización harán falta para todas las rondas (hasta la última)
                    self.__finish_agregation_lock.acquire()
                    print("-------------------------NO ESTA FUNCIONANDO, LA RONDA 2 NO LA ESPERA. o quiza computa una ronda de más, el asunto es que se agregan cosas despues del finish------------------------------")
                    # Send ready message --> quizá ya no haga falta bloquear el socket
                    while not self.broadcast(CommunicationProtocol.build_ready_msg(self.round,self.models_added)):
                        time.sleep(0.1)
                    
                    # Wait for ready messages
                    while True:
                        logging.info("({}) Waiting other nodes.".format(self.get_addr()))
                        # If the trainning has been interrupted, stop waiting
                        if self.round is None:
                            logging.info("({}) Shutting down round finished process.".format(self.get_addr()))
                            sys.exit()
                        
                        if all([ nc.get_ready_status()[0]>=self.round for nc in self.neightboors]):
                            break
                        self.__finish_wait_lock.acquire(timeout=2)
                        
                    # Set Next Round
                    self.round = self.round + 1
                    logging.info("({}) Round {} of {} finished.".format(self.get_addr(),self.round,self.totalrounds))

                    # Next Step
                    if self.round < self.totalrounds:
                        self.__train_step()  
                    else:
                        self.round = None
                        self.is_model_init = False
                        logging.info("({}) Finish!!.".format(self.get_addr()))          
                else:
                    ### BORRARLO, RESQUICIO
                    self.round = None
                    self.is_model_init = False
                    logging.info("({}) Finish!!.".format(self.get_addr()))
            else:
                logging.info("({}) FL not running but models received".format(self.get_addr()))
        except Exception as e:
            logging.error("({}) Concurrence Error: {}".format(self.get_addr(),e))

    ################
    # Trainig step # 
    ################

    def __train_step(self):
        # Check if Learning has been interrupted
        if self.round is not None:
            self.__train()
        
        if self.round is not None:
            self.agredator.add_model(str(self.get_addr()),self.learner.get_parameters(), self.learner.get_num_samples())
            self.__bc_model()

        if self.round is not None:
            self.on_round_finished()
       
        
    def __train(self):
        logging.info("({}) Training...".format(self.get_addr()))
        self.learner.fit()

    def __bc_model(self):
        encoded_msgs = CommunicationProtocol.build_params_msg(self.learner.encode_parameters())
        logging.info("({}) Broadcasting model to {} clients. (size: {} bytes)".format(self.get_addr(),len(self.neightboors),len(encoded_msgs)*BUFFER_SIZE))

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
