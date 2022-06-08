from concurrent.futures import thread
from distutils.log import debug
import socket
import threading
import logging
import sys
import time
from p2pfl.base_node import BaseNode
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

class Node(BaseNode, Observer):

    #####################
    #     Node Init     #
    #####################

    def __init__(self, model, data, host="127.0.0.1", port=0, agregator=FedAvg):
        BaseNode.__init__(self,host,port)
        Observer.__init__(self)

        # Learning
        log_dir = str(self.host) + "_" + str(self.port)
        self.learner = LightningLearner(model, data, log_name=log_dir) 
        self.round = None
        self.totalrounds = None
        self.agredator = agregator(self)
        self.agredator.add_observer(self)
        self.is_model_init = False
        self.__finish_wait_lock = threading.Lock()
        self.__finish_agregation_lock = threading.Lock()
        self.__finish_agregation_lock.acquire()
        
    #######################
    #   Node Management   #
    #######################
    
    def stop(self): 
        if self.round is not None:
            self.stop_learning()
        super().stop()

    ################
    #   Observer   #
    ################

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

    ####################################
    #         Learning Setters         #
    ####################################

    def set_model(self, model):
        self.learner.set_model(model)

    def set_data(self, data):
        self.learner.set_data(data)


    ###############################################
    #         Network Learning Management         #
    ###############################################

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


    ##################################
    #         Local Learning         #
    ##################################

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

    ####################################
    #         Model Agregation         #
    ####################################

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

    
    ################################
    #         Trainig step         #
    ################################

    def __train_step(self):
        # Check if Learning has been interrupted
        if self.round is not None:
            self.__train()
        
        if self.round is not None:
            self.agredator.add_model(str(self.get_addr()),self.learner.get_parameters(), self.learner.get_num_samples())
            self.__bc_model()

        if self.round is not None:
            self.__on_round_finished()
       
        
    def __train(self):
        logging.info("({}) Training...".format(self.get_addr()))
        self.learner.fit()

    def __bc_model(self):
        encoded_msgs = CommunicationProtocol.build_params_msg(self.learner.encode_parameters())
        logging.info("({}) Broadcasting model to {} clients. (size: {} bytes)".format(self.get_addr(),len(self.neightboors),len(encoded_msgs)*BUFFER_SIZE))

        # Lock Neightboors Communication
        #self.__set_sending_model(True)
        # Send Fragments
        for msg in encoded_msgs:
            self.broadcast(msg,is_model=True)
        # UnLock Neightboors Communication
        #self.__set_sending_model(False)

    def __on_round_finished(self):
        try:
            if self.round is not None:
                
                # Wait to finish self agregation
                self.__finish_agregation_lock.acquire()
                
                #print("-------------------------NO ESTA FUNCIONANDO, LA RONDA 2 NO LA ESPERA. o quiza computa una ronda de más, el asunto es que se agregan cosas despues del finish------------------------------")
                # Send ready message --> quizá ya no haga falta bloquear el socket
                while not self.broadcast(CommunicationProtocol.build_ready_msg(self.round)):
                    time.sleep(0.1)
                
                # Wait for ready messages
                while True:
                    logging.info("({}) Waiting other nodes.".format(self.get_addr()))
                    # If the trainning has been interrupted, stop waiting
                    if self.round is None:
                        logging.info("({}) Shutting down round finished process.".format(self.get_addr()))
                        return
                        
                    if all([ nc.get_ready_status()>=self.round for nc in self.neightboors]):
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
                logging.info("({}) FL not running but models received".format(self.get_addr()))
        except Exception as e:
            logging.error("({}) Concurrence Error: {}".format(self.get_addr(),e))


    def __set_sending_model(self, flag):
        for node in self.neightboors:
            node.set_sending_model(flag)