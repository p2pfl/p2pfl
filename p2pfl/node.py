import math
import random
import threading
import logging
import time
from p2pfl.base_node import BaseNode
from p2pfl.communication_protocol import CommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.learning.agregators.fedavg import FedAvg
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.pytorch.lightninglearner import LightningLearner
from p2pfl.utils.observer import Events, Observer



# FRACCIONES -> radom o por mecanismos de votación

# revisar test que falla: test_node_down_on_learning

# REVISAR LO DE BLOQUEAR CUANDO SE MANDA EL MODELO

# Cambiar algunos int nones por -1 para preservar el tipo

# Num samples -> meterlo en el handshaking -> puede traer problemas en topoligías no completamente conectadas

###################################################################################################################
# FULL CONNECTED HAY QUE IMPLEMENTARLO DE FORMA QUE CUANDO SE INTRODUCE UN NODO EN LA RED, SE HACE UN BROADCAST
###################################################################################################################

class Node(BaseNode, Observer):

    #####################
    #     Node Init     #
    #####################

    def __init__(self, model, data, host="127.0.0.1", port=0, learner=LightningLearner, agregator=FedAvg):
        """
        Class based on a base node that allows ***p2p Federated Learning**. 
            
        Args:
            model (torch.nn.Module): Model to be learned.
            data (torch.utils.data.Dataset): Dataset to be used in the learning process.
            host (str): Host where the node will be listening.
            port (int): Port where the node will be listening.
            agregator (Agregator): Agregator to be used in the learning process.

        Attributes:
            log_dir (str): Directory where the logs will be saved.
            learner (Learner): Learner to be used in the learning process.
            round (int): Round of the learning process.
            totalrounds (int): Total number of rounds of the learning process.
            agredator (Agregator): Agregator to be used in the learning process.
            is_model_init (bool): Flag to indicate if the model has been initialized.
        """
        BaseNode.__init__(self,host,port)
        Observer.__init__(self)

        # Learning
        log_dir = str(self.host) + "_" + str(self.port)
        self.learner = learner(model, data, log_name=log_dir) 
        self.round = None
        self.totalrounds = None
        self.train_set = []
        self.agregator = agregator( node_name = self.get_addr()[0] + ":" + str(self.get_addr()[1]) )
        self.agregator.add_observer(self)
        self.is_model_init = False

        # Locks
        self.__wait_models_ready_lock = threading.Lock()
        self.__wait_votes_ready_lock = threading.Lock()
        self.__finish_agregation_lock = threading.Lock()
        self.__finish_agregation_lock.acquire()
        
    #######################
    #   Node Management   #
    #######################
    
    def stop(self): 
        """
        Stop the node and the learning if it is running.
        """
        if self.round is not None:
            self.__stop_learning()
        self.learner.close()
        super().stop()

    ################
    #   Observer   #
    ################

    def update(self,event,obj):
        """
        Observer update method. Used to handle events that can occur in the agregator or neightboors.
        
        Args:
            event (Events): Event that has occurred.
            obj: Object that has been updated. 
        """
        if event == Events.END_CONNECTION:
            self.rm_neighbor(obj)
            self.agregator.remove_node_to_agregate()
            try:
                self.__wait_models_ready_lock.release()
                self.__wait_votes_ready_lock.release()
            except:
                pass

        elif event == Events.NODE_MODELS_READY_EVENT:
            # Try to unlock to check if all nodes are ready (on_finish_round (agregator_thread))
            try:
                self.__wait_models_ready_lock.release()
            except:
                pass

        elif event == Events.AGREGATION_FINISHED:
            if obj is not None:
                self.learner.set_parameters(obj)
            try:
                self.__finish_agregation_lock.release()
            except:
                pass

        elif event == Events.CONN_TO:
            self.connect_to(obj[0], obj[1], full=False)

        elif event == Events.START_LEARNING:
            self.__start_learning_thread(obj[0],obj[1])

        elif event == Events.STOP_LEARNING:
            self.__stop_learning()
    
        elif event == Events.PARAMS_RECEIVED:
            self.add_model(obj[0],obj[1],obj[2])
        
        elif event == Events.METRICS_RECEIVED:
            # Log Metrics
            name, round, loss, metric = obj
            self.learner.log_validation_metrics(loss,metric,round=round,name=name)

        elif event == Events.TRAIN_SET_VOTE_RECEIVED_EVENT:
            try:
                self.__wait_votes_ready_lock.release()
            except:
                pass
                
        else:
            logging.error("({}) Event not handled: {}".format(self.get_addr(),event))
    ####################################
    #         Learning Setters         #
    ####################################

    def set_model(self, model):
        """"
        Set the model to be learned (learner). 

        Carefully, model, not weights.

        Args:
            model: Model to be learned.
        """
        self.learner.set_model(model)

    def set_data(self, data):
        """
        Set the data to be used in the learning process (learner).

        Args:
            data: Dataset to be used in the learning process.
        """

        self.learner.set_data(data)


    ###############################################
    #         Network Learning Management         #
    ###############################################

    def set_start_learning(self, rounds=1, epochs=1): 
        """
        Start the learning process in the entire network.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        if self.round is None:
            # Start Learning
            logging.info("({}) Broadcasting start learning...".format(self.get_addr()))
            self.broadcast(CommunicationProtocol.build_start_learning_msg(rounds,epochs))
            # Initialize model
            logging.info("({}) Sending Initial Model Weights".format(self.get_addr()))
            self.is_model_init = True
            self.__bc_model()
            # Learning Thread
            self.__start_learning_thread(rounds,epochs)
        else:
            logging.debug("({}) Learning already started".format(self.get_addr()))

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.__stop_learning()
        else:
            logging.debug("({}) Learning already stopped".format(self.get_addr()))


    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self,rounds,epochs):
        learning_thread = threading.Thread(target=self.__start_learning,args=(rounds,epochs))
        learning_thread.name = "learning_thread-" + self.get_addr()[0] + ":" + str(self.get_addr()[1])
        learning_thread.daemon = True
        learning_thread.start()

    def __start_learning(self,rounds,epochs):
        """
        Start the learning process in the local node.
        
        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        self.round = 0
        self.totalrounds = rounds
        self.learner.init()

        # Indicates samples that be used in the learning process
        logging.info("({}) Broadcasting Number of Samples...".format(self.get_addr()))  # esto meterlo en el handsaking

        #esto ya no hará falta -> ahora multilee -> tb nos perjudica porque si ya se está mandado el modelo, va a promediarlo x 0
        self.broadcast(CommunicationProtocol.build_num_samples_msg(self.learner.get_num_samples())) # si no se manda bien promedia x 0
        
        #esto de aqui es una apaño de los malos -> cambiar por lock
        if not self.is_model_init:
            logging.info("({}) Initialicing Model Weights".format(self.get_addr()))
            while not self.is_model_init:
               time.sleep(0.1)

        # Train
        self.learner.set_epochs(epochs)
        self.__train_step()

    def __stop_learning(self): 
        """
        Stop the learning process in the local node. Interrupts learning process if its running.
        """
        logging.info("({}) Stopping learning".format(self.get_addr()))
        # Rounds
        self.round = None
        self.totalrounds = None
        # Leraner
        self.learner.interrupt_fit()
        # Agregator
        self.agregator.check_and_run_agregation(force=True)  
        self.agregator.set_nodes_to_agregate(None)
        self.agregator.clear()
        # Try to free wait locks
        try:
            self.__wait_models_ready_lock.release()
            self.__wait_votes_ready_lock.release()
        except:
            pass

    ####################################
    #         Model Agregation         #
    ####################################

    #-------------------------------------------------------
    # FUTURO -> validar quien introduce moedelos (llevar cuenta) |> (2 aprox)
    #-------------------------------------------------------
    
    
    def add_model(self,node,m,w): 
        """
        Add a model. The model isn't inicializated, the recieved model is used for it. Otherwise, the model is agregated using the **agregator**.

        Args:
            node (str): Node that has sent the model.
            m (Weights): Model to be added.
            w: Number of samples used to train the model.
        """
        # Check if Learning is running
        if self.round is not None:
            try:
                if self.is_model_init:
                    # Add model to agregator
                    decoded_model = self.learner.decode_parameters(m)
                    if self.learner.check_parameters(decoded_model):
                        self.agregator.add_model(node,decoded_model,w)
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    # Initialize model
                    self.is_model_init = True
                    logging.info("({}) Model initialized".format(self.get_addr()))
                    self.learner.set_parameters(self.learner.decode_parameters(m))
            
            except DecodingParamsError as e:
                # Bajamos el nodo
                logging.error("({}) Error decoding parameters".format(self.get_addr()))
                self.stop()

                # ----------------------- temporal -----------------------
                # ------------------ used to debug errors -------------------
                # append m in a file
                with open('paramserror.log','a') as f:
                    f.write(str(m))
                    f.write("\n\n\n")

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

        # Set train set
        if self.round is not None:
            train_set  = self.__vote_train_set() # este trainset es de strincgs no de node conections
        
            if train_set is not None:
                self.train_set = train_set

            print("{} Train set: {}".format(self.get_addr(),self.train_set))

            self.agregator.set_nodes_to_agregate(len(self.train_set)) ## en caso de que se caida un nodo se tiene que validar si es del trainset


        
        if self.get_addr() in self.train_set:
                
            # Evaluate and send metrics
            if self.round is not None:
                self.__bc_metrics(self.__evaluate())

            # Train
            if self.round is not None:
                self.__train()
            
            # Send Model
            if self.round is not None:
                self.agregator.add_model(str(self.get_addr()),self.learner.get_parameters(), self.learner.get_num_samples()[0])
                self.__bc_model()

        # Wait for model agregation
        if self.round is not None:
            self.__wait_model_agregation()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()
    
       
        
    def __train(self):
        logging.info("({}) Training...".format(self.get_addr()))
        self.learner.fit()

    def __evaluate(self):
        logging.info("({}) Evaluating...".format(self.get_addr()))
        retults = self.learner.evaluate()
        logging.info("({}) Evaluated. Losss: {}, Metric: {}. (Check tensorboard for more info)".format(self.get_addr(),retults[0],retults[1]))
        return retults



    def __bc_model(self):
        encoded_msgs = CommunicationProtocol.build_params_msg(self.learner.encode_parameters())
        logging.info("({}) Broadcasting model to {} clients. (size: {} bytes)".format(self.get_addr(),len(self.neightboors),len(encoded_msgs)*Settings.BUFFER_SIZE))

        # Lock Neightboors Communication
        self.__set_sending_model(True)
        # Send Fragments
        for msg in encoded_msgs:
            self.broadcast(msg)
        # UnLock Neightboors Communication
        self.__set_sending_model(False)


    # SI ESTAMOS EN SIMULACION NO PERMITIR BROADCASTEAERLAS -> tonter'ia
    def __bc_metrics(self,metrics): 
        logging.info("({}) Broadcasting metrics to {} clients.".format(self.get_addr(),len(self.neightboors)))
        encoded_msgs = CommunicationProtocol.build_metrics_msg(self.round,metrics[0],metrics[1])
        self.broadcast(encoded_msgs)

    def __set_sending_model(self, flag):
        for node in self.neightboors:
            node.set_sending_model(flag)

    # Returns participants in the training round
    def __vote_train_set(self):

        # Vote
        if self.neightboors != []:
            # Send vote
            candidates = random.choices(self.neightboors, k=3)
            candidates = [candidate.get_addr() for candidate in candidates]
            weights = [random.randint(0,1000),math.floor(random.randint(0,1000)/2),math.floor(random.randint(0,1000)/4)]
            votes = list(zip(candidates,weights))
            self.broadcast(CommunicationProtocol.build_vote_train_set_msg(votes))
                    
            # Wait for other votes
            logging.info("({}) Waiting other node votes.".format(self.get_addr()))
            while True:
                # If the trainning has been interrupted, stop waiting
                if self.round is None:
                    logging.info("({}) Stopping on_round_finished process.".format(self.get_addr()))
                    return None
                            
                if all([ nc.get_train_set_votes()!=[] for nc in self.neightboors]):

                    # Y EN CASO DE EMPATE?

                    results = dict(votes)


                    for nc in self.neightboors:
                        
                        for i in range(len(nc.get_train_set_votes())):
                            k = list(nc.get_train_set_votes().keys())[i]
                            v = list(nc.get_train_set_votes().values())[i]

                            if k in results:
                                results[k] += v
                            else:
                                results[k] = v


                    # Order by votes and get TOP X
                    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                    top = min(len(results), Settings.TRAIN_SET_SIZE)
                    results = results[0:top]
                    results = {k: v for k, v in results}

                    # Clear votes    
                    for n in self.neightboors:
                        n.clear_train_set_votes()

                    return results 
                self.__wait_votes_ready_lock.acquire()
            
                                
    def __wait_model_agregation(self):
        try:
            # Wait to finish self agregation
            self.__finish_agregation_lock.acquire()
                
            # Send ready message --> quizá ya no haga falta bloquear el socket
            self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
                
            # Wait for ready messages
            logging.info("({}) Waiting other nodes.".format(self.get_addr()))
            while True:
                # If the trainning has been interrupted, stop waiting
                if self.round is None:
                    logging.info("({}) Stopping on_round_finished process.".format(self.get_addr()))
                    return
                        
                if all([ nc.get_ready_model_status()>=self.round for nc in self.neightboors]):
                    break
                self.__wait_models_ready_lock.acquire()
                               
        except Exception as e:
            logging.error("({}) Concurrence Error: {}".format(self.get_addr(),e))

    def __on_round_finished(self):
        # Set Next Round
        self.round = self.round + 1
        logging.info("({}) Round {} of {} finished.".format(self.get_addr(),self.round,self.totalrounds))

        # Next Step or Finish
        if self.round < self.totalrounds:
            self.__train_step()  
        else:
            # Calculate final metrics before finishing
            self.__bc_metrics(self.__evaluate())
            # Finish
            self.round = None
            self.is_model_init = False
            self.agregator.set_nodes_to_agregate(None)
            logging.info("({}) Finish!!.".format(self.get_addr()))   