from distutils.log import debug
import math
import random
import threading
import logging
import time
from p2pfl.base_node import BaseNode
from p2pfl.command import *
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

# concurrencia .copy() vs locks -> tratar de borrar los copys (sobre todo en nei)

###################################################################################################################
# FULL CONNECTED HAY QUE IMPLEMENTARLO DE FORMA QUE CUANDO SE INTRODUCE UN NODO EN LA RED, SE HACE UN BROADCAST
###################################################################################################################

class Node(BaseNode):

    #####################
    #     Node Init     #
    #####################

    def __init__(self, model, data, host="127.0.0.1", port=None, learner=LightningLearner, agregator=FedAvg, simulation=True):
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
        BaseNode.__init__(self,host,port, simulation)
        Observer.__init__(self)

        # Learning
        log_dir = str(self.host) + "_" + str(self.port)
        self.learner = learner(model, data, log_name=log_dir) 
        self.round = None
        self.__start_thread_lock = threading.Lock()
        self.totalrounds = None
        self.train_set = []
        self.train_set_lock = threading.Lock()
        self.agregator = agregator( node_name = self.get_name() )
        self.agregator.add_observer(self)
        self.model_initialized = False

        # Locks
        self.__wait_votes_ready_lock = threading.Lock()
        self.__finish_agregation_lock = threading.Lock()
        self.__finish_agregation_lock.acquire()
        self.__wait_init_model_lock = threading.Lock()
        self.__wait_init_model_lock.acquire()

        
    #######################
    #   Node Management   #
    #######################

    def connect_to(self, h, p, full=True):
        nc = None
        if self.round is None:
            nc = super().connect_to(h, p, full)
            if nc is not None:
                # Send number of samples
                nc.send(CommunicationProtocol.build_num_samples_msg(self.learner.get_num_samples()))
        else:
            logging.info("({}) Cant connect to other nodes when learning is running. (however, other nodes can be connected to the node.)".format(self.get_name()))
        return nc

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
        # Execute BaseNode update
        super().update(event,obj)


        # For non directly connected nodes
        if event == Events.NODE_DISCONNECTED:
            print("POR HACER!!!!")

        # For directly connected nodes
        if event == Events.END_CONNECTION:
            # If a training process is running, comunicate the disconnection
            if self.round is not None:
                # Try to remove from trainset
                try:
                    self.train_set_lock.acquire()
                    node = obj.get_name()
                    self.train_set.remove(node)
                    self.train_set_lock.release()
                    # It cant produce training, if aggregation is running, clients only decrement
                    self.agregator.check_and_run_agregation()
                    
                except:
                    self.train_set_lock.release()

                # Refresh vote process waiters            
                try:
                    self.__wait_votes_ready_lock.release()
                except:
                    pass
                
        elif event == Events.NODE_CONNECTED_EVENT:
            # Send number of samples
            obj.send(CommunicationProtocol.build_num_samples_msg(self.learner.get_num_samples())) #----ojo tb tiene que hacerlo el que se conecta
            # Comunicate to the new node that a training process is running
            if self.round is not None:
                print("TO IMPLEMET WHEN THE TOPOLOGY WAS NOT FULLY CONNECTED")
                obj.stop()
                obj.send(CommunicationProtocol.build_learning_is_running_msg(self.round, self.totalrounds))

        elif event == Events.AGREGATION_FINISHED:
            # Set parameters and communate it to the training process
            if obj is not None:
                self.learner.set_parameters(obj)
                # Share that agregation is done
                self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))

            try:
                self.__finish_agregation_lock.release()
            except:
                pass

        elif event == Events.START_LEARNING:
            self.__start_learning_thread(obj[0],obj[1])

        elif event == Events.STOP_LEARNING:
            self.__stop_learning()
    
        elif event == Events.PARAMS_RECEIVED:
            self.add_model(obj)
        
        elif event == Events.METRICS_RECEIVED:
            # Log Metrics
            name, round, loss, metric = obj
            self.learner.log_validation_metrics(loss,metric,round=round,name=name)

        elif event == Events.TRAIN_SET_VOTE_RECEIVED_EVENT:
            # Communicate to the training process that a vote has been received
            try:
                self.__wait_votes_ready_lock.release()
            except:
                pass

        elif event == Events.LEARNING_IS_RUNNING_EVENT:
            print("NOT IMPLEMETED YET",obj)

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
            logging.info("({}) Broadcasting start learning...".format(self.get_name()))
            self.broadcast(CommunicationProtocol.build_start_learning_msg(rounds,epochs))
            # Initialize model
            self.broadcast(CommunicationProtocol.build_model_initialized_msg())
            self.__wait_init_model_lock.release()
            self.model_initialized = True # esto seguramente sobre, con locks es suficiente
            # Learning Thread
            self.__start_learning_thread(rounds,epochs)
        else:
            logging.debug("({}) Learning already started".format(self.get_name()))

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.__stop_learning()
        else:
            logging.debug("({}) Learning already stopped".format(self.get_name()))


    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self,rounds,epochs):
        
        if self.round is None:
            learning_thread = threading.Thread(target=self.__start_learning,args=(rounds,epochs))
            learning_thread.name = "learning_thread-" + self.get_name()
            learning_thread.daemon = True
            learning_thread.start()

    def __start_learning(self,rounds,epochs):
        """
        Start the learning process in the local node.
        
        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        self.__start_thread_lock.acquire() # Used to avoid create duplicated training threads
        if self.round is None:
            self.round = 0
            self.totalrounds = rounds
            self.learner.init()
            self.__start_thread_lock.release()

            # Gossiping model inicialization            
            self.__gossip_model_difusion(initialization=True)

            # Updates the number of samples by node
            self.broadcast(CommunicationProtocol.build_num_samples_msg(self.learner.get_num_samples())) # si no se manda bien promedia x 0

            # Train
            self.learner.set_epochs(epochs)
            self.__train_step()

    def __stop_learning(self): 
        """
        Stop the learning process in the local node. Interrupts learning process if its running.
        """
        logging.info("({}) Stopping learning".format(self.get_name()))
        # Rounds
        self.round = None
        self.totalrounds = None
        # Leraner
        self.learner.interrupt_fit()
        # Agregator
        self.agregator.check_and_run_agregation(force=True)  
        self.agregator.set_nodes_to_agregate([])
        self.agregator.clear()
        # Try to free wait locks
        try:
            self.__wait_votes_ready_lock.release()
        except:
            pass

    ####################################
    #         Model Agregation         #
    ####################################

    #-------------------------------------------------------
    # FUTURO -> validar quien introduce moedelos (llevar cuenta) |> (2 aprox)
    #-------------------------------------------------------
    
    
    def add_model(self,m): 
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
                if self.model_initialized:
                    # Add model to agregator
                    decoded_model, contributors = self.learner.decode_parameters(m)
                    
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.agregator.add_model(decoded_model,contributors)
                        if models_added is not None:
                            self.broadcast(CommunicationProtocol.build_models_agregated_msg(models_added))
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    # Initialize model
                    self.model_initialized = True
                    logging.info("({}) Initialicing Model Weights".format(self.get_name()))
                    self.__wait_init_model_lock.release()    
                    self.broadcast(CommunicationProtocol.build_model_initialized_msg())
                    model, _ = self.learner.decode_parameters(m)
                    self.learner.set_parameters(model)
            
            except DecodingParamsError as e:
                # Bajamos el nodo
                logging.error("({}) Error decoding parameters".format(self.get_name()))
                self.stop()

                # ----------------------- temporal -----------------------
                # ------------------ used to debug errors -------------------
                # append m in a file
                with open('paramserror.log','a') as f:
                    f.write(str(m))
                    f.write("\n\n\n")

            except ModelNotMatchingError as e:
                # Bajamos el nodo
                logging.error("({}) Models not matching.".format(self.get_name()))
                self.stop()
                    
            except Exception as e:
                # Bajamos el nodo
                self.stop()
                raise(e)
        else: 
            logging.error("({}) Tried to add a model while learning is not running".format(self.get_name()))

    
    ################################
    #         Trainig step         #
    ################################

    def __train_step(self):

        # Set train set
        if self.round is not None:
            self.train_set = self.__vote_train_set() # stringified to avoid comparing pointers
            self.__validate_train_set()
        
        # Train if the node was selected or if no exist candidates (node non-connected) 
        is_train_set = self.get_name() in self.train_set
        if is_train_set:

            # Set Models To Agregate and their weights
            self.agregator.set_nodes_to_agregate(self.train_set) # reference to self node stringified list of nodes
            node_w = [(nc.get_name(), nc.get_num_samples()[0]) for nc in self.neightboors if nc.get_name() in self.train_set]
            node_w.append((self.get_name(),self.learner.get_num_samples()[0]))
            self.agregator.set_node_weights(dict(node_w))
            
            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()
            
            # Agregate Model
            if self.round is not None:
                self.agregator.add_model(self.learner.get_parameters(),[self.get_name()])
                self.broadcast(CommunicationProtocol.build_models_agregated_msg([self.get_name()]))
                self.__gossip_model_agregation() # this is going to produce duplicated models -> buut its fault tolerent

        else: 
            # Set Models To Agregate 
            self.agregator.set_waiting_agregated_model()
            self.model_initialized = True # no hace falta inicializar modelo, no se usará

        # Gossip agregated model (also syncrhonizes nodes)
        if self.round is not None:
            self.__gossip_model_difusion()

        # DIFUNDIR MODELO DESPUES DE AGREGARLO -> ÚNICAMENTE SE DEBE DE AGREGAR CON EL TRAINSET
        # EL RESTO ÚNICAMENTE ESPERA 1 MODELO

        # Finish round
        if self.round is not None:
            self.__on_round_finished()

        
    def __train(self):
        logging.info("({}) Training...".format(self.get_name()))
        self.learner.fit()

    def __evaluate(self):
        logging.info("({}) Evaluating...".format(self.get_name()))
        results = self.learner.evaluate()
        if results is not None:
            logging.info("({}) Evaluated. Losss: {}, Metric: {}. (Check tensorboard for more info)".format(self.get_name(),results[0],results[1]))
            # Send metrics
            if not self.simulation:
                self.__bc_metrics(results)


    # SI ESTAMOS EN SIMULACION NO PERMITIR BROADCASTEAERLAS -> tonter'ia
    def __bc_metrics(self,metrics): 
        logging.info("({}) Broadcasting metrics to {} clients.".format(self.get_name(),len(self.neightboors)))
        encoded_msgs = CommunicationProtocol.build_metrics_msg(self.round,metrics[0],metrics[1])
        self.broadcast(encoded_msgs)

    # Returns participants in the training round. (stringified list of nodes)
    def __vote_train_set(self):

        # Vote
        candidates = [candidate.get_name() for candidate in self.neightboors.copy()] # to avoid concurrent modification
        initial_candidates_len = len(candidates)
        if candidates != []:
            # Send vote
            logging.info("({}) Sending train set vote.".format(self.get_name()))
            samples = min(Settings.TRAIN_SET_SIZE,len(candidates))
            candidates = random.sample(candidates, samples)
            weights = [random.randint(0,1000),math.floor(random.randint(0,1000)/2),math.floor(random.randint(0,1000)/4)]
            votes = list(zip(candidates,weights))

            logging.info("({}) Self Vote: {}".format(self.get_name(),votes))

            self.broadcast(CommunicationProtocol.build_vote_train_set_msg(votes))
                    
            # Wait for other votes
            logging.info("({}) Waiting other node votes.".format(self.get_name()))

            # Get time
            count = 0
            begin = time.time()                

            while True:
    
                # If the trainning has been interrupted, stop waiting
                if self.round is None:
                    logging.info("({}) Stopping on_round_finished process.".format(self.get_name()))
                    return []

                # Update time counters (timeout)
                count = count + (time.time() - begin)
                timeout = count > Settings.TIMEOUT_WAIT_VOTE

                if all([ nc.get_train_set_votes()!=[] for nc in self.neightboors.copy()]) or timeout:
                    # Printea cuando se van a tener problemas de desincronizacion
                    if initial_candidates_len != len(self.neightboors):
                        logging.error("({}) Not all nodes voted. Problem will be resolved on gossip without full connected topology.".format(self.get_name()))

                    if timeout:
                        logging.info("({}) Timeout for vote agregation.".format(self.get_name()))

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
                    results = sorted(results.items(), key=lambda x: x[0], reverse=True) # to equal solve of draw
                    results = sorted(results, key=lambda x: x[1], reverse=True)

                    #logging.info("({}) VOTING RESULTS: {}".format(self.get_name(),results))

                    top = min(len(results), Settings.TRAIN_SET_SIZE)
                    results = results[0:top]
                    results = {k: v for k, v in results}

                    votes = list(results.keys())

                    # Clear votes    
                    for n in self.neightboors:
                        n.clear_train_set_votes()

                    logging.info("({}) Computed {} votes.".format(self.get_name(),len(self.neightboors)+1))

                    return votes

                self.__wait_votes_ready_lock.acquire(timeout=2) 
                begin = time.time()                

        else:
            return []
                                
    def __validate_train_set(self):
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        #   ----> ESTA PARTE VA A SER TEDIOSA PARA REDES COMPLETAMENTE DESCENTRALIZADAS PUES NO SE TIENE UN DIRECTORIO DE NODOS
        for tsn in self.train_set:
            if tsn not in [ n.get_name() for n in self.neightboors]:
                if tsn != self.get_name():
                    self.train_set.remove(tsn)
                
        # If the node isnt connected
        if self.train_set == []:
            self.train_set = [self.get_name()]

        logging.info("{} Train set of {} nodes. {}".format(self.get_name(),len(self.train_set),self.train_set))


    def __on_round_finished(self):
        # Set Next Round
        self.agregator.clear()
        self.learner.finalize_round() # revisar x si esto pueiera quedar mejor
        self.round = self.round + 1
        # Clear node agregation
        for nc in [nc for nc in self.neightboors if nc.get_name() in self.train_set]:
            nc.set_models_agregated([])

        logging.info("({}) Round {} of {} finished.".format(self.get_name(),self.round,self.totalrounds))

        # Next Step or Finish
        if self.round < self.totalrounds:
            self.__train_step()  
        else:
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            self.round = None
            self.model_initialized = False
            logging.info("({}) Taining finished!!.".format(self.get_name(),self.round,self.totalrounds))

    #########################
    #    Model Gossiping    #    ->    Metodos bastante parecidos, mirar si se podr'ian unificar
    #########################

    def __gossip_model_agregation(self):
        # Anonymous function 
        candidate_condition = lambda nc: nc.get_name() in self.train_set and len(nc.get_models_agregated())<len(self.train_set)
        status_function = lambda nc: ( nc.get_name(),len(nc.get_models_agregated()) )
        model_function = lambda nc: self.agregator.get_partial_agregation(nc.get_models_agregated())

        # Gossip
        self.__gossip_model(candidate_condition,status_function,model_function)
         
    def __gossip_model_difusion(self,initialization=False):

        # Wait a model (init or agregated)
        if initialization:
            logging.info("({}) Waiting initialization.".format(self.get_name()))
            self.__wait_init_model_lock.acquire()
            logging.info("({}) Gossiping model initialization.".format(self.get_name(), len(self.neightboors)))
            candidate_condition = lambda nc: not nc.get_model_initialized()
        else:
            logging.info("({}) Waiting aregation.".format(self.get_name()))
            self.__finish_agregation_lock.acquire()
            logging.info("({}) Gossiping agregated model.".format(self.get_name(), len(self.neightboors)))
            candidate_condition = lambda nc: nc.get_ready_model_status()<=self.round

        # Anonymous function 
        status_function = lambda nc: nc.get_name()
        model_function = lambda _: (self.learner.get_parameters(),None) # At diffusion, contributors are not relevant

        # Gossip
        self.__gossip_model(candidate_condition,status_function,model_function)       
        
    def __gossip_model(self, candidate_condition, status_function, model_function):

        # Initialize list with status of nodes in the last X iterations
        last_x_status = [] 
        j = 0

        while True:
            
            # Get time to calculate frequency
            begin = time.time()

            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info("({}) Stopping model gossip process.".format(self.get_name()))
                return

            # Get nodes wich need models
            nei = [nc for nc in self.neightboors if candidate_condition(nc)]

            # Determine end of gossip
            if nei == []:
                logging.info("({}) Gossip finished.".format(self.get_name()))
                return

            # Save state of neightboors. If nodes are not responding gossip will stop
            if len(last_x_status) != Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS:
                last_x_status.append([status_function(nc) for nc in nei])
            else:
                last_x_status[j] = str([status_function for nc in nei])
                j = (j+1)%Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS

                # Check if las messages are the same
                for i in range(len(last_x_status)-1):
                    if last_x_status[i] != last_x_status[i+1]:
                        break
                    return

            # Select a random subset of neightboors
            samples = min(Settings.GOSSIP_MODELS_PER_ROUND,len(nei))
            nei = random.sample(nei, samples)

            # Lock Neightboors Communication
            for nc in nei:
                nc.set_sending_model(True)

            logging.info("({}) Gossiping model to {} train set nodes.".format(self.get_name(), len(nei)))

            # Generate and Send Model Partial Agregations (model, node_contributors)
            for nc in nei:
                model,contributors = model_function(nc)

                # Send Partial Agregation
                if model is not None:
                    encoded_model = self.learner.encode_parameters(params=model, contributors=contributors)
                    encoded_msgs = CommunicationProtocol.build_params_msg(encoded_model)
                    # Send Fragments
                    for msg in encoded_msgs:
                        nc.send(msg, True)
                        if Settings.FRAGMENTS_DELAY > 0:
                            time.sleep(Settings.FRAGMENTS_DELAY)
                
            # Lock Neightboors Communication
            for nc in nei:
                nc.set_sending_model(False)

            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1/Settings.GOSSIP_MODELS_FREC-time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)
