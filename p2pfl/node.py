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

###################################################################################################################
# FULL CONNECTED HAY QUE IMPLEMENTARLO DE FORMA QUE CUANDO SE INTRODUCE UN NODO EN LA RED, SE HACE UN BROADCAST
###################################################################################################################

class Node(BaseNode):

    #####################
    #     Node Init     #
    #####################

    def __init__(self, model, data, host="127.0.0.1", port=0, learner=LightningLearner, agregator=FedAvg, simulation=True):
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
        self.is_model_init = False

        # Locks
        self.__wait_models_ready_lock = threading.Lock()
        self.__wait_votes_ready_lock = threading.Lock()
        self.__finish_agregation_lock = threading.Lock()
        self.__finish_agregation_lock.acquire()
        
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

        # Execute Node update
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

                # Refresh training process waiters            
                try:
                    self.__wait_models_ready_lock.release()
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

        elif event == Events.NODE_MODELS_READY_EVENT:
            # Try to unlock to check if all nodes are ready (on_finish_round (agregator_thread))
            try:
                self.__wait_models_ready_lock.release()
            except:
                pass

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
            self.add_model(obj[0],obj[1],obj[2])
        
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
            logging.info("({}) Sending Initial Model Weights".format(self.get_name()))
            self.is_model_init = True
            self.__bc_model()
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

            # Indicates samples that be used in the learning process
            logging.info("({}) Broadcasting Number of Samples...".format(self.get_name()))  # esto meterlo en el handsaking

            #esto ya no hará falta -> ahora multilee -> tb nos perjudica porque si ya se está mandado el modelo, va a promediarlo x 0
            self.broadcast(CommunicationProtocol.build_num_samples_msg(self.learner.get_num_samples())) # si no se manda bien promedia x 0
            
            #esto de aqui es una apaño de los malos -> cambiar por lock
            if not self.is_model_init:
                logging.info("({}) Initialicing Model Weights".format(self.get_name()))
                while not self.is_model_init:
                    time.sleep(0.1)

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
                    decoded_model, contributors = self.learner.decode_parameters(m)
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.agregator.add_model(node,decoded_model,w)
                        if models_added is not None:
                            self.broadcast(CommunicationProtocol.build_models_agregated_msg(models_added))
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    # Initialize model
                    self.is_model_init = True
                    logging.info("({}) Model initialized".format(self.get_name()))
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

            # Set Models To Agregate 
            self.agregator.set_nodes_to_agregate(self.train_set) # reference to self node stringified list of nodes
            
            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()
            
            # Agregate Model
            if self.round is not None:
                self.agregator.add_model(self.get_name(),self.learner.get_parameters(), self.learner.get_num_samples()[0])
                self.__gossip_agregation() # this is going to produce duplicated models -> buut its fault tolerent

            # Broadcast to non train_set nodes
            if self.round is not None:
                self.__bc_model(train_set=False)

        else: 
            # Set Models To Agregate 
            self.agregator.set_waiting_agregated_model()

        # Synchronize all nodes
        if self.round is not None:
            self.__sync_nodes()

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


    def __gossip_agregation(self):

        #
        # Meterle un timeout para no enviar infinito
        #
        
        # MODELS AGREGATED vs MODELS READY -> revisar diferencias

        while True:
            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info("({}) Stopping on_round_finished process.".format(self.get_name()))
                return

            # Get time to calculate frequency
            begin = time.time()

            # Model for agregation is a tuple (model, node_contributors)
            model = self.learner.encode_parameters()
            encoded_msgs = CommunicationProtocol.build_params_msg(model)

            logging.info("({}) Broadcasting model to train set nodes. (size: {} bytes)".format(self.get_name(),len(encoded_msgs)*Settings.BUFFER_SIZE))
            nei = [nc for nc in self.neightboors if nc.get_name() in self.train_set and len(nc.get_models_agregated())<len(self.train_set)]

            if nei == []:
                logging.info("({}) GOssip finished.".format(self.get_name()))
                # HARDCODED PERO ES PA VER SI TIRA
                for nc in [nc for nc in self.neightboors if nc.get_name() in self.train_set]:
                    nc.set_models_agregated([])
                break

            # Select a random subset of neightboors
            samples = min(Settings.GOSSIP_MODELS_PER_ROUND,len(nei))
            nei = random.sample(nei, samples)

            # Lock Neightboors Communication
            for nc in nei:
                nc.set_sending_model(True)

            # Send Fragments
            for msg in encoded_msgs:
                for n in nei:
                    n.send(msg, True)

            # Lock Neightboors Communication
            for nc in nei:
                nc.set_sending_model(False)

            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1/Settings.GOSSIP_MODELS_FREC-time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)


    def __bc_model(self, train_set=None):
        encoded_msgs = CommunicationProtocol.build_params_msg(self.learner.encode_parameters())
        exclude = []
        if train_set is not None:
            if train_set:
                logging.info("({}) Broadcasting model to train set nodes. (size: {} bytes)".format(self.get_name(),len(encoded_msgs)*Settings.BUFFER_SIZE))
                exclude = [x for x in self.neightboors if x.get_name() not in self.train_set]
            else:
                logging.info("({}) Broadcasting model to non train set nodes. (size: {} bytes)".format(self.get_name(),len(encoded_msgs)*Settings.BUFFER_SIZE))
                exclude = [x for x in self.neightboors if x.get_name() in self.train_set]
        else:
            logging.info("({}) Broadcasting model to {} nodes. (size: {} bytes)".format(self.get_name(),len(self.neightboors),len(encoded_msgs)*Settings.BUFFER_SIZE))

        # Lock Neightboors Communication
        self.__set_sending_model(True)
        # Send Fragments
        for msg in encoded_msgs:
            self.broadcast(msg, exc=exclude)
        # UnLock Neightboors Communication
        self.__set_sending_model(False)


    # SI ESTAMOS EN SIMULACION NO PERMITIR BROADCASTEAERLAS -> tonter'ia
    def __bc_metrics(self,metrics): 
        logging.info("({}) Broadcasting metrics to {} clients.".format(self.get_name(),len(self.neightboors)))
        encoded_msgs = CommunicationProtocol.build_metrics_msg(self.round,metrics[0],metrics[1])
        self.broadcast(encoded_msgs)

    def __set_sending_model(self, flag):
        for node in self.neightboors:
            node.set_sending_model(flag)

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
            while True:
                # If the trainning has been interrupted, stop waiting
                if self.round is None:
                    logging.info("({}) Stopping on_round_finished process.".format(self.get_name()))
                    return []
                            
                logging.debug("({}) Waiting other node votes: {}".format(self.get_name(),[ (nc.get_name(),nc.get_train_set_votes()!=[]) for nc in self.neightboors]))

                if all([ nc.get_train_set_votes()!=[] for nc in self.neightboors]):
                    # Printea cuando se van a tener problemas de desincronizacion
                    if initial_candidates_len != len(self.neightboors):
                        logging.error("({}) Not all nodes voted. Problem will be resolved on gossip without full connected topology.".format(self.get_name()))

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

                    logging.info("({}) VOTING RESULTS: {}".format(self.get_name(),results))

                    top = min(len(results), Settings.TRAIN_SET_SIZE)
                    results = results[0:top]
                    results = {k: v for k, v in results}

                    votes = list(results.keys())

                    # Clear votes    
                    for n in self.neightboors:
                        n.clear_train_set_votes()

                    logging.info("({}) Computed {} votes.".format(self.get_name(),len(self.neightboors)+1))

                    return votes

                self.__wait_votes_ready_lock.acquire(timeout=2) # si se tarda m'as de x continuar 
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
    
    def __sync_nodes(self):
        try:
            # Wait to finish self agregation
            self.__finish_agregation_lock.acquire()
                
            # Verify that trainning has not been interrupted
            if self.round is None:
                return
                
            # Wait for ready messages
            logging.info("({}) Waiting other nodes to synchorinize the end of the round.".format(self.get_name()))
            while True:
                # If the trainning has been interrupted, stop waiting
                if self.round is None:
                    logging.info("({}) Stopping on_round_finished process.".format(self.get_name()))
                    return
                        
                train_set = [x for x in self.neightboors if x.get_name() in self.train_set]
                
                if all([ nc.get_ready_model_status()>=self.round for nc in train_set]):
                    break
                self.__wait_models_ready_lock.acquire(timeout=2)
                               
        except Exception as e:
            logging.error("({}) Concurrence Error: {}".format(self.get_name(),e))

    def __on_round_finished(self):
        # Set Next Round
        self.learner.finalize_round() # revisar x si esto pueiera quedar mejor
        self.round = self.round + 1
        logging.info("({}) Round {} of {} finished.".format(self.get_name(),self.round,self.totalrounds))

        # Next Step or Finish
        if self.round < self.totalrounds:
            self.__train_step()  
        else:
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            self.round = None
            self.is_model_init = False
            logging.info("({}) Finish!!.".format(self.get_name()))   