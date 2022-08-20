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

class Node(BaseNode):
    """
    Class based on a base node that allows **p2p Federated Learning**. 
    
    Metrics will be saved under a folder with the name of the node.
            
    Args:
        model: Model to be learned. Carefull, model should be compatible with data and the learner.
        data: Dataset to be used in the learning process. Carefull, model should be compatible with data and the learner.
        host (str): Host where the node will be listening.
        port (int): Port where the node will be listening.
        learner: Learner to be used in the learning process. Default: LightningLearner.
        agregator (Agregator): Agregator to be used in the learning process. Default: FedAvg.
        simulation (bool): If False, node will share metrics and communication will be encrypted. Default: True.

    Attributes:
        round (int): Round of the learning process.
        totalrounds (int): Total number of rounds of the learning process.
        learner (Learner): Learner to be used in the learning process.
        agregator (Agregator): Agregator to be used in the learning process.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(self, model, data, host="127.0.0.1", port=None, learner=LightningLearner, agregator=FedAvg, simulation=True):
        # Super init
        BaseNode.__init__(self,host,port, simulation)
        Observer.__init__(self)

        # Learning
        self.round = None 
        self.totalrounds = None 
        self.__model_initialized = False
        self.__initial_neighbors = []
        self.__start_thread_lock = threading.Lock()

        # Learner
        self.learner = learner(model, data, log_name=self.get_name()) 

        # Agregator
        self.agregator = agregator( node_name = self.get_name() )
        self.agregator.add_observer(self)

        # Train Set Votes
        self.__train_set = []
        self.__train_set_votes = {}
        self.__train_set_votes_lock = threading.Lock()

        # Locks
        self.__wait_votes_ready_lock = threading.Lock()
        self.__finish_agregation_lock = threading.Lock()
        self.__finish_agregation_lock.acquire()
        self.__wait_init_model_lock = threading.Lock()
        self.__wait_init_model_lock.acquire()

    #########################
    #    Node Management    #
    #########################

    def connect_to(self, h, p, full=False, force=False):
        """"
        Connects a node to other. If learning is running connections are not allowed (it should be forced).
        Carefull, if connection is forced with a new node, it will produce timeouts in the network.
    
        Args:
            h (str): The host of the node.
            p (int): The port of the node.
            full (bool): If True, the node will be connected to the entire network.
            force (bool): If True, the the node will be connected even though it should not be.

        Returns:
            node: The node that has been connected to.
        """
        # Check if learning is running
        if self.round is not None and not force:
            logging.info("({}) Cant connect to other nodes when learning is running.".format(self.get_name()))
            return None

        # Connect
        return super().connect_to(h, p, full, force)

    def stop(self): 
        """
        Stop the node and the learning if it is running.
        """
        if self.round is not None:
            self.__stop_learning()
        self.learner.close()
        super().stop()

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data):
        """
        Set the data to be used in the learning process (learner).

        Args:
            data: Dataset to be used in the learning process.
        """
        self.learner.set_data(data)

    def set_model(self, model):
        """"
        Set the model to use. 
        Carefully, model, not only weights.

        Args:
            model: Model to be learned.
        """
        self.learner.set_model(model)

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
        if self._terminate_flag.is_set():
            logging.info("({}) Node must be running to start learning".format(self.get_name()))
            return
        if self.round is None:
            # Start Learning
            logging.info("({}) Broadcasting start learning...".format(self.get_name()))
            self.broadcast(CommunicationProtocol.build_start_learning_msg(rounds,epochs))
            # Initialize model
            self.broadcast(CommunicationProtocol.build_model_initialized_msg())
            self.__wait_init_model_lock.release()
            self.__model_initialized = True # esto seguramente sobre, con locks es suficiente
            # Learning Thread
            self.__start_learning_thread(rounds,epochs)
        else:
            logging.info("({}) Learning already started".format(self.get_name()))

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.__stop_learning()
        else:
            logging.info("({}) Learning already stopped".format(self.get_name()))


    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self,rounds,epochs):
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

            begin = time.time()

            # Wait and gossip model inicialization            
            self.__gossip_model_difusion(initialization=True)

            # Wait to guarantee new connection heartbeats convergence and fix neighbors
            wait_time = Settings.WAIT_HEARTBEATS_CONVERGENCE - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)
            self.__initial_neighbors = self.get_neighbors() # used to restore the original list of neighbors after the learning round

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

    def add_model(self,m): 
        """
        Add a model. If the model isn't inicializated, the recieved model is used for it. Otherwise, the model is agregated using the **agregator**.

        Args:
            m: Encoded model. Contains model and their contributors
        """
        # Check if Learning is running
        if self.round is not None:
            try:
                if self.__model_initialized:
                    # Add model to agregator
                    decoded_model, contributors, weight = self.learner.decode_parameters(m)
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.agregator.add_model(decoded_model,contributors,weight)
                        if models_added is not None:
                            # CAREFULL RARE BUG at MACBOOCK: When CPU is high, only new nodes will be sent.
                            self.broadcast(CommunicationProtocol.build_models_agregated_msg(models_added))
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    # Initialize model
                    model, _, _ = self.learner.decode_parameters(m)
                    self.learner.set_parameters(model)
                    self.__model_initialized = True
                    logging.info("({}) Initialicing Model Weights".format(self.get_name()))
                    self.__wait_init_model_lock.release()    
                    self.broadcast(CommunicationProtocol.build_model_initialized_msg())
            
            except DecodingParamsError as e:
                logging.error("({}) Error decoding parameters".format(self.get_name()))
                self.stop()

            except ModelNotMatchingError as e:
                logging.error("({}) Models not matching.".format(self.get_name()))
                self.stop()
                    
            except Exception as e:
                self.stop()
                raise(e)
        else: 
            logging.error("({}) Tried to add a model while learning is not running".format(self.get_name()))
    
    #######################
    #    Trainig Steps    #
    #######################

    def __train_step(self):

        # Set train set
        if self.round is not None:
            self.__train_set = self.__vote_train_set()
            self.__validate_train_set()

        # Determine if node is in the train set
        is_train_set = self.get_name() in self.__train_set
        if is_train_set:

            # Full connect train set
            if self.round is not None:
                self.__connect_and_set_agregator()

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()
            
            # Agregate Model
            if self.round is not None:
                self.agregator.add_model(self.learner.get_parameters(),[self.get_name()],self.learner.get_num_samples()[0])
                self.broadcast(CommunicationProtocol.build_models_agregated_msg([self.get_name()])) # Notify agregation
                self.__gossip_model_agregation()
        else: 

            # Set Models To Agregate 
            self.agregator.set_waiting_agregated_model()

        # Gossip agregated model (also syncrhonizes nodes)
        if self.round is not None:
            self.__gossip_model_difusion()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()

    ################
    #    Voting    #
    ################

    def __vote_train_set(self):

        # Vote
        candidates = self.get_network_nodes() # al least himself
        logging.debug("({}) {} candidates to train set".format(self.get_name(),len(candidates)))
        if self.get_name() not in candidates:
            candidates.append(self.get_name())

        # Send vote
        samples = min(Settings.TRAIN_SET_SIZE,len(candidates))
        nodes_voted = random.sample(candidates, samples)
        weights = [math.floor(random.randint(0,1000)/(i+1)) for i in range(samples)]
        votes = list(zip(nodes_voted,weights))

        # Adding votes
        self.__train_set_votes_lock.acquire()
        self.__train_set_votes[self.get_name()] = dict(votes)
        self.__train_set_votes_lock.release()

        # Send and wait for votes
        logging.info("({}) Sending train set vote.".format(self.get_name()))
        logging.debug("({}) Self Vote: {}".format(self.get_name(),votes))
        self.broadcast(CommunicationProtocol.build_vote_train_set_msg(self.get_name(),votes))
        logging.debug("({}) Waiting other node votes.".format(self.get_name()))

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
            begin = time.time()                
            timeout = count > Settings.VOTE_TIMEOUT

            # Clear non candidate votes
            self.__train_set_votes_lock.acquire()
            nc_votes = {k:v for k,v in self.__train_set_votes.items() if k in candidates}
            self.__train_set_votes_lock.release()
                
            # Determine if all votes are received
            votes_ready = set(candidates) == set(nc_votes.keys())
            if votes_ready or timeout:

                if timeout and not votes_ready:
                    logging.info("({}) Timeout for vote agregation. Missing votes from {}".format(self.get_name(), set(candidates) - set(nc_votes.keys())))

                results = {}
                for node_vote in list(nc_votes.values()):
                    for i in range(len(node_vote)):
                        k = list(node_vote.keys())[i]
                        v = list(node_vote.values())[i]
                        if k in results:
                            results[k] += v
                        else:
                            results[k] = v

                # Order by votes and get TOP X
                results = sorted(results.items(), key=lambda x: x[0], reverse=True) # to equal solve of draw
                results = sorted(results, key=lambda x: x[1], reverse=True)
                top = min(len(results), Settings.TRAIN_SET_SIZE)
                results = results[0:top]
                results = {k: v for k, v in results}
                votes = list(results.keys())

                # Clear votes
                self.__train_set_votes = {}
                logging.info("({}) Computed {} votes.".format(self.get_name(),len(nc_votes)))
                return votes
                
            # Wait for votes or refresh every 2 seconds
            self.__wait_votes_ready_lock.acquire(timeout=2) 
        
                                
    def __validate_train_set(self):
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in self.__train_set:
            if tsn not in self.get_network_nodes():
                if tsn != self.get_name():
                    self.__train_set.remove(tsn)
                
        logging.info("{} Train set of {} nodes: {}".format(self.get_name(),len(self.__train_set),self.__train_set))

    ##########################
    #    Connect Trainset    #
    ##########################

    def __connect_and_set_agregator(self):
        # Set Models To Agregate 
        self.agregator.set_nodes_to_agregate(self.__train_set)

        # Connect Train Set Nodes
        for node in self.__train_set :
            if node != self.get_name():
                h,p = node.split(":")
                if p.isdigit():
                    nc = self.get_neighbor(h,int(p))
                    # If the node is not connected, connect it (to avoid duplicated connections only a node connects to the other)
                    if nc is None and self.get_name() > node: 
                        self.connect_to(h,int(p),force=True)
                else:
                    logging.info("({}) Node {} has a valid port".format(self.get_name(),node.split(":")))

        # Wait connections
        count = 0
        begin = time.time()
        while True:
            count = count + (time.time() - begin)
            if count > Settings.TRAIN_SET_CONNECT_TIMEOUT:
                logging.info("({}) Timeout for train set connections.".format(self.get_name()))
                break
            if len(self.__train_set) == len([nc for nc in self.get_neighbors() if nc.get_name() in self.__train_set])+1:
                break
            time.sleep(0.1) 


    ############################
    #    Train and Evaluate    #
    ############################

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
                logging.info("({}) Broadcasting metrics.".format(self.get_name(),len(self.get_neighbors())))
                encoded_msgs = CommunicationProtocol.build_metrics_msg(self.get_name(),self.round,results[0],results[1])
                self.broadcast(encoded_msgs)

    ######################
    #    Round finish    #
    ######################

    def __on_round_finished(self):
        # Remove trainset connections
        for nc in self.get_neighbors():
            if nc not in self.__initial_neighbors:
                self.rm_neighbor(nc)
        # Set Next Round
        self.agregator.clear()
        self.learner.finalize_round() # revisar x si esto pueiera quedar mejor
        self.round = self.round + 1
        # Clear node agregation
        for nc in self.get_neighbors():
            nc.clear_models_agregated()

        # Next Step or Finish
        logging.info("({}) Round {} of {} finished.".format(self.get_name(),self.round,self.totalrounds))
        if self.round < self.totalrounds:
            self.__train_step()  
        else:
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            self.round = None
            self.totalrounds = None
            self.__model_initialized = False
            logging.info("({}) Training finished!!.".format(self.get_name(),self.round,self.totalrounds))

    #########################
    #    Model Gossiping    #
    #########################

    def __gossip_model_agregation(self):
        # Anonymous functions
        candidate_condition = lambda nc: nc.get_name() in self.__train_set and len(nc.get_models_agregated())<len(self.__train_set)
        status_function = lambda nc: ( nc.get_name(),len(nc.get_models_agregated()) )
        model_function = lambda nc: self.agregator.get_partial_agregation(nc.get_models_agregated())

        # Gossip
        self.__gossip_model(candidate_condition,status_function,model_function)
         
    def __gossip_model_difusion(self,initialization=False):
        # Wait a model (init or agregated)
        if initialization:
            logging.info("({}) Waiting initialization.".format(self.get_name()))
            self.__wait_init_model_lock.acquire()
            logging.info("({}) Gossiping model initialization.".format(self.get_name(), len(self.get_neighbors())))
            candidate_condition = lambda nc: not nc.get_model_initialized()
        else:
            logging.info("({}) Waiting aregation.".format(self.get_name()))
            self.__finish_agregation_lock.acquire()
            logging.info("({}) Gossiping agregated model.".format(self.get_name(), len(self.get_neighbors())))
            candidate_condition = lambda nc: nc.get_model_ready_status()<self.round

        # Anonymous functions
        status_function = lambda nc: nc.get_name()
        model_function = lambda _: (self.learner.get_parameters(),None, None) # At diffusion, contributors are not relevant
        
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
            nei = [nc for nc in self.get_neighbors() if candidate_condition(nc)]

            # Determine end of gossip
            if nei == []:
                logging.info("({}) Gossip finished.".format(self.get_name()))
                return

            # Save state of neightbors. If nodes are not responding gossip will stop
            if len(last_x_status) != Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS:
                last_x_status.append([status_function(nc) for nc in nei])
            else:
                last_x_status[j] = str([status_function(nc) for nc in nei])
                j = (j+1)%Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS

                # Check if las messages are the same
                for i in range(len(last_x_status)-1):
                    if last_x_status[i] != last_x_status[i+1]:
                        break
                    logging.info("({}) Gossiping exited for {} equal reounds.".format(self.get_name(), Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS))
                    return

            # Select a random subset of neightbors
            samples = min(Settings.GOSSIP_MODELS_PER_ROUND,len(nei))
            nei = random.sample(nei, samples)

            # Generate and Send Model Partial Agregations (model, node_contributors)
            for nc in nei:
                model,contributors,weights = model_function(nc)

                # Send Partial Agregation
                if model is not None:
                    logging.info("({}) Gossiping model to {}.".format(self.get_name(), nc.get_name()))
                    encoded_model = self.learner.encode_parameters(params=model, contributors=contributors, weight=weights)
                    encoded_msgs = CommunicationProtocol.build_params_msg(encoded_model)
                    # Send Fragments
                    for msg in encoded_msgs:
                        nc.send(msg)
                
            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1/Settings.GOSSIP_MODELS_FREC-time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)


    ###########################
    #     Observer Events     #
    ###########################

    def update(self,event,obj):
        """
        Observer update method. Used to handle events that can occur in the different components and connections of the node.
        
        Args:
            event (Events): Event that has occurred.
            obj: Object that has been updated. 
        """
        if event == Events.NODE_CONNECTED_EVENT:
            n, force = obj
            if self.round is not None and not force:
                logging.info("({}) Cant connect to other nodes when learning is running. (however, other nodes can be connected to the node.)".format(self.get_name()))
                n.stop()
                return
                
        elif event == Events.AGREGATION_FINISHED_EVENT:
            # Set parameters and communate it to the training process
            if obj is not None:
                self.learner.set_parameters(obj)
                # Share that agregation is done
                self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
            else:
                logging.error("({}) Agregation finished with no parameters".format(self.get_name()))
                self.stop()
            try:
                self.__finish_agregation_lock.release()
            except:
                pass

        elif event == Events.START_LEARNING_EVENT:
            self.__start_learning_thread(obj[0],obj[1])

        elif event == Events.STOP_LEARNING_EVENT:
            self.__stop_learning()
    
        elif event == Events.PARAMS_RECEIVED_EVENT:
            self.add_model(obj)
        
        elif event == Events.METRICS_RECEIVED_EVENT:
            name, round, loss, metric = obj
            self.learner.log_validation_metrics(loss,metric,round=round,name=name)

        elif event == Events.TRAIN_SET_VOTE_RECEIVED_EVENT:
            node,votes = obj
            self.__train_set_votes_lock.acquire()
            self.__train_set_votes[node] = votes
            self.__train_set_votes_lock.release()
            # Communicate to the training process that a vote has been received
            try:
                self.__wait_votes_ready_lock.release()
            except:
                pass

        # Execute BaseNode update
        super().update(event,obj)