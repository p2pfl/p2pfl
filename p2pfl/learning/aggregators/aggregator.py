import threading
import logging
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable
   
class Aggregator(threading.Thread, Observable):
    """
    Class to manage the aggregation of models. Its a thread so, aggregation will be done in background if all models was added or timeouts have gone. 
    Also its a observable so, it will notify the node when the aggregation was done.

    Args:
        node_name: (str): String with the name of the node.
    """
    def __init__(self, node_name="unknown"):
        self.node_name = node_name
        threading.Thread.__init__(self, name = "aggregator-" + node_name)
        self.daemon = True
        Observable.__init__(self)
        self.__train_set = []
        self.__waiting_aggregated_model = False
        self.__aggregated_waited_model = False
        self.__models = {}
        self.__lock = threading.Lock()
        self.__aggregation_lock = threading.Lock()
        self.__aggregation_lock.acquire()
        self.__thread_executed = False

    def run(self):
        """
        Wait for the aggregation to be done or timeout. Then aggregate the models and notify.
        """
        self.__thread_executed = True

        # Wait for all models to be added or TIMEOUT
        self.__aggregation_lock.acquire(timeout=Settings.AGGREGATION_TIMEOUT) 
        
        # Check if node still running (could happen if aggregation thread was a residual thread)
        if self.__train_set == []:
            logging.info("({}) Shutting Down Aggregator Process".format(self.node_name))
            self.notify(Events.AGGREGATION_FINISHED_EVENT,None) # To avoid residual trainning-thread
            return
        
        # Start aggregation
        n_model_aggregated = sum([len(nodes.split()) for nodes in list(self.__models.keys())])
        if n_model_aggregated != len(self.__train_set):
            logging.info("({}) Aggregating models, timeout reached. Missing models: {}".format(self.node_name,set(self.__train_set)-set(self.__models.keys())))
        else:
            logging.info("({}) Aggregating models.".format(self.node_name))

        # Notify node
        self.notify(Events.AGGREGATION_FINISHED_EVENT,self.aggregate(self.__models)) 

    def aggregate(self,models): 
        """
        Aggregate the models.
        """
        print("Not implemented")
            
    def set_nodes_to_aggregate(self, l):
        """
        List with the name of nodes to aggregate.

        Args:
            l: List of nodes to aggregate. Empty for no aggregation.
        """
        self.__train_set = l

    def set_waiting_aggregated_model(self):
        """
        Indicates that the node is waiting for an aggregation. It won't participate in aggregation process.
        """
        self.__waiting_aggregated_model = True

    def add_model(self, model, nodes, weight):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            nodes: Nodes that colaborated to get the model.
            weight: Number of samples used to get the model.
        """
        if self.__waiting_aggregated_model and not self.__aggregated_waited_model:
            logging.info("({}) Recived an aggregated model.".format(self.node_name))
            self.__aggregated_waited_model = True
            self.notify(Events.AGGREGATION_FINISHED_EVENT,model) 
        else:
            if nodes is not None:
                self.__lock.acquire()

                # Start aggregation timeout
                if self.__train_set != [] and not self.__thread_executed:
                    self.start()  

                # Get a list of nodes added
                models_added = [n.split() for n in list(self.__models.keys())] 
                models_added = [element for sublist in models_added for element in sublist] # Flatten list
                
                # Check if aggregation is needed
                if len(self.__train_set)>len(models_added):
                    # Check if all nodes are in the train_set 
                    if all([n in self.__train_set for n in nodes]): 
                        # Check if all nodes are not aggregated                    
                        if all([n not in models_added for n in nodes]):
                            # Aggregate model
                            self.__models[" ".join(nodes)] = ((model, weight))
                            logging.info("({}) Model added ({}/{}) from {}".format(self.node_name, str(len(models_added)+len(nodes)), str(len(self.__train_set)), str(nodes)))
                            # Check if all models have been added
                            self.check_and_run_aggregation()
                            # Build response 
                            response = models_added + nodes
                            # Unloock
                            self.__lock.release()
                            
                            return response
                        else:
                            self.__lock.release()
                            logging.debug("({}) Can't add a model that has already been added {}".format(self.node_name, nodes))
                    else:
                        self.__lock.release()
                        logging.debug("({}) Can't add a model from a node ({}) that is not in the training test.".format(self.node_name, nodes))                
                else:
                    self.__lock.release()

        return None
                
    def get_partial_aggregation(self,except_nodes):
        """
        Get the partial aggregation of the models.

        Args:
            except_nodes: Nodes to exclude.

        Returns:
            (model, nodes, weight): Model, nodes and number of samples for the partial aggregation.
        """
        dict_aux = {}
        nodes_aggregated = []
        aggregation_weight = 0
        models = self.__models.copy()
        for n,(m,s) in list(models.items()):
            splited_nodes = n.split() 
            if all([n not in except_nodes for n in splited_nodes]):
                dict_aux[n] = (m,s)
                nodes_aggregated += splited_nodes
                aggregation_weight += s
        
        # If there are no models to aggregate
        if len(dict_aux) == 0:
            return None,None,None

        return (self.aggregate(dict_aux), nodes_aggregated, aggregation_weight)
            
    def check_and_run_aggregation(self,force=False):
        """
        Check if all models have been added and start aggregation if so.

        Args:
            force: If true, aggregation will be started even if not all models have been added.
        """
        models_added = [nodes.split() for nodes in list(self.__models.keys())] 
        models_added = [element for sublist in models_added for element in sublist] # Flatten list
        # Try Unloock
        try:
            if (force or len(models_added)>=len(self.__train_set)) and self.__train_set!=[]: 
                self.__aggregation_lock.release()
        except:
            pass

    def clear(self):
        """
        Clear all for a new aggregation.
        """
        observers = self.get_observers()
        self.__init__(node_name=self.node_name)
        for o in observers:
            self.add_observer(o)
