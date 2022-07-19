import threading
import logging
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable
   
class Agregator(threading.Thread, Observable):
    """
    Class to manage the agregation of models. Its a thread so, agregation will be done in background if all models was added or timeouts have gone. 
    Also its a observable so, it will notify the node when the agregation was done.

    Args:
        node_name: (str): String with the name of the node.
    """
    def __init__(self, node_name="unknown"):
        self.node_name = node_name
        threading.Thread.__init__(self, name = "agregator-" + node_name)
        self.daemon = True
        Observable.__init__(self)
        self.__train_set = []
        self.__waiting_agregated_model = False
        self.__agregated_waited_model = False
        self.__models = {}
        self.__lock = threading.Lock()
        self.__agregation_lock = threading.Lock()
        self.__agregation_lock.acquire()
        self.__thread_executed = False

    def run(self):
        """
        Wait for the agregation to be done or timeout. Then agregate the models and notify.
        """
        self.__thread_executed = True

        # Wait for all models to be added or TIMEOUT
        self.__agregation_lock.acquire(timeout=Settings.AGREGATION_TIMEOUT) 
        
        # Check if node still running (could happen if agregation thread was a residual thread)
        if self.__train_set == []:
            logging.info("({}) Shutting Down Agregator Process".format(self.node_name))
            self.notify(Events.AGREGATION_FINISHED_EVENT,None) # To avoid residual trainning-thread
            return
        
        # Start agregation
        n_model_agregated = sum([len(nodes.split()) for nodes in list(self.__models.keys())])
        if n_model_agregated != len(self.__train_set):
            logging.info("({}) Agregating models, timeout reached. Missing models: {}".format(self.node_name,set(self.__train_set)-set(self.__models.keys())))
        else:
            logging.info("({}) Agregating models.".format(self.node_name))

        # Notify node
        self.notify(Events.AGREGATION_FINISHED_EVENT,self.agregate(self.__models)) 

    def agregate(self,models): 
        """
        Agregate the models.
        """
        print("Not implemented")
            
    def set_nodes_to_agregate(self, l):
        """
        List with the name of nodes to agregate.

        Args:
            l: List of nodes to agregate. Empty for no agregation.
        """
        self.__train_set = l

    def set_waiting_agregated_model(self):
        """
        Indicates that the node is waiting for an agregation. It won't participate in agregation process.
        """
        self.__waiting_agregated_model = True

    def add_model(self, model, nodes, weight):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            nodes: Nodes that colaborated to get the model.
            weight: Number of samples used to get the model.
        """
        if self.__waiting_agregated_model and not self.__agregated_waited_model:
            logging.info("({}) Recived an agregated model.".format(self.node_name))
            self.__agregated_waited_model = True
            self.notify(Events.AGREGATION_FINISHED_EVENT,model) 
        else:
            if nodes is not None:
                self.__lock.acquire()

                # Start agregation timeout
                if self.__train_set != [] and not self.__thread_executed:
                    self.start()  

                # Get a list of nodes added
                models_added = [n.split() for n in list(self.__models.keys())] 
                models_added = [element for sublist in models_added for element in sublist] # Flatten list
                
                # Check if agregation is needed
                if len(self.__train_set)>len(models_added):
                    # Check if all nodes are in the train_set 
                    if all([n in self.__train_set for n in nodes]): 
                        # Check if all nodes are not agregated                    
                        if all([n not in models_added for n in nodes]):
                            # Agregar modelo
                            self.__models[" ".join(nodes)] = ((model, weight))
                            logging.info("({}) Model added ({}/{}) from {}".format(self.node_name, str(len(models_added)+len(nodes)), str(len(self.__train_set)), str(nodes)))
                            # Check if all models have been added
                            self.check_and_run_agregation()
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
                
    def get_partial_agregation(self,except_nodes):
        """
        Get the partial agregation of the models.

        Args:
            except_nodes: Nodes to exclude.

        Returns:
            (model, nodes, weight): Model, nodes and number of samples for the partial agregation.
        """
        dict_aux = {}
        nodes_agregated = []
        agregation_weight = 0
        models = self.__models.copy()
        for n,(m,s) in list(models.items()):
            splited_nodes = n.split() 
            if all([n not in except_nodes for n in splited_nodes]):
                dict_aux[n] = (m,s)
                nodes_agregated += splited_nodes
                agregation_weight += s
        
        # If there are no models to agregate
        if len(dict_aux) == 0:
            return None,None,None

        return (self.agregate(dict_aux), nodes_agregated, agregation_weight)
            
    def check_and_run_agregation(self,force=False):
        """
        Check if all models have been added and start agregation if so.

        Args:
            force: If true, agregation will be started even if not all models have been added.
        """
        models_added = [nodes.split() for nodes in list(self.__models.keys())] 
        models_added = [element for sublist in models_added for element in sublist] # Flatten list
        # Try Unloock
        try:
            if (force or len(models_added)>=len(self.__train_set)) and self.__train_set!=[]: 
                self.__agregation_lock.release()
        except:
            pass

    def clear(self):
        """
        Clear all for a new agregation.
        """
        observers = self.get_observers()
        self.__init__(node_name=self.node_name)
        for o in observers:
            self.add_observer(o)
