from distutils.log import debug
import threading
import logging
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable
   
#-----------------------------------------------------------------------
# 
# PUESTOS A HACER MENSAJES MAS COMPLEJOS, REVISAR QUE LAS ACTUALIZACIONES REALIZADAS POR LOS DIFERENTES NODOS SEAN CORRESPONDIENTES A LAS RONDAS ACTUALES
#
#-----------------------------------------------------------------------

class Agregator(threading.Thread, Observable):
    """
    Class to manage the agregation of models. Its a thread so, agregation will be done in background if all models was added or timeouts have gone. 
    Also its a observable so, it will notify when the agregation was done.

    Args:
        node_name: (str): String with the name of the node.
    """

    def __init__(self, node_name="unknown"):
        threading.Thread.__init__(self)
        self.daemon = True
        Observable.__init__(self)
        self.train_set = []
        self.node_weights = {}
        self.waiting_agregated_model = False
        self.__agregated_waited_model = False
        self.node_name = node_name
        self.name = "agregator-" + node_name
        self.models = {}
        self.lock = threading.Lock()
        self.agregation_lock = threading.Lock()
        self.agregation_lock.acquire()

        self.thread_executed = False

    def run(self):
        """
        Wait for the agregation to be done or timeout. Then agregate the models and notify.
        """
        self.thread_executed = True

        # Wait for all models to be added or TIMEOUT
        self.agregation_lock.acquire(timeout=Settings.AGREGATION_TIMEOUT) 
        
        # Check if node still running (could happen if agregation thread was a residual thread)
        if self.train_set == []:
            logging.info("({}) Shutting Down Agregator Process".format(self.node_name))
            self.notify(Events.AGREGATION_FINISHED,None) # To avoid residual trainning-thread
            return
        
        # Start agregation
        n_model_agregated = sum([len(nodes.split()) for nodes in list(self.models.keys())])
        if n_model_agregated != len(self.train_set):
            logging.info("({}) Agregating models, timeout reached. Missing models: {}".format(self.node_name,set(self.train_set)-set(self.models.keys())))
        else:
            logging.info("({}) Agregating models.".format(self.node_name))

        # Notificamos al nodo
        self.notify(Events.AGREGATION_FINISHED,self.agregate(self.models)) 

    def agregate(self,models): 
        """
        Agregate the models.
        """
        print("Not implemented")
            
    def set_nodes_to_agregate(self, l):
        """
        List with the name of nodes to agregate.

        Args:
            n: Number of nodes to agregate. Empty for no agregation.
        """
        # Start Timeout            
        self.train_set = l
        if self.train_set != [] and not self.thread_executed:
            self.start()

    def set_waiting_agregated_model(self):
        """
        Indicates that the node is waiting for an agregation. It won't participate in agregation process.
        """
        self.waiting_agregated_model = True

    def set_node_weights(self,w):
        self.node_weights = w

    def get_node_weight(self,node):
        try:
            return self.node_weights[node]
        except:
            # If not exist, then return 0 (ponderate by 0)
            return 0

    def add_model(self, model, nodes):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            n: Node. Used to identify the model.
            m: Model.
            w: Number of samples used to train the model.

        """

        #
        # Weights should be in a list
        #

        if self.waiting_agregated_model and not self.__agregated_waited_model:
            logging.info("({}) Recived an agregated model.".format(self.node_name))
            self.__agregated_waited_model = True
            self.notify(Events.AGREGATION_FINISHED,model) 
        else:
            if nodes is not None:
                self.lock.acquire()

                # Get a list of nodes added
                models_added = [nodes.split() for nodes in list(self.models.keys())] 
                models_added = [element for sublist in models_added for element in sublist] # Flatten list
                
                #logging.debug("{} models_added: {} / {}".format(self.node_name,models_added,self.train_set))
                
                # Check if agregation is needed
                if len(self.train_set)>len(models_added):
                    # Check if all nodes are in the train_set 
                    if all([n in self.train_set for n in nodes]): 
                        # Check if all nodes are not agregated                    
                        if all([n not in models_added for n in nodes]):
                            # Agregar modelo
                            self.models[" ".join(nodes)] = ((model, sum(self.get_node_weight(n) for n in nodes)))
                            logging.info("({}) Model added ({}/{}) from {}".format(self.node_name, str(len(models_added)+len(nodes)), str(len(self.train_set)), str(nodes)))
                            # Check if all models have been added
                            self.check_and_run_agregation()
                            # Try Unloock
                            try:
                                self.lock.release()
                            except:
                                pass 
                            
                            
                            #print(self.models.keys())
                            return models_added + nodes
                    """
                        else:
                            logging.debug("({}) Can't add a model that has already been added {}".format(self.node_name, nodes))
                    else:
                        logging.info("nodes: {} | trainset: {}".format(nodes,self.train_set))
                        logging.debug("({}) Can't add a model from a node ({}) that is not in the training test.".format(self.node_name, nodes))
                    """
                    
                    
        try:
            self.lock.release()
        except:
            pass 
                        
        return None
                
    def get_partial_agregation(self,except_nodes):
        """
        Get the partial agregation of the models.

        Args:
            except_nodes: Nodes to exclude.
        """

        dict_aux = {}
        nodes_agregated = []
        models = self.models.copy()
        for n,m in list(models.items()):
            splited_nodes = n.split() 
            if all([n not in except_nodes for n in splited_nodes]):
                dict_aux[n] = m
                nodes_agregated += splited_nodes

        if len(dict_aux) == 0:
            return None,None

        return (self.agregate(dict_aux),nodes_agregated)
            
    def check_and_run_agregation(self,force=False):
        """
        Check if all models have been added and start agregation if so.

        Args:
            force: If true, agregation will be started even if not all models have been added.
        """
        models_added = [nodes.split() for nodes in list(self.models.keys())] 
        models_added = [element for sublist in models_added for element in sublist] # Flatten list
        # Try Unloock
        try:
            if (force or len(models_added)>=len(self.train_set)) and self.train_set!=[]: 
                self.agregation_lock.release()
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
