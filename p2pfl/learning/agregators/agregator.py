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
        self.waiting_agregated_model = False
        self.node_name = node_name
        self.name = "agregator-" + node_name
        self.models = {}
        self.lock = threading.Lock()
        self.agregation_lock = threading.Lock()
        self.agregation_lock.acquire()

    def run(self):
        """
        Wait for the agregation to be done or timeout. Then agregate the models and notify.
        """
        # Wait for all models to be added or TIMEOUT
        self.agregation_lock.acquire(timeout=Settings.AGREGATION_TIEMOUT) 
        
        # Check if node still running (could happen if agregation thread was a residual thread)
        if self.train_set == []:
            logging.info("({}) Shutting Down Agregator Process".format(self.node_name))
            self.notify(Events.AGREGATION_FINISHED,None) # To avoid residual trainning-thread
            return
        
        # Start agregation
        if len(self.models)!=len(self.train_set):
            logging.info("({}) Agregating models. Timeout reached".format(self.node_name))
        else:
            logging.info("({}) Agregating models.".format(self.node_name))

        # Notificamos al nodo
        self.notify(Events.AGREGATION_FINISHED,self.agregate(self.models)) 
        self.clear()

    def agregate(self,models): 
        """
        Agregate the models.
        """
        print("Not implemented")
            
    def set_nodes_to_agregate(self, l):
        """
        Indicate the number of nodes to agregate. None when agregation is not needed.

        Args:
            n: Number of nodes to agregate. None for no agregation.
        """
        self.train_set = l
    
    def remove_node_to_agregate(self, node):
        """
        Indicates that a node is not going to be agregated.

        Args:
            node: Nodes to remove.
        """
        self.train_set.remove(node)
        logging.info("({}) Node Removed ({}/{})".format(self.node_name, str(len(self.models)), str(len(self.train_set))))
        # It cant produce training, if aggregation is running, clients only decrement
        self.check_and_run_agregation()
        
    def set_waiting_agregated_model(self):
        self.waiting_agregated_model = True

    def add_model(self, n, m, w):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            n: Node. Used to identify the model.
            m: Model.
            w: Number of samples used to train the model.

        """
        if self.waiting_agregated_model:
            logging.info("({}) Recived a model from {}".format(self.node_name, n))
            self.waiting_agregated_model = False
            self.notify(Events.AGREGATION_FINISHED,m) 
        else:
            if len(self.train_set)>len(self.models):
                if (str(n)) in [str(x) for x in self.train_set]: # To avoid comparing pointers
                    if n not in self.models:
                        # Agregar modelo
                        self.lock.acquire()
                        self.models[n] = ((m, w))
                        logging.info("({}) Model added ({}/{}) from {}".format(self.node_name, str(len(self.models)), str(len(self.train_set)), n))
                        # Start Timeout
                        if not self.is_alive():
                            self.start()
                        # Check if all models have been added
                        self.check_and_run_agregation()
                        # Try Unloock
                        try:
                            self.lock.release()
                        except:
                            pass 
                
                    else:
                        logging.info("({}) Can't add a model that has already been added {}".format(self.node_name, n))
                else:
                    logging.info("({}) Can't add a model from a node ({}) that is not in the training test.".format(self.node_name, n))
  
                
            
    def check_and_run_agregation(self,force=False):
        """
        Check if all models have been added and start agregation if so.

        Args:
            force: If true, agregation will be started even if not all models have been added.
        """
        # Try Unloock
        try:
            if (force or len(self.models)>=len(self.train_set)) and self.train_set!=[]: 
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
