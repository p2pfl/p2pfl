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
        self.train_set_size = None
        self.node_name = node_name
        self.name = "agregator-" + node_name
        self.models = {}
        self.result_model = None # ---------------------------------------
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
        if self.train_set_size is None:
            logging.info("({}) Shutting Down Agregator Process".format(self.node_name))
            self.notify(Events.AGREGATION_FINISHED,None) # To avoid residual trainning-thread
            return
        
        # Start agregation
        if len(self.models)!=(self.train_set_size):
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
            
    def set_nodes_to_agregate(self, n):
        """
        Indicate the number of nodes to agregate. None when agregation is not needed.

        Args:
            n: Number of nodes to agregate. None for no agregation.
        """
        self.train_set_size = n
    
    def remove_node_to_agregate(self, ammount=1):
        """
        Indicates that a node/s is not going to be agregated.

        Args:
            ammount: Number of nodes to remove.
        """
        if self.train_set_size is not None:
            self.train_set_size = self.train_set_size-ammount
            # It cant produce training, if aggregation is running, clients only decrement
            logging.info("({}) Node Removed ({}/{})".format(self.node_name, str(len(self.models)), str(self.train_set_size)))
            self.check_and_run_agregation()

    def add_model(self, n, m, w):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            n: Node. Used to identify the model.
            m: Model.
            w: Number of samples used to train the model.

        """
        if self.train_set_size is None:
            logging.error("({}) Error, trying to add a model when the neighbors are not specificated".format(self.node_name))
        else:
            if self.train_set_size>len(self.models):
                # Agregar modelo
                self.lock.acquire()
                self.models[n] = ((m, w))
                logging.info("({}) Model added ({}/{}) from {}".format(self.node_name, str(len(self.models)), str(self.train_set_size), n))
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
        
    def check_and_run_agregation(self,force=False):
        """
        Check if all models have been added and start agregation if so.

        Args:
            force: If true, agregation will be started even if not all models have been added.
        """
        # Try Unloock
        try:
            if force or len(self.models)>=(self.train_set_size): 
                self.agregation_lock.release()
        except:
            pass


    def clear(self):
        """
        Clear all for a new agregation.
        """
        observers = self.get_observers()
        train_set_size = self.train_set_size
        self.__init__(node_name=self.node_name)
        self.train_set_size = train_set_size
        for o in observers:
            self.add_observer(o)
