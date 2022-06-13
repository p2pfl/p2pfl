from multiprocessing import Event
import sys
import threading
import logging
from p2pfl.const import AGREGATION_TIEMOUT

from p2pfl.learning.exceptions import ModelNotMatchingError
from p2pfl.utils.observer import Events, Observable
   
#-----------------------------------------------------------------------
# 
# Revisar otras estrategias de agregación para saber si se adaptarían
#
#-----------------------------------------------------------------------

class Agregator(threading.Thread, Observable):
    """
    Class to manage the agregation of models. Its a thread so, agregation will be done in background if all models was added or timeouts have gone. 
    Also its a observable so, it will notify when the agregation was done.

    Args:
        n: Node. Used to check the neightboors and decode parameters with the learner.
    """

    def __init__(self, n):
        threading.Thread.__init__(self)
        self.daemon = True
        Observable.__init__(self)
        self.node = n
        self.name = "agregator-" + n.get_addr()[0] + ":" + str(n.get_addr()[1])
        self.models = {}
        self.lock = threading.Lock()
        self.agregation_lock = threading.Lock()
        self.agregation_lock.acquire()

    def run(self):
        """
        Wait for the agregation to be done or timeout. Then agregate the models and notify.
        """
        # Wait for all models to be added or TIMEOUT
        self.agregation_lock.acquire(timeout=AGREGATION_TIEMOUT) 
        # Start agregation
        if len(self.models)!=(len(self.node.neightboors)+1):
            logging.info("({}) Agregating models. Timeout reached".format(self.node.get_addr()))
            # Validamos que el nodo siga operativo (si no puediera quedar residual)
            if self.node.round is None:
                logging.info("({}) Shutting Down Agregator Process".format(self.node.get_addr()))
                self.notify(Events.AGREGATION_FINISHED,None) # To avoid residual trainning-thread
                return
        else:
            logging.info("({}) Agregating models.".format(self.node.get_addr()))

        # Notificamos al nodo
        self.notify(Events.AGREGATION_FINISHED,self.agregate(self.models)) 
        self.clear()

    def agregate(self,models): 
        """
        Agregate the models.
        """
        print("Not implemented")
            
    def add_model(self, n, m, w):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            n: Node. Used to identify the model.
            m: Model.
            w: Number of samples used to train the model.

        """
        # Agregar modelo
        self.lock.acquire()
        self.models[n] = ((m, w))
        logging.info("({}) Model added ({}/{}) from {}".format(self.node.get_addr(), str(len(self.models)), str(len(self.node.neightboors)+1), n))
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
            if force or len(self.models)==(len(self.node.neightboors)+1): 
                print("({})releasing".format(self.node.get_addr()))
                self.agregation_lock.release()
        except:
            pass


    def clear(self):
        """
        Clear all for a new agregation.
        """
        observers = self.get_observers()
        self.__init__(self.node)
        for o in observers:
            self.add_observer(o)
