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
        self.node.learner.set_parameters(self.agregate(self.models))
        self.clear()
        # Notificamos al nodo
        self.notify(Events.AGREGATION_FINISHED,None) 

    def agregate(self,models): print("Not implemented")
            
    def add_model(self, n, m, w):
        # Validar que el modelo sea del mismo tipo

        if self.node.learner.check_parameters(m):
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
        else:
            raise ModelNotMatchingError("Not matching models")
        
    def check_and_run_agregation(self,force=False):
        # Try Unloock
        try:
            if force or len(self.models)==(len(self.node.neightboors)+1): 
                print("({})releasing".format(self.node.get_addr()))
                self.agregation_lock.release()
        except:
            pass


    def clear(self):
        observers = self.get_observers()
        self.__init__(self.node)
        for o in observers:
            self.add_observer(o)
