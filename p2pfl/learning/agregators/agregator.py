import sys
import threading
import logging
from p2pfl.const import AGREGATION_TIEMOUT

from p2pfl.learning.exceptions import ModelNotMatchingError
   
#-----------------------------------------------------------------------
# 
# Revisar otras estrategias de agregación para saber si se adaptarían
#
#-----------------------------------------------------------------------

class Agregator(threading.Thread):

    def __init__(self, n):
        threading.Thread.__init__(self)
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
            logging.info("Agregating models. Timeout reached")
            # Validamos que el nodo siga operativo (si no puediera quedar residual)
            if self.node.round is None:
                logging.info("Shutting Down Agregator Process")
                sys.exit() 
        else:
            logging.info("Agregating models.")
        self.node.learner.set_parameters(self.agregate(self.models))
        models_added = len(self.models)
        self.clear()
        # Notificamos al nodo
        self.node.on_round_finished(models_added)

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
        
    def check_and_run_agregation(self):
        # Try Unloock
        try:
            if len(self.models)==(len(self.node.neightboors)+1): 
                self.agregation_lock.release()
        except:
            pass


    def clear(self):
        self.__init__(self.node)
