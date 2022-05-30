import threading
import logging
   
#-----------------------------------------------------------------------
# 
# Revisar otras estrategias de agregación para saber si se adaptarían
#
#-----------------------------------------------------------------------

class Agregator(threading.Thread):

    def __init__(self, n):
        threading.Thread.__init__(self)
        self.node = n
        self.models = []
        self.lock = threading.Lock()

    def run(self):
        logging.info("Agregating models.")
        self.node.learner.set_parameters(self.agregate(self.models))
        self.clear()
        # Notificamos al nodo
        self.node.on_round_finished()

    def agregate(self,models): print("Not implemented")
            
    def add_model(self, m, w):
        self.lock.acquire()
        self.models.append((m, w))
        logging.info("Model added (" + str(len(self.models)) + "/" + str(len(self.node.neightboors)+1) + ")")
        # Check if all models have been added
        if len(self.models)==(len(self.node.neightboors)+1):
            self.start()
        self.lock.release()
        
    def clear(self):
        self.__init__(self.node)
