import threading
import logging
   
#-----------------------------------------------------------------------
# 
# Revisar otras estrategias de agregación para saber si se adaptarían
#
#-----------------------------------------------------------------------

class Agregator(threading.Thread):
 
    def add_model(self, m): pass
        
    def clear(self):
        self.__init__(self.node)

    def run(self):
        logging.info("Agregating models.")
        self.node.learner.set_parameters(FedAvg.agregate(self.models))
        self.clear()
        # Notificamos al nodo
        self.node.on_round_finished()

    def agregate(models): # (MEAN)
        # Sum
        sum= models[-1].copy()
        for m in models[:-1]:
            for layer in m:
                sum[layer] = sum[layer] + m[layer]  
        
        # Divide by the number of models
        for layer in sum:
            sum[layer] = sum[layer]/(len(models))

        return sum
            
        