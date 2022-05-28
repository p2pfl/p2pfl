import threading
import logging

from p2pfl.learning.agregators.agregator import Agregator
   
#-----------------------------------------------------
# Implementar un observador en condiciones -> notificar cuando acabe agregación -> algo acoplado pero ns como haría
#-----------------------------------------------------

#-----------------------------------------------------
# 
# FALTA LA EVALUACIÓN DEL MODELO
#
# Mencionar que num de epoches locales va en el nodo mas que en el agregador
#-----------------------------------------------------

"""
Federated Averaging (FedAvg) [McMahan et al., 2016]
Paper: https://arxiv.org/abs/1602.05629
"""

class FedAvg(Agregator):
    def __init__(self, n):
        threading.Thread.__init__(self)
        self.node = n
        self.models = []
        self.lock = threading.Lock()

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
            
    def add_model(self, m):
        self.lock.acquire()
        self.models.append(m)
        logging.info("Model added (" + str(len(self.models)) + "/" + str(len(self.node.neightboors)+1) + ")")
        # Check if all models have been added
        if len(self.models)==(len(self.node.neightboors)+1):
            self.start()
        self.lock.release()
        
    def clear(self):
        super().clear()
