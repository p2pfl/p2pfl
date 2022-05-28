import threading
import logging

import torch

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
        # Total Samples
        total_samples = sum([y for _,y in models])

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Add weighteds models
        for m,w in models:
            for layer in m:
                accum[layer] = accum[layer] + m[layer]*w

        # Normalize Accum
        for layer in accum:
            accum[layer] = accum[layer]/total_samples

        return accum
            
    def add_model(self, m, w):
        self.lock.acquire()
        self.models.append((m, w))
        logging.info("Model added (" + str(len(self.models)) + "/" + str(len(self.node.neightboors)+1) + ")")
        # Check if all models have been added
        if len(self.models)==(len(self.node.neightboors)+1):
            self.start()
        self.lock.release()
        
    def clear(self):
        super().clear()
