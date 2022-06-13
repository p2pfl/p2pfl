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

class FedAvg(Agregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """
    def __init__(self, n):
        super().__init__(n)

    def agregate(self,models): 
        """
        Ponderated average of the models.

        Args:
            models: Dictionary with the models.
        """
        
        models = list(models.values())
        
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