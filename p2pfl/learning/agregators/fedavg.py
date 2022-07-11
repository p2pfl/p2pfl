import logging
import torch
from p2pfl.learning.agregators.agregator import Agregator
   
class FedAvg(Agregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def agregate(self,models): 
        """
        Ponderated average of the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """

        # Check if there are models to agregate
        if len(models)==0:
            logging.error("({}) Trying to agregate models when there is no models".format(self.node_name))
            return None

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