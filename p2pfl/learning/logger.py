from pytorch_lightning.loggers import TensorBoardLogger
import torch

class FederatedTensorboardLogger(TensorBoardLogger):

    def __init__(self, save_dir, name = "training_logs" , version = 0, **kwargs):
        super().__init__(save_dir, name, version=version, **kwargs)
        self.round = 0
        self.actual_round = 0
        

    # esto hace que no funcione nada :)
    def finalize(self, status: str):
    #    print("me ejecuto")
        pass
    

    