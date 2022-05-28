from torch import tensor
from pytorch_lightning import Trainer
from p2pfl.learning.learner import NodeLearning
from p2pfl.learning.logger import FederatedTensorboardLogger
from p2pfl.learning.pytorch.models.mlp import MLP
from collections import OrderedDict
import pickle

###########################
#    LightningLearning    #
###########################

class LightningLearning(NodeLearning):

    def __init__(self, data, log_name=None, model=None):
        self.model = MLP()
        # Loads Weights
        if model is not None:
            print("Not Impemented Yet") 
                   
        self.data = data
        self.log_name =log_name
        self.trainer = None
        self.epochs = 1 # recordar parametrizar epochs
        if log_name is None:
            self.logger = FederatedTensorboardLogger("training_logs")
        else:
            self.logger = FederatedTensorboardLogger("training_logs", name=self.log_name)
            

    def encode_parameters(self):
        array = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return pickle.dumps(array)

    #meter un test para comprobar encode y decode

    #agregar la validación del modelo (que sea un stete dict con x keys y de x longitud)

    def decode_parameters(self, data):
        params = pickle.loads(data)
        params_dict = zip(self.model.state_dict().keys(), params)
        return OrderedDict({k: tensor(v) for k, v in params_dict})

    def set_parameters(self, params):
        self.model.load_state_dict(params)

    def get_parameters(self):
        return self.model.state_dict()

    def set_epochs(self, epochs):
        self.epochs = epochs

    def fit(self):
        self.trainer = Trainer(max_epochs=self.epochs, accelerator="auto", logger=self.logger, enable_checkpointing=False) 
        self.trainer.fit(self.model, self.data)
        self.trainer = None
        #data_ammount = len(self.data.train_dataloader().dataset) #revisarlo
        #return self.get_parameters(), data_ammount, {}

    def interrupt_fit(self):
        if self.trainer is not None:
            self.trainer.should_stop = True
            self.trainer = None
        else:
            print("No trainer running")

    def evaluate(self, params):
        self.set_parameters(params)

        results = self.trainer.test(self.model, self.data)
        loss = results[0]["test_loss"]
        acc = results[0]["test_acc"]

        data_ammount = len(self.data.test_dataloader().dataset)
        return loss, data_ammount, {"loss": loss, "test_acc": acc}


    def predict(self):
        pass