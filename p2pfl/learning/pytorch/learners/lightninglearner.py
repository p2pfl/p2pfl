from torch import tensor
from pytorch_lightning import Trainer
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.pytorch.models.mlp import MLP
from p2pfl.learning.pytorch.utils.logger import FederatedTensorboardLogger
from collections import OrderedDict
import pickle

###########################
#    LightningLearner    #
###########################

class LightningLearner(NodeLearner):

    def __init__(self, model, data, log_name=None):
        self.model = model            
        self.data = data
        self.logger = FederatedTensorboardLogger("training_logs", name=log_name)
        self.trainer = None
        self.epochs = 1

    # Encoded to numpy serialized
    def encode_parameters(self):
        array = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return pickle.dumps(array)

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
        if self.epochs > 0:
            self.trainer = Trainer(max_epochs=self.epochs, accelerator="auto", logger=self.logger, enable_checkpointing=False) 
            self.trainer.fit(self.model, self.data)
            self.trainer = None
        
        #data_ammount = len(self.data.train_dataloader().dataset) #revisarlo
        #return self.get_parameters(), data_ammount, {}

    def evaluate(self, params):
        self.set_parameters(params)

        results = self.trainer.test(self.model, self.data)
        loss = results[0]["test_loss"]
        acc = results[0]["test_acc"]

        data_ammount = len(self.data.test_dataloader().dataset)
        return loss, data_ammount, {"loss": loss, "test_acc": acc}

    def predict(self):
        pass

    def interrupt_fit(self):
        if self.trainer is not None:
            self.trainer.should_stop = True
            self.trainer = None
        else:
            print("No trainer running")