

####################
# patron plantilla # -> añadir
####################


class NodeLearning:

    def encode_parameters(self): pass

    def decode_parameters(self): pass

    def set_parameters(self, params): pass

    def fit(self): pass

    def evaluate(self): pass

    def predict(self): pass



from p2pfl.learning.mlp import MLP
import torch
from collections import OrderedDict
import pytorch_lightning as pl

class MyNodeLearning(NodeLearning):

    def __init__(self, data):
        self.model = MLP()
        self.data = data
        self.trainer = pl.Trainer(gpus=self.gpus)


    def encode_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def decode_parameters(self, params):
        params_dict = zip(self.model.state_dict().keys(), params)
        return OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    def set_parameters(self, params):
        self.model.load_state_dict(self.decode_parameters(params))

    def fit(self, parameters):
        self.set_parameters(parameters)
        self.trainer.fit(self.model, self.data)

        data_ammount = len(self.data.train_dataloader().dataset) #revisarlo
        return self.get_parameters(), data_ammount, {}

    def evaluate(self, params):
        self.set_parameters(params)

        results = self.trainer.test(self.model, self.data)
        loss = results[0]["test_loss"]
        acc = results[0]["test_acc"]

        data_ammount = len(self.data.test_dataloader().dataset)
        return loss, data_ammount, {"loss": loss, "test_acc": acc}


    def predict(self):
        pass