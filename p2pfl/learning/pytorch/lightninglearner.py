import torch
from pytorch_lightning import Trainer
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.pytorch.logger import FederatedTensorboardLogger
from collections import OrderedDict
import pickle

###########################
#    LightningLearner     #
###########################

# Agregue metodo close() para cerrar el logger y reset() para resetear el logger

class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.
    """

    def __init__(self, model, data, log_name=None):
        self.model = model            
        self.data = data
        self.log_name = log_name
        self.logger = None
        self.trainer = None
        self.epochs = 1

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data
        
    # Encoded to numpy serialized
    def encode_parameters(self):
        array = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return pickle.dumps(array)

    def decode_parameters(self, data):
        try:
            params = pickle.loads(data)
            params_dict = zip(self.model.state_dict().keys(), params)
            return OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        except:
            raise DecodingParamsError("Error decoding parameters")

    def check_parameters(self, params):
        # Check ordered dict keys
        if set(params.keys()) != set(self.model.state_dict().keys()):
            return False

        # Check tensor shapes
        for key, value in params.items():
            if value.shape != self.model.state_dict()[key].shape:
                return False

        return True

    def set_parameters(self, params):
        try:
            self.model.load_state_dict(params)
        except:
            raise ModelNotMatchingError("Not matching models")

    def get_parameters(self):
        return self.model.state_dict()

    def set_epochs(self, epochs):
        self.epochs = epochs

    def fit(self):
        if self.epochs > 0:
            self.trainer = Trainer(max_epochs=self.epochs, accelerator="auto", logger=self.logger, enable_checkpointing=False) 
            self.trainer.fit(self.model, self.data)
            self.trainer = None
            self.logger.finalize_round()

    def evaluate(self):
        self.trainer = Trainer(max_epochs=self.epochs, accelerator="auto", logger=None, enable_checkpointing=False) 
        results = self.trainer.test(self.model, self.data, verbose=False)
        loss = results[0]["test_loss"]
        metric = results[0]["test_metric"]
        self.trainer = None
        self.log_validation_metrics(loss, metric)

        return loss,metric

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        self.logger.log_scalar("test_loss", loss, round,name=name)
        self.logger.log_scalar("test_metric", metric, round,name=name)

    def interrupt_fit(self):
        if self.trainer is not None:
            self.trainer.should_stop = True
            self.trainer = None

    def get_num_samples(self):
        return (len(self.data.train_dataloader().dataset), len(self.data.test_dataloader().dataset))

    def init(self):
        self.close()
        self.logger = FederatedTensorboardLogger("training_logs", name=self.log_name)
   
    def close(self):
        if self.logger is not None:
            self.logger.close()
