from collections import OrderedDict
import pickle
import torch
from pytorch_lightning import Trainer
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.pytorch.logger import FederatedTensorboardLogger
import logging

###########################
#    LightningLearner     #
###########################

class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        log_name: Name of the log.
        epochs: Number of epochs to train.
        logger: Logger.
    """
    def __init__(self, model, data, log_name=None):
        self.model = model            
        self.data = data
        self.log_name = log_name
        self.logger = None
        self.__trainer = None
        self.epochs = 1
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data
        
    def encode_parameters(self, params=None, contributors=None, weight=None):
        if params is None:
            params = self.model.state_dict()
        array = [val.cpu().numpy() for _, val in params.items()]
        return pickle.dumps((array,contributors,weight))

    def decode_parameters(self, data):
        try:
            params, contributors, weight = pickle.loads(data)
            params_dict = zip(self.model.state_dict().keys(), params)
            return (OrderedDict({k: torch.tensor(v) for k, v in params_dict}), contributors, weight)
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
        try:
            if self.epochs > 0:
                self.__trainer = Trainer(max_epochs=self.epochs, accelerator="auto", logger=self.logger, enable_checkpointing=False, enable_model_summary=False) 
                self.__trainer.fit(self.model, self.data)
                self.__trainer = None
        except Exception as e:
            logging.error("Something went wrong with pytorch lightning. {}".format(e))

    def interrupt_fit(self):
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None
            
    def evaluate(self):
        try:
            if self.epochs > 0:
                self.__trainer = Trainer(max_epochs=self.epochs, accelerator="auto", logger=None, log_every_n_steps=0, enable_checkpointing=False)
                results = self.__trainer.test(self.model, self.data, verbose=False)
                loss = results[0]["test_loss"]
                metric = results[0]["test_metric"]
                self.__trainer = None
                self.log_validation_metrics(loss, metric)
                return loss,metric
            else:
                return None
        except Exception as e:
            logging.error("Something went wrong with pytorch lightning. {}".format(e))
            return None

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        self.logger.log_scalar("test_loss", loss, round,name=name)
        self.logger.log_scalar("test_metric", metric, round,name=name)

    def get_num_samples(self):
        return (len(self.data.train_dataloader().dataset), len(self.data.test_dataloader().dataset))

    def init(self):
        self.close()
        self.logger = FederatedTensorboardLogger("training_logs", name=self.log_name)
   
    def close(self):
        if self.logger is not None:
            self.logger.close()

    def finalize_round(self):
        self.logger.finalize_round()