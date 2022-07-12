from collections import defaultdict
import os
import json
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
import numpy
import torch
from torch.utils.data import DataLoader, Subset, random_split

class JSON_MNISTDataModule(LightningDataModule):
    def __init__(self, train_dataset, test_dataset, batch_size=32, num_workers=4, val_percent=0.1):
        super().__init__()
        train_samples = round(len(train_dataset)*(1-val_percent))
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_samples, len(train_dataset) - train_samples])
        self.test_dataset = test_dataset
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
class JSON_MNIST(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, samples, labels):
        """
        Args:
            samples (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = torch.tensor(samples) 
        self.samples = torch.reshape(self.samples, (-1, 1, 28, 28)) # channels, width, height

        self.labels = torch.tensor(labels)
        assert len(self.samples)==len(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx): 
        # adding a channel (b/w) 
        return self.samples[idx],self.labels[idx]
        

    
def get_samples(data_dir):
    train_data_dir = os.path.join(data_dir, 'data', 'train')
    test_data_dir = os.path.join(data_dir, 'data', 'test')

    # recordar cargarse groups
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    
    result = []
    for u in users:
        return train_data[u]["x"]
    

def build_datamodules(data_dir):
    train_data_dir = os.path.join(data_dir, 'data', 'train')
    test_data_dir = os.path.join(data_dir, 'data', 'test')

    # recordar cargarse groups
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    
    result = []
    for u in users:
        result.append( 
            JSON_MNISTDataModule( 
                JSON_MNIST(train_data[u]["x"],train_data[u]["y"]),
                JSON_MNIST(test_data[u]["x"],test_data[u]["y"])
            )
        )
    
    return result
    
    
    
def setup_clients(data_dir, model=None, use_val_set=False):
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join(data_dir, 'data', 'train')
    test_data_dir = os.path.join(data_dir, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    return users, groups, train_data, test_data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data



def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])

        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, [], data


from pytorch_lightning import Trainer, LightningModule
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

class MLP(LightningModule):
    """
    Multilayer Perceptron (MLP) to solve MNIST with PyTorch Lightning.
    """

    def __init__(self, metric = Accuracy, lr_rate=0.001): # low lr to avoid overfitting
        
        # Set seed for reproducibility iniciialization
        seed = 666
        torch.manual_seed(seed)

        super().__init__()
        self.lr_rate = lr_rate
        self.metric = metric()

        # 10 clases
        self.l1 = torch.nn.Linear(28 * 28, 128)
        self.l2 = torch.nn.Linear(128, 256)
        self.l3 = torch.nn.Linear(256, 62)

    def forward(self, x):
        """
        """
        batch_size, channels, width, height = x.size()
        
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        """
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_id):
        """
        """
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y) 
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, prog_bar=True)    
        return loss

    def test_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y) 
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
        return loss
        

datamodules = build_datamodules("/home/pedro/Downloads/leaf/data/femnist")
tr = Trainer(max_epochs=10, accelerator="auto", logger=None, enable_checkpointing=False, enable_model_summary=False) 
tr.fit(MLP(), datamodules[55])