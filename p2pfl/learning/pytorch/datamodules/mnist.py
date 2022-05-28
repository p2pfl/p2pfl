from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from math import floor
from pytorch_lightning import LightningDataModule

#######################################
#    FederatedDataModule for MNIST    #
#######################################

class MnistFederatedDM(LightningDataModule):
    def __init__(self, sub_id=0, number_sub=1, batch_size=32, num_workers=4, val_percent=0.1):
        super().__init__()
        self.sub_id=sub_id
        self.number_sub=number_sub
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.val_percent=val_percent

    #MNIST
    def setup(self, stage = None): 
        
        # recordarse de añadir que no se pueda dividir m'as de len test

        if self.sub_id+1 > self.number_sub:
            raise("Se exceden la cantidad de subconjuntos")

        # Training / validation set
        trainset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
        rows_by_sub = floor(len(trainset)/self.number_sub)
        tr_subset = Subset(trainset,range(self.sub_id*rows_by_sub,(self.sub_id+1)*rows_by_sub))
        mnist_train, mnist_val = random_split(tr_subset, [round(len(tr_subset)*(1-self.val_percent)), round(len(tr_subset)*self.val_percent)])
        
        # Test set
        testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
        rows_by_sub = floor(len(testset)/self.number_sub)
        te_subset = Subset(testset,range(self.sub_id*rows_by_sub,(self.sub_id+1)*rows_by_sub))
        
        #DataLoaders
        self.train_loader = DataLoader(mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader   = DataLoader(mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader  = DataLoader(te_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        print("Train: {} Val:{} Test:{}".format(len(mnist_train),len(mnist_val),len(te_subset)))

    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader