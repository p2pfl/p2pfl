from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from math import floor
from pytorch_lightning import LightningDataModule
# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#######################################
#    FederatedDataModule for MNIST    #
#######################################

class MnistFederatedDM(LightningDataModule):
    """
    LightningDataModule of partitioned MNIST. Its used to generate **IID** distribucions over MNIS. Toy Problem.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.
    """

    # Singleton
    mnist_train = None
    mnist_val = None

    def __init__(self, sub_id=0, number_sub=1, batch_size=32, num_workers=4, val_percent=0.1, iid=True):
        super().__init__()
        self.sub_id=sub_id
        self.number_sub=number_sub
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.val_percent=val_percent
        
        # Singletons of MNIST train and test datasets
        if MnistFederatedDM.mnist_train is None:
            MnistFederatedDM.mnist_train = MNIST("", train=True, download=True, transform=transforms.ToTensor())
            if not iid:
                sorted_indexes = MnistFederatedDM.mnist_train.targets.sort()[1]
                MnistFederatedDM.mnist_train.targets = MnistFederatedDM.mnist_train.targets[sorted_indexes]
                MnistFederatedDM.mnist_train.data = MnistFederatedDM.mnist_train.data[sorted_indexes]
        if MnistFederatedDM.mnist_val is None:
            MnistFederatedDM.mnist_val = MNIST("", train=False, download=True, transform=transforms.ToTensor())
            if not iid:
                sorted_indexes = MnistFederatedDM.mnist_val.targets.sort()[1]
                MnistFederatedDM.mnist_val.targets = MnistFederatedDM.mnist_val.targets[sorted_indexes]
                MnistFederatedDM.mnist_val.data = MnistFederatedDM.mnist_val.data[sorted_indexes]
        if self.sub_id+1 > self.number_sub:
            raise("Not exist the subset {}".format(self.sub_id))

        # Training / validation set
        trainset = MnistFederatedDM.mnist_train
        rows_by_sub = floor(len(trainset)/self.number_sub)
        tr_subset = Subset(trainset,range(self.sub_id*rows_by_sub,(self.sub_id+1)*rows_by_sub))
        mnist_train, mnist_val = random_split(tr_subset, [round(len(tr_subset)*(1-self.val_percent)), round(len(tr_subset)*self.val_percent)])
        
        # Test set
        testset = MnistFederatedDM.mnist_val
        rows_by_sub = floor(len(testset)/self.number_sub)
        te_subset = Subset(testset,range(self.sub_id*rows_by_sub,(self.sub_id+1)*rows_by_sub))

        if len(testset) < self.number_sub:
            raise("Too much partitions")
        
        #DataLoaders
        self.train_loader = DataLoader(mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader   = DataLoader(mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader  = DataLoader(te_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        print("Train: {} Val:{} Test:{}".format(len(mnist_train),len(mnist_val),len(te_subset)))

    
    def train_dataloader(self):
        """
        """
        return self.train_loader

    def val_dataloader(self):
        """
        """
        return self.val_loader

    def test_dataloader(self):
        """
        """
        return self.test_loader