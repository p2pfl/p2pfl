#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from collections import defaultdict
import random
from p2pfl.node import Node
import os
import json
import time
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, random_split
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.settings import Settings
import sys


def set_settings():
    Settings.BLOCK_SIZE = 10240
    Settings.NODE_TIMEOUT = 300
    Settings.VOTE_TIMEOUT = 1200
    Settings.AGGREGATION_TIMEOUT = 300
    Settings.HEARTBEAT_PERIOD = 60
    Settings.HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD = 2
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 10
    Settings.TRAIN_SET_SIZE = 4
    Settings.TRAIN_SET_CONNECT_TIMEOUT = 5
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 500
    Settings.GOSSIP_MESSAGES_FREC = 1
    Settings.GOSSIP_MESSAGES_PER_ROUND = 10
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 30
    Settings.GOSSIP_MODELS_FREC = 10
    Settings.GOSSIP_MODELS_PER_ROUND = 10


##################################
#    Datamodules and Datasets    #
##################################


class JSON_MNISTDataModule(LightningDataModule):
    def __init__(
        self, train_dataset, test_dataset, batch_size=32, num_workers=4, val_percent=0.1
    ):
        super().__init__()
        train_samples = round(len(train_dataset) * (1 - val_percent))
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_samples, len(train_dataset) - train_samples]
        )
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class JSON_MNIST(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples)
        self.samples = torch.reshape(
            self.samples, (-1, 1, 28, 28)
        )  # channels, width, height

        self.labels = torch.tensor(labels)
        assert len(self.samples) == len(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


#############################
#    Load data functions    #
#############################


def build_datamodules(data_dir):
    train_data_dir = os.path.join(data_dir, "data", "train")
    test_data_dir = os.path.join(data_dir, "data", "test")

    # recordar cargarse groups
    users, _, train_data, test_data = read_data(train_data_dir, test_data_dir)

    result = []
    for u in users:
        result.append(
            JSON_MNISTDataModule(
                JSON_MNIST(train_data[u]["x"], train_data[u]["y"]),
                JSON_MNIST(test_data[u]["x"], test_data[u]["y"]),
            )
        )
    return result


def build_big_dataset(data_dir):
    train_data_dir = os.path.join(data_dir, "data", "train")
    test_data_dir = os.path.join(data_dir, "data", "test")

    users, _, train_data, test_data = read_data(train_data_dir, test_data_dir)

    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    for u in random.sample(users, 80):
        train_samples.extend(train_data[u]["x"])
        train_labels.extend(train_data[u]["y"])
        test_samples.extend(test_data[u]["x"])
        test_labels.extend(test_data[u]["y"])
    return JSON_MNISTDataModule(
        JSON_MNIST(train_samples, train_labels), JSON_MNIST(test_samples, test_labels)
    )


def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])

        data.update(cdata["user_data"])

    clients = list(sorted(data.keys()))
    return clients, [], data


def federated_train(femnist_path):
    print("Loading data...")
    datamodules = build_datamodules(femnist_path)
    print("Data Loaded ({} clients)".format(len(datamodules)))

    nodes = [Node(CNN(out_channels=62), dm) for dm in random.sample(datamodules, 40)]
    [n.start() for n in nodes]

    # Connect in a ring
    for i in range(len(nodes)):
        h, p = nodes[(i + 1) % len(nodes)].get_addr()
        nodes[i].connect_to(h, p)

    # Connect random nodes -> increase gossip convergence
    for n in random.sample(nodes, round(len(nodes) / 2)):
        # pick a random node from nodes
        random_dirs = [n.get_addr() for n in random.sample(nodes, 1)]
        for h, p in random_dirs:
            n.connect_to(h, p)

    # Compute train dataloader num samples mean
    train_num_samples = [len(dm.train_dataset) for dm in datamodules]
    train_num_samples_mean = sum(train_num_samples) / len(train_num_samples)

    print(
        "Creados {} nodos con una media de {} muestras en el conjunto de entrenamiento.".format(
            len(nodes), train_num_samples_mean
        )
    )

    nodes[0].set_start_learning(rounds=120, epochs=2)
    time.sleep(1)

    while True:
        time.sleep(10)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break


def centralized_train(femnist_path):
    data = build_big_dataset(femnist_path)
    print(
        "{} muestras en el conjunto de entrenamiento.".format(len(data.train_dataset))
    )

    n = Node(CNN(out_channels=62), data)
    n.start()
    n.set_start_learning(rounds=1, epochs=1)

    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in [n]]:
            finish = finish and f

        if finish:
            break

    for node in [n]:
        node.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} <femnist_path>".format(sys.argv[0]))
        sys.exit(1)

    set_settings()
    femnist_path = sys.argv[1]
    federated_train(femnist_path)
    centralized_train(femnist_path)
