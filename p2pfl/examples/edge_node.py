#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

"""
Example of a P2PFL MNIST edge node using a MLP model and a MnistFederatedDM.

This node only starts, create a node2 and connect to it in order to start the federated learning process.
"""

import argparse
import asyncio

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.pytorch.lightning_model import MLP, LightningModel
from p2pfl.nodes.edge_node import EdgeNode


def __get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST edge node using a MLP model and a MnistFederatedDM.")
    parser.add_argument("--addr", type=str, help="The addres of the proxy.", required=True)
    return parser.parse_args()

async def edge_node(addr: str) -> None:
    """
    Start a node1 and waits for a key press to stop it.

    Args:
        port: The port where the node will be listening.

    """
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(10, RandomIIDPartitionStrategy)  # type: ignore
    node = EdgeNode(
        addr,
        LightningModel(MLP()),
        partitions[0],
        learner=LightningLearner
    )
    await node.start()

if __name__ == "__main__":
    args = __get_args()
    asyncio.run(edge_node(args.addr))
