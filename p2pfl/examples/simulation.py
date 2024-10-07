import argparse

from p2pfl.utils import (
    wait_convergence,
    set_test_settings,
    wait_4_results,
)
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
from p2pfl.management.logger.logger import logger
import time
import matplotlib.pyplot as plt

from p2pfl.simulation.simulation import *

parser = argparse.ArgumentParser(description="Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)

def wait_convergence(nodes, n_neis, wait=5, only_direct=False):
    acum = 0
    while True:
        begin = time.time()
        if all(
            [len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes]
        ):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > wait:
            assert False

def test_convergence(n, r, epochs=2):
    start_time = time.time()
    # Node Creation
    model = MLP()
    data = MnistFederatedDM(sub_id=0, number_sub=20)

    start_simulation(model, data, n)

    print("--- %s seconds ---" % (time.time() - start_time))
    return


if __name__ == "__main__":
    # Settings
    set_test_settings()
    # Launch experiment
    test_convergence(5, 2, epochs=0)