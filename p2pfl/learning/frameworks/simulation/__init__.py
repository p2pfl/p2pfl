"""Module for the efficient parallel local simulation of the learning process (based on Ray)."""

from typing import Type

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.utils.check_ray import ray_installed

###
#   Ray
###


def try_init_learner_with_ray(learner: Type[Learner], model: P2PFLModel, data: P2PFLDataset, addr: str, aggregator: Aggregator) -> Learner:
    """
    Create a learner instance.

    Args:
        learner: The learner class.
        model: The model of the learner.
        data: The data of the learner.
        addr: The address of the learner.
        aggregator: The aggregator that we are using.

    """
    learner_instance = learner(model, data, addr, aggregator=aggregator)
    if ray_installed():
        from p2pfl.learning.frameworks.simulation.virtual_learner import VirtualNodeLearner

        learner_instance = VirtualNodeLearner(learner_instance, addr)
    return learner_instance
