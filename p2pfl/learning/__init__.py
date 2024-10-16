"""
Package that contains all the logic and classes to manage the learning process in P2PFL.

Here basically you can find the different aggregation algorithms and the learners that allows integration with
different machine learning libraries.
"""

import importlib

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel


def create_learner(learner: type[NodeLearner],
                   model: P2PFLModel,
                   data: P2PFLDataset,
                   addr: str):
    """
    Create a learner instance.

    Args:
        learner: The learner class.
        model: The model of the learner.
        data: The data of the learner.
        addr: The address of the learner.

    Returns:
        The learner instance.

    """
    learner_instance = learner(model, data, addr)
    if importlib.util.find_spec("ray") is not None:
        from p2pfl.simulation.virtual_learner import VirtualNodeLearner
        learner_instance = VirtualNodeLearner(learner_instance, addr)
    return learner_instance
