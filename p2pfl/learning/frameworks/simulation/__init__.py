"""Module for the efficient parallel local simulation of the learning process (based on Ray)."""

from p2pfl.learning.frameworks.learner import Learner
from p2pfl.utils.check_ray import ray_installed

###
#   Ray
###


def try_init_learner_with_ray(learner: Learner) -> Learner:
    """
    Create a learner instance.

    Args:
        learner: The learner to wrap.

    """
    if ray_installed():
        from p2pfl.learning.frameworks.simulation.virtual_learner import VirtualNodeLearner

        learner = VirtualNodeLearner(learner)
    return learner
