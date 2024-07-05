from p2pfl.commands.command import Command
from p2pfl.management.logger import logger


"""
hacer un test para esto, revisar coverage tests
"""


class StopLearningCommand(Command):

    def __init__(self, state, aggregator) -> None:
        self.state = state
        self.aggregator = aggregator

    @staticmethod
    def get_name() -> str:
        return "stop_learning"

    def execute(self, source: str, round: int) -> None:
        logger.info(self.state.addr, "Stopping learning")
        # Leraner
        self.state.learner.interrupt_fit()
        self.state.learner = None
        # Aggregator
        self.aggregator.clear()
        # State
        self.state.clear()
        logger.experiment_finished(self.state.addr)
        # Try to free wait locks
        try:
            self.state.wait_votes_ready_lock.release()
        except Exception:
            pass
