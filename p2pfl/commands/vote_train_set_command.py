from p2pfl.commands.command import Command
from p2pfl.management.logger import logger


class VoteTrainSetCommand(Command):

    def __init__(self, state):
        self.state = state

    @staticmethod
    def get_name() -> str:
        return "vote_train_set"

    def execute(self, source: str, round: int, *args) -> None:
        # check moment: round or round + 1 because of node async
        ########################################################
        # try to improve clarity in message moment check
        ########################################################
        if self.state.learner is not None:
            if round in [self.state.round, self.state.round + 1]:
                # build vote dict
                votes = args
                tmp_votes = {}
                for i in range(0, len(votes), 2):
                    tmp_votes[votes[i]] = int(votes[i + 1])
                # set votes
                self.state.train_set_votes_lock.acquire()
                self.state.train_set_votes[source] = tmp_votes
                self.state.train_set_votes_lock.release()
                # Communicate to the training process that a vote has been received
                try:
                    self.state.wait_votes_ready_lock.release()
                except Exception:
                    pass
            else:
                logger.error(
                    self.state.addr,
                    f"Vote received in a late round. Ignored. {round} != {self.state.round} / {self.state.round+1}",
                )
        else:
            logger.error(self.state.addr, "Vote received when learning is not running")
