from p2pfl.commands.command import Command
from p2pfl.management.logger import logger


class ModelsReadyCommand(Command):
    def __init__(self, state):
        self.state = state

    @staticmethod
    def get_name() -> str:
        return "models_ready"

    def execute(self, source: str, round: int, *args) -> None:
        # revisar validación al igual que en VoteTrainSetCommand
        ########################################################
        # try to improve clarity in message moment check
        ########################################################
        if self.state.round is not None:
            if round in [self.state.round - 1, self.state.round]:
                self.state.nei_status[source] = self.state.round
            else:
                # Ignored
                logger.error(
                    self.state.addr,
                    f"Models ready from {source} in a late round. Ignored. {round} != {self.state.round} / {self.state.round-1}",
                )
        else:
            logger.warning(self.state.addr, "Models ready received when learning is not running")
