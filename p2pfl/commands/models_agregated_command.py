from p2pfl.commands.command import Command
from p2pfl.management.logger import logger

class ModelsAggregatedCommand(Command):

    def __init__(self, state):
        self.state = state

    @staticmethod
    def get_name() -> str:
        return "models_aggregated"

    def execute(self, source: str, round: int, *args) -> None:
        if round == self.state.round:
            # esto meterlo en estado o agg
            self.state.models_aggregated[source] = list(args)
        else:
            logger.debug(
                self.state.addr,
                f"Models Aggregated message from {source} in a late round. Ignored. {round} != {self.state.round}",
            )
