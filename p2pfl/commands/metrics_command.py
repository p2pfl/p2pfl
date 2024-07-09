from p2pfl.commands.command import Command
from p2pfl.management.logger import logger


class MetricsCommand(Command):
    def __init__(self, state):
        super().__init__()
        self.state = state

    @staticmethod
    def get_name():
        return "metrics"

    def execute(self, source: str, round: int, *args) -> None:
        logger.info(self.state.addr, f"Metrics received from {source}")
        # process metrics
        for i in range(0, len(args), 2):
            key = args[i]
            value = float(args[i + 1])
            logger.log_metric(source, key, value, round=round)
