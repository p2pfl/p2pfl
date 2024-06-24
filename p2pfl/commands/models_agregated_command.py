from p2pfl.commands.command import Command


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
