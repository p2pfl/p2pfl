from p2pfl.commands.command import Command


class ModelInitializedCommand(Command):

    def __init__(self, state):
        super().__init__()
        self.state = state

    @staticmethod
    def get_name() -> str:
        return "model_initialized"

    def execute(self, source: str, round: int) -> None:
        self.state.nei_status[source] = -1
