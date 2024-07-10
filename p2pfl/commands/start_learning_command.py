from p2pfl.commands.command import Command


class StartLearningCommand(Command):
    def __init__(self, start_learning_fn):
        super().__init__()
        self.__learning_fn = start_learning_fn

    @staticmethod
    def get_name() -> str:
        return "start_learning"

    def execute(self, source: str, round: int, learning_rounds, learning_epochs) -> None:
        # Start learning thread
        self.__learning_fn(int(learning_rounds), int(learning_epochs))
