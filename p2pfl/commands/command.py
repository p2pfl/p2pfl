import abc


class Command(abc.ABC):

    @staticmethod
    def get_name() -> str:
        pass

    @abc.abstractmethod
    def execute(self, source: str, round: int, *args) -> None:
        pass
