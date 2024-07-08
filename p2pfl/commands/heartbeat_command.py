from p2pfl.commands.command import Command
from p2pfl.communication.grpc.heartbeater import Heartbeater, heartbeater_cmd_name


class HeartbeatCommand(Command):
    def __init__(self, heartbeat: Heartbeater) -> None:
        self.__heartbeat = heartbeat

    @staticmethod
    def get_name() -> str:
        return heartbeater_cmd_name

    def execute(self, source: str, round: int, time: str) -> None:
        self.__heartbeat.beat(source, time=float(time))
