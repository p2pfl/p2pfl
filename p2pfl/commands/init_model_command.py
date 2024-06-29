from typing import List
from p2pfl.commands.command import Command
from p2pfl.commands.model_initialized_command import ModelInitializedCommand
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger


"""
revisar el tema de parado de nodos: importante enviar que es lo que falló cacheando el error (haría un re-raise)

- diversificar agregación de modelos en diferentes tareas (por ejemplo init o demás): simplicidad y organización
    - __add_model_aggregator
    - __initialize_model
"""


class InitModelCommand(Command):

    def __init__(
        self,
        state,
        stop,
        aggregator,
        comm_proto,
    ) -> None:
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.communication_protocol = comm_proto

    @staticmethod
    def get_name() -> str:
        return "init_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: List[str],
        weight: int,
    ) -> None:

        # Check if Learning is running
        if self.state.learner is not None:
            # Check source
            if round != self.state.round:
                logger.error(
                    self.state.addr,
                    f"Model Reception in a late round ({round} != {self.state.round}).",
                )
                return

            # Check moment (not init and invalid round)
            if not self.state.model_initialized_lock.locked():
                logger.error(
                    self.state.addr,
                    "Model initizalization message when the model is already initialized. Ignored.",
                )
                return

            try:
                model = self.state.learner.decode_parameters(weights)
                self.state.learner.set_parameters(model)
                self.state.model_initialized_lock.release()
                logger.info(self.state.addr, "Model Weights Initialized")
                # Communicate Initialization
                self.communication_protocol.broadcast(
                    self.communication_protocol.build_msg(
                        ModelInitializedCommand.get_name()
                    )
                )

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.state.addr, "Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError:
                logger.error(self.state.addr, "Models not matching.")
                self.stop()

            except Exception as e:
                logger.error(self.state.addr, f"Unknown error adding model: {e}")
                self.stop()

        else:
            logger.debug(
                self.state.addr, "Tried to add a model while learning is not running"
            )
