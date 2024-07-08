from typing import List

from p2pfl.commands.command import Command
from p2pfl.commands.models_agregated_command import ModelsAggregatedCommand
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger

"""
revisar el tema de parado de nodos: importante enviar que es lo que falló cacheando el error (haría un re-raise)

- diversificar agregación de modelos en diferentes tareas (por ejemplo init o demás): simplicidad y organización
    - __add_model_aggregator
    - __initialize_model
"""


class AddModelCommand(Command):
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
        return "add_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: List[str],
        weight: int,
    ) -> None:
        # Check if Learning is running
        if self.state.round is not None:
            # Check source
            if round != self.state.round:
                logger.error(
                    self.state.addr,
                    f"Model Reception in a late round ({round} != {self.state.round}).",
                )
                return

            # Check moment (not init and invalid round)
            if len(self.state.train_set) == 0:
                logger.error(
                    self.state.addr, "Model Reception when there is no trainset"
                )
                return

            try:
                # Add model to aggregator
                models_added = self.aggregator.add_model(
                    self.state.learner.decode_parameters(weights),
                    list(contributors),
                    weight,
                )
                if models_added != []:
                    # Communicate Aggregation
                    self.communication_protocol.broadcast(
                        self.communication_protocol.build_msg(
                            ModelsAggregatedCommand.get_name(),
                            models_added,
                            round=self.state.round,
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
