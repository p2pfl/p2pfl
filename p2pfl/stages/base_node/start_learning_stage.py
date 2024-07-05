import time
from typing import Any, List, Union
from p2pfl.learning.learner import NodeLearner
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.commands.init_model_command import InitModelCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.stages.stage_factory import StageFactory


class StartLearningStage(Stage):

    @staticmethod
    def name():
        return "StartLearningStage"
    
    @staticmethod
    def execute(
        rounds: int = None,
        epochs: int = None,
        state: NodeState = None,
        learner_class: NodeLearner = None,
        communication_protocol: CommunicationProtocol = None,
        model: Any = None,
        data: Any = None,
        **kwargs
    ) -> Union["Stage", None]:
        if (
            rounds is None
            or epochs is None
            or state is None
            or learner_class is None
            or model is None
            or data is None
            or communication_protocol is None
        ):
            raise Exception("Invalid parameters on StartLearningStage.")

        state.start_thread_lock.acquire()  # Used to avoid create duplicated training threads
        if state.round is None:
            # Init
            state.set_experiment("experiment", rounds)
            logger.experiment_started(state.addr)
            state.learner = learner_class(model, data, state.addr, epochs)
            state.start_thread_lock.release()
            begin = time.time()

            # Wait and gossip model inicialization
            logger.info(state.addr, "Waiting initialization.")
            state.model_initialized_lock.acquire()
            logger.info(state.addr, "Gossiping model initialization.")
            StartLearningStage.__gossip_model(initialization=True)

            # Wait to guarantee new connection heartbeats convergence
            wait_time = Settings.WAIT_HEARTBEATS_CONVERGENCE - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)

            # Vote
            return StageFactory.get_stage("VoteTrainSetStage")
        
        else:
            state.start_thread_lock.release()
            return None

    def __gossip_model(self, state, communication_protocol, aggregator) -> None:
        def early_stopping_fn():
            return state.round is None

        # Wait a model (init or aggregated)
        def candidate_condition(node: str) -> bool:
            return node not in state.nei_status.keys()

        def get_candidates_fn() -> List[str]:
            return [
                n
                for n in communication_protocol.get_neighbors(only_direct=True)
                if candidate_condition(n)
            ]

        def status_fn() -> Any:
            return get_candidates_fn()

        def model_fn(_: str) -> Any:
            model = state.learner.get_parameters()
            contributors = aggregator.get_aggregated_models()  # Poner a NONE
            weight = 1  # Poner a NONE
            encoded_model = state.learner.encode_parameters(params=model)
            return communication_protocol.build_weights(
                InitModelCommand.get_name(),
                state.round,
                encoded_model,
                contributors,
                weight,
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
        )
