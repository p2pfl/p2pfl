import math
import random
import time
from typing import Dict, List, Union

from p2pfl.commands.vote_train_set_command import VoteTrainSetCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class VoteTrainSetStage(Stage):
    @staticmethod
    def name():
        return "VoteTrainSetStage"

    @staticmethod
    def execute(
        state: NodeState = None,
        communication_protocol: CommunicationProtocol = None,
        **kwargs,
    ) -> Union["Stage", None]:
        if state is None or communication_protocol is None:
            raise Exception("Invalid parameters on VoteTrainSetStage.")

        VoteTrainSetStage.__vote(state, communication_protocol)
        state.train_set = VoteTrainSetStage.__validate_train_set(
            VoteTrainSetStage.__aggregate_votes(state, communication_protocol),
            state,
            communication_protocol,
        )
        logger.info(
            state.addr,
            f"Train set of {len(state.train_set)} nodes: {state.train_set}",
        )

        # Next stage
        if state.addr in state.train_set:
            return StageFactory.get_stage("TrainStage")
        else:
            return StageFactory.get_stage("WaitAggregatedModelsStage")

    def __vote(state: NodeState, communication_protocol: CommunicationProtocol) -> None:
        # Vote (at least itself)
        candidates = list(
            communication_protocol.get_neighbors(only_direct=False).keys()
        )
        if state.addr not in candidates:
            candidates.append(state.addr)
        logger.debug(state.addr, f"{len(candidates)} candidates to train set")

        # Send vote
        samples = min(Settings.TRAIN_SET_SIZE, len(candidates))
        nodes_voted = random.sample(candidates, samples)
        weights = [
            math.floor(random.randint(0, 1000) / (i + 1)) for i in range(samples)
        ]
        votes = list(zip(nodes_voted, weights))

        # Adding votes
        state.train_set_votes_lock.acquire()
        state.train_set_votes[state.addr] = dict(votes)
        state.train_set_votes_lock.release()

        # Send and wait for votes
        logger.info(state.addr, "Sending train set vote.")
        logger.debug(state.addr, f"Self Vote: {votes}")
        communication_protocol.broadcast(
            communication_protocol.build_msg(
                VoteTrainSetCommand.get_name(),
                list(map(str, list(sum(votes, tuple())))),
                round=state.round,
            )
        )

    def __aggregate_votes(
        state: NodeState, communication_protocol: CommunicationProtocol
    ) -> List[str]:
        logger.debug(state.addr, "Waiting other node votes.")

        # Get time
        count = 0.0
        begin = time.time()

        while True:
            # If the trainning has been interrupted, stop waiting
            if state.round is None:
                logger.info(state.addr, "Stopping on_round_finished process.")
                return []

            # Update time counters (timeout)
            count = count + (time.time() - begin)
            begin = time.time()
            timeout = count > Settings.VOTE_TIMEOUT

            # Clear non candidate votes
            state.train_set_votes_lock.acquire()
            nc_votes = {
                k: v
                for k, v in state.train_set_votes.items()
                if k
                in list(communication_protocol.get_neighbors(only_direct=False).keys())
                or k == state.addr
            }
            state.train_set_votes_lock.release()

            # Determine if all votes are received
            needed_votes = set(
                list(communication_protocol.get_neighbors(only_direct=False).keys())
                + [state.addr]
            )
            votes_ready = needed_votes == set(nc_votes.keys())

            if votes_ready or timeout:
                if timeout and not votes_ready:
                    logger.info(
                        state.addr,
                        f"Timeout for vote aggregation. Missing votes from {set(list(communication_protocol.get_neighbors(only_direct=False).keys()) + [state.addr]) - set(nc_votes.keys())}",
                    )

                results: Dict[str, int] = {}
                for node_vote in list(nc_votes.values()):
                    for i in range(len(node_vote)):
                        k = list(node_vote.keys())[i]
                        v = list(node_vote.values())[i]
                        if k in results:
                            results[k] += v
                        else:
                            results[k] = v

                # Order by votes and get TOP X
                results_ordered = sorted(
                    results.items(), key=lambda x: x[0], reverse=True
                )  # to equal solve of draw (node name alphabetical order)
                results_ordered = sorted(
                    results_ordered, key=lambda x: x[1], reverse=True
                )
                top = min(len(results_ordered), Settings.TRAIN_SET_SIZE)
                results_ordered = results_ordered[0:top]

                # Clear votes
                state.train_set_votes = {}
                logger.info(state.addr, f"Computed {len(nc_votes)} votes.")
                return [i[0] for i in results_ordered]

            # Wait for votes or refresh every 2 seconds
            state.wait_votes_ready_lock.acquire(timeout=2)

    def __validate_train_set(
        train_set: List[str],
        state: NodeState,
        communication_protocol: CommunicationProtocol,
    ) -> List[str]:
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in train_set:
            if tsn not in list(
                communication_protocol.get_neighbors(only_direct=False).keys()
            ):
                if tsn != state.addr:
                    train_set.remove(tsn)
        return train_set
