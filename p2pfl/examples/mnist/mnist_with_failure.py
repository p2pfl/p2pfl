#!/usr/bin/env python3
"""
MNIST example with node failure simulation.

This script runs a MNIST federated learning experiment where specific nodes
fail and recover at predetermined rounds:
- Nodes fail after failure_round completes
- When node shortage is detected, wait 10 seconds for more nodes, then proceed with available nodes
- Nodes recover at recovery_round and restore from checkpoint
"""

import argparse
import random
import threading
import time
import uuid

import matplotlib.pyplot as plt

from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_convergence, wait_to_finish
from p2pfl.checkpoints import rejoin


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment with node failure simulation.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=4)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=10)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--token", type=str, help="The API token for the Web Logger.", default="")
    parser.add_argument("--protocol", type=str, help="The protocol to use.", default="memory", choices=["grpc", "unix", "memory"])
    parser.add_argument("--framework", type=str, help="The framework to use.", default="pytorch", choices=["pytorch", "tensorflow", "flax"])
    parser.add_argument("--aggregator", type=str, help="The aggregator to use.", default="fedavg", choices=["fedavg", "scaffold"])
    parser.add_argument("--profiling", action="store_true", help="Enable profiling.", default=False)
    parser.add_argument("--reduced_dataset", action="store_true", help="Use a reduced dataset just for testing.", default=False)
    parser.add_argument("--use_scaffold", action="store_true", help="Use the Scaffold aggregator.", default=False)
    parser.add_argument("--seed", type=int, help="The seed to use.", default=666)
    parser.add_argument("--batch_size", type=int, help="The batch size for training.", default=128)
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="line",
        help="The network topology (star, full, line, ring).",
    )
    parser.add_argument(
        "--failure_round",
        type=int,
        default=2,
        help="Round after which nodes should fail (default: 2).",
    )
    parser.add_argument(
        "--recovery_round",
        type=int,
        default=4,
        help="Round after which nodes should recover (default: 4).",
    )
    parser.add_argument(
        "--failure_nodes",
        type=str,
        nargs="+",
        default=["node-0"],
        help="List of node addresses to fail (default: node-0).",
    )
    args = parser.parse_args()
    # parse topology to TopologyType enum
    args.topology = TopologyType(args.topology)

    return args


def monitor_rounds_and_control_nodes(
    nodes: list[Node],
    failure_nodes: list[str],
    failure_round: int,
    recovery_round: int,
    adjacency_matrix: list[list[int]],
    stop_event: threading.Event,
    node_creation_params: dict,
    protocol_type: str,
    aggregator_type: str,
    model_fn,
) -> None:
    """
    Monitor training rounds and control node failures/recoveries.
    
    Logic:
    1. Wait for all nodes to complete round (failure_round - 1), then fail nodes before round failure_round starts
    2. Monitor for "not enough nodes" errors: If detected, wait 10 seconds for more nodes,
       then proceed with available nodes for voting and training
    3. Wait for all nodes to complete round (recovery_round - 1), then recover nodes before round recovery_round starts
    """
    failed_node_info: list[dict] = []
    failure_triggered = False
    recovery_triggered = False
    node_shortage_detected = False
    node_shortage_wait_start = None

    while not stop_event.is_set():
        try:
            # Check current round and active nodes
            node_rounds = {}  # node_addr -> round
            active_nodes = []
            for node in nodes:
                try:
                    if hasattr(node, "state") and node.state is not None:
                        round_val = node.state.round
                        if round_val is not None:
                            try:
                                node.get_neighbors(only_direct=True)
                                node_rounds[node.addr] = round_val
                                active_nodes.append(node.addr)
                            except Exception:
                                pass
                except Exception:
                    pass

            if not node_rounds:
                time.sleep(0.5)
                continue

            # Get all active node rounds (excluding failed nodes)
            active_node_rounds = [r for addr, r in node_rounds.items() if addr not in failure_nodes]
            
            if not active_node_rounds:
                time.sleep(0.5)
                continue

            min_round = min(active_node_rounds)
            max_round = max(active_node_rounds)
            
            # Debug logging for critical rounds
            if min_round in [failure_round - 1, failure_round, recovery_round - 1, recovery_round]:
                logger.debug("", f"🔍 Monitoring: min_round={min_round}, max_round={max_round}, active_nodes={active_nodes}, failure_triggered={failure_triggered}, recovery_triggered={recovery_triggered}")
            
            # STEP 1: Handle node failure BEFORE failure_round starts
            # Wait for all active nodes to complete round (failure_round - 1), then fail nodes before round failure_round starts
            if not failure_triggered:
                # We need ALL nodes (including those that will fail) to have completed round (failure_round - 1)
                # Get rounds for ALL nodes (not excluding failed nodes yet, since they haven't failed yet)
                all_node_rounds = [r for r in node_rounds.values()]
                
                # Only proceed if we have rounds from ALL nodes (including those that will fail)
                if len(all_node_rounds) >= len(nodes):
                    # Check if ALL nodes (including those that will fail) have completed round (failure_round - 1)
                    # This means all nodes are at round >= failure_round
                    all_completed_prev_round = all(r >= failure_round for r in all_node_rounds)
                    all_started_training = all(r >= 1 for r in all_node_rounds)  # Ensure training has started
                    
                    # Additional check: make sure we're not in the middle of round 1
                    # If any node is still at round 1, don't fail yet
                    if all_completed_prev_round and all_started_training and min(all_node_rounds) >= failure_round:
                        logger.info("", f"💥💥💥 All nodes completed round {failure_round - 1}. Failing nodes {failure_nodes} before round {failure_round} starts 💥💥💥")
                        logger.info("", f"🔍 Debug: all_node_rounds={all_node_rounds}, active_node_rounds={active_node_rounds}")
                        failure_triggered = True
                        
                        # Disconnect failed nodes from all other nodes
                        for node_addr in failure_nodes:
                            for other_node in nodes:
                                if other_node.addr != node_addr:
                                    try:
                                        neighbors = other_node.get_neighbors(only_direct=False)
                                        if isinstance(neighbors, (dict, list)) and node_addr in neighbors:
                                            other_node.disconnect(node_addr)
                                            logger.debug("", f"Disconnected {node_addr} from {other_node.addr}")
                                    except Exception:
                                        pass
                        
                        time.sleep(0.5)  # Wait for disconnection to propagate
                        
                        # Stop failed nodes and store info for recovery
                        for node_addr in failure_nodes:
                            for idx, node in enumerate(nodes):
                                if node.addr == node_addr:
                                    try:
                                        try:
                                            node.get_neighbors(only_direct=True)
                                        except Exception:
                                            logger.debug("", f"Node {node_addr} already stopped")
                                            break
                                        
                                        logger.info("", f"🛑 Stopping node {node_addr}...")
                                        
                                        # First stop the learning workflow to avoid race conditions
                                        # Set round to None to trigger early stop in the workflow
                                        if node.state.experiment is not None:
                                            node.state.experiment.round = None
                                        
                                        # Wait a bit for the workflow to stop
                                        time.sleep(0.5)  # Reduced from 1s to 0.5s for faster failure
                                        
                                        # Now stop the node (which stops the protocol)
                                        node.stop()
                                        time.sleep(0.2)
                                        
                                        # Store node info for recovery
                                        node_params = node_creation_params.get(node_addr, {})
                                        failed_node_info.append({
                                            "node": node,
                                            "idx": idx,
                                            "params": node_params,
                                        })
                                        logger.info("", f"✅ Node {node_addr} stopped and removed from available nodes.")
                                    except Exception as e:
                                        logger.error("", f"❌ Failed to stop node {node_addr}: {e}")
                                    break
                        
                        # CRITICAL: Update train_set for all active nodes to remove failed nodes
                        # This ensures aggregation doesn't wait for models from failed nodes
                        for active_node in nodes:
                            if active_node.addr not in failure_nodes:
                                try:
                                    # Remove failed nodes from train_set
                                    if hasattr(active_node.state, 'train_set') and active_node.state.train_set:
                                        original_train_set = active_node.state.train_set.copy()
                                        active_node.state.train_set = [
                                            n for n in active_node.state.train_set 
                                            if n not in failure_nodes
                                        ]
                                        if active_node.state.train_set != original_train_set:
                                            logger.info("", f"📝 Updated train_set for {active_node.addr}: removed {failure_nodes}, new train_set: {active_node.state.train_set}")
                                    
                                    # Update aggregator's train_set if it's already set and aggregation is not running
                                    if hasattr(active_node, 'aggregator') and active_node.aggregator:
                                        # Only update if aggregation is not currently running
                                        # Check if aggregation event is set (meaning aggregation is finished/not started)
                                        if hasattr(active_node.aggregator, '_finish_aggregation_event'):
                                            if active_node.aggregator._finish_aggregation_event.is_set():
                                                # Aggregation is not running, safe to update
                                                if hasattr(active_node.state, 'train_set') and active_node.state.train_set:
                                                    try:
                                                        active_node.aggregator.set_nodes_to_aggregate(active_node.state.train_set)
                                                        logger.debug("", f"Updated aggregator train_set for {active_node.addr}")
                                                    except Exception as e:
                                                        logger.debug("", f"Could not update aggregator train_set for {active_node.addr}: {e}")
                                            else:
                                                # Aggregation is running, don't update aggregator train_set
                                                # The state.train_set is already updated, so next round will use the correct train_set
                                                logger.debug("", f"Aggregation is running for {active_node.addr}, skipping aggregator train_set update. Will use updated train_set in next round.")
                                except Exception as e:
                                    logger.warning("", f"⚠️ Failed to update train_set for {active_node.addr}: {e}")

            # STEP 2: Monitor for node shortage after failure
            # Check if we're in a round after failure and detect node shortage
            if failure_triggered and min_round > failure_round and not recovery_triggered:
                # Check if any active node has fewer neighbors than expected
                for node in nodes:
                    if node.addr not in failure_nodes:
                        try:
                            neighbors = node.get_neighbors(only_direct=False)
                            neighbor_count = len(neighbors) if isinstance(neighbors, (dict, list)) else 0
                            expected_count = len([n for n in nodes if n.addr not in failure_nodes]) - 1  # -1 for self
                            
                            # Detect shortage: fewer neighbors than expected
                            if neighbor_count < expected_count:
                                if not node_shortage_detected:
                                    node_shortage_detected = True
                                    node_shortage_wait_start = time.time()
                                    logger.warning("", f"⚠️ Node shortage detected at {node.addr}: {neighbor_count} neighbors, expected {expected_count}. Waiting 10 seconds for more nodes...")
                                break
                        except Exception:
                            pass
                
                # If shortage detected, wait 10 seconds then proceed
                if node_shortage_detected and node_shortage_wait_start:
                    elapsed = time.time() - node_shortage_wait_start
                    if elapsed >= 10.0:
                        logger.info("", f"⏱️ 10 seconds elapsed. Proceeding with available nodes for voting and training.")
                        node_shortage_detected = False
                        node_shortage_wait_start = None
                    # Otherwise continue waiting (will check again in next iteration)

            # STEP 3: Handle node recovery BEFORE recovery_round starts
            # Wait for all active nodes to complete round (recovery_round - 1), then recover nodes before round recovery_round starts
            if failure_triggered and not recovery_triggered and failed_node_info:
                # Check if all active nodes have completed round (recovery_round - 1)
                # This means all nodes are at round >= recovery_round - 1 (i.e., they've finished round recovery_round - 1)
                # We need at least as many active nodes reporting as we expect (excluding failed nodes)
                expected_active_count = len([n for n in nodes if n.addr not in failure_nodes])
                all_completed_prev_round = len(active_node_rounds) >= expected_active_count and all(r >= recovery_round - 1 for r in active_node_rounds)
                
                # Trigger recovery when all active nodes have completed round (recovery_round - 1)
                # We want to recover before they start round recovery_round
                if all_completed_prev_round and min_round >= recovery_round - 1:
                    logger.info("", f"🔄🔄🔄 All nodes completed round {recovery_round - 1}. Recovering nodes {failure_nodes} before round {recovery_round} starts 🔄🔄🔄")
                    recovery_triggered = True
                    
                    # Recover each failed node
                    for failed_info in failed_node_info:
                        node_addr = failed_info["node"].addr
                        idx = failed_info["idx"]
                        params = failed_info["params"]
                        
                        try:
                            logger.info("", f"🔄 Recovering node {node_addr}...")
                            
                            # CRITICAL: For Memory protocol, remove the original address from registry
                            # so the recovered node can use the same address (seamless replacement)
                            if protocol_type == "memory":
                                from p2pfl.communication.protocols.protobuff.memory.server import AddressCounter
                                AddressCounter().remove(node_addr)
                                logger.info("", f"🔧 Removed {node_addr} from address registry to allow seamless recovery.")
                            
                            # Create new protocol instance
                            if protocol_type == "memory":
                                new_protocol = MemoryCommunicationProtocol()
                            elif protocol_type == "grpc":
                                new_protocol = GrpcCommunicationProtocol()
                            elif protocol_type == "unix":
                                from p2pfl.communication.protocols.protobuff.unix import UnixCommunicationProtocol
                                new_protocol = UnixCommunicationProtocol()
                            else:
                                raise ValueError(f"Unknown protocol type: {protocol_type}")
                            
                            # Create new aggregator instance
                            if aggregator_type == "scaffold":
                                new_aggregator = Scaffold()
                            else:
                                new_aggregator = None
                            
                            # Create new model instance for recovery (checkpoint will restore parameters)
                            new_model = model_fn()
                            
                            # Create new node instance
                            new_node = Node(
                                new_model,
                                params["data"],
                                protocol=new_protocol,
                                addr=params["addr"],
                                aggregator=new_aggregator,
                            )
                            new_node.start()
                            time.sleep(0.2)
                            
                            # Replace old node in nodes list
                            nodes[idx] = new_node
                            logger.info("", f"✅ Node {node_addr} restarted.")

                            # Reconnect to neighbors (only connect to active nodes, not failed ones)
                            for j, neighbor_node in enumerate(nodes):
                                if idx != j and adjacency_matrix[idx][j] == 1:
                                    # Only connect if neighbor is not in failure_nodes (i.e., it's an active node)
                                    if neighbor_node.addr not in failure_nodes:
                                        try:
                                            neighbors = new_node.get_neighbors(only_direct=True)
                                            if isinstance(neighbors, (dict, list)) and neighbor_node.addr not in neighbors:
                                                new_node.connect(neighbor_node.addr)
                                                neighbor_node.connect(new_node.addr)
                                                logger.info("", f"Reconnected {new_node.addr} to {neighbor_node.addr}")
                                        except Exception as e:
                                            logger.warning("", f"Failed to reconnect {new_node.addr} to {neighbor_node.addr}: {e}")

                            # Restore experiment state
                            experiment_restored = False
                            for active_node in nodes:
                                if active_node.addr != new_node.addr and active_node.state.experiment is not None:
                                    exp = active_node.state.experiment
                                    new_node.state.set_experiment(
                                        exp_name=exp.exp_name,
                                        total_rounds=exp.total_rounds,
                                        dataset_name=exp.dataset_name,
                                        model_name=exp.model_name,
                                        aggregator_name=exp.aggregator_name,
                                        framework_name=exp.framework_name,
                                        learning_rate=exp.learning_rate,
                                        batch_size=exp.batch_size,
                                        epochs_per_round=exp.epochs_per_round,
                                    )
                                    experiment_restored = True
                                    logger.info("", f"Restored experiment state for {new_node.addr}")
                                    break

                            if not experiment_restored:
                                logger.warning("", f"Could not restore experiment state for {new_node.addr}")

                            # Recover from checkpoint
                            if new_node.state.experiment is not None:
                                # First set round to failure_round - 1 to load checkpoint from that round
                                checkpoint_round = failure_round - 1
                                new_node.state.experiment.round = checkpoint_round
                                
                                # IMPORTANT: Memory protocol may have changed the address (e.g., node-0 -> node-0_1)
                                # We need to use the original address (node_addr) to find checkpoints
                                # Temporarily set state.addr to original address for checkpoint lookup
                                original_addr_for_checkpoint = node_addr
                                current_addr = new_node.addr
                                
                                # If address changed, temporarily use original address for checkpoint lookup
                                if current_addr != original_addr_for_checkpoint:
                                    logger.info("", f"⚠️ Address changed from {original_addr_for_checkpoint} to {current_addr}. Using {original_addr_for_checkpoint} for checkpoint lookup.")
                                    # Temporarily update state.addr for checkpoint lookup
                                    new_node.state.addr = original_addr_for_checkpoint
                                    new_node.learner.set_addr(original_addr_for_checkpoint)
                                    new_node.aggregator.set_addr(original_addr_for_checkpoint)
                                
                                logger.info("", f"🔄 Calling rejoin() for {new_node.addr} (original: {original_addr_for_checkpoint}) from round {checkpoint_round}...")
                                success = rejoin(
                                    state=new_node.state,
                                    learner=new_node.learner,
                                    communication_protocol=new_node._communication_protocol,
                                    round=checkpoint_round,
                                )
                                
                                # Restore current address after checkpoint lookup
                                if current_addr != original_addr_for_checkpoint:
                                    new_node.state.addr = current_addr
                                    new_node.learner.set_addr(current_addr)
                                    new_node.aggregator.set_addr(current_addr)
                                
                                if success:
                                    logger.info("", f"✅ Node {new_node.addr} recovered from checkpoint (round {checkpoint_round}).")
                                else:
                                    logger.warning("", f"⚠️ Node {new_node.addr} could not recover from checkpoint, using fresh model.")
                                
                                # Update round to recovery_round so node can participate in current round
                                new_node.state.experiment.round = recovery_round
                                
                                # CRITICAL: Manually set train_set for the recovered node
                                # The recovered node needs to be in the train_set to receive models
                                # We'll get the train_set from an active neighbor node
                                recovered_train_set = None
                                for neighbor_node in nodes:
                                    if neighbor_node.addr != new_node.addr and neighbor_node.addr not in failure_nodes:
                                        if hasattr(neighbor_node.state, 'train_set') and neighbor_node.state.train_set:
                                            # Get the train_set from neighbor and add the recovered node
                                            recovered_train_set = neighbor_node.state.train_set.copy()
                                            if new_node.addr not in recovered_train_set:
                                                recovered_train_set.append(new_node.addr)
                                            break
                                
                                if recovered_train_set:
                                    new_node.state.train_set = recovered_train_set
                                    # Also update aggregator's train_set, but only if aggregator is not currently running
                                    if hasattr(new_node.aggregator, 'set_nodes_to_aggregate'):
                                        try:
                                            # Check if aggregator is currently running an aggregation
                                            if hasattr(new_node.aggregator, '_finish_aggregation_event'):
                                                if new_node.aggregator._finish_aggregation_event.is_set():
                                                    # Aggregator is not running, safe to update
                                                    new_node.aggregator.set_nodes_to_aggregate(recovered_train_set)
                                                else:
                                                    # Aggregator is running, skip update (will be updated in next round)
                                                    logger.debug("", f"Aggregator for {new_node.addr} is running, will update train_set in next round")
                                            else:
                                                # Fallback: try to set anyway
                                                new_node.aggregator.set_nodes_to_aggregate(recovered_train_set)
                                        except Exception as e:
                                            logger.warning("", f"Could not set aggregator train_set for {new_node.addr}: {e}")
                                    logger.info("", f"📝 Set train_set for {new_node.addr}: {recovered_train_set}")
                                    
                                    # CRITICAL: Update train_set for ALL active nodes to include the recovered node
                                    # This ensures all nodes know about the recovered node and can send/receive models from it
                                    for active_node in nodes:
                                        if active_node.addr != new_node.addr and active_node.addr not in failure_nodes:
                                            try:
                                                # Update state.train_set
                                                if new_node.addr not in active_node.state.train_set:
                                                    active_node.state.train_set.append(new_node.addr)
                                                    logger.info("", f"📝 Updated train_set for {active_node.addr}: added {new_node.addr}, new train_set: {active_node.state.train_set}")
                                                
                                                # Update aggregator's train_set if it's not currently running
                                                if hasattr(active_node, 'aggregator') and active_node.aggregator:
                                                    if hasattr(active_node.aggregator, '_finish_aggregation_event'):
                                                        if active_node.aggregator._finish_aggregation_event.is_set():
                                                            # Aggregator is not running, safe to update
                                                            try:
                                                                active_node.aggregator.set_nodes_to_aggregate(active_node.state.train_set)
                                                                logger.debug("", f"Updated aggregator train_set for {active_node.addr} to include {new_node.addr}")
                                                            except Exception as e:
                                                                logger.debug("", f"Could not update aggregator train_set for {active_node.addr}: {e}")
                                                        else:
                                                            # Aggregator is running, will update in next round
                                                            logger.debug("", f"Aggregator for {active_node.addr} is running, will update train_set in next round")
                                            except Exception as e:
                                                logger.warning("", f"⚠️ Failed to update train_set for {active_node.addr}: {e}")
                                else:
                                    logger.warning("", f"⚠️ Could not determine train_set for {new_node.addr}. It may not receive models.")
                                
                                # Initialize nei_status to let neighbors know this node is behind and needs models
                                # This will trigger neighbors to send their latest models via gossip
                                neighbors = new_node.get_neighbors(only_direct=False)
                                if isinstance(neighbors, dict):
                                    neighbor_addrs = list(neighbors.keys())
                                elif isinstance(neighbors, list):
                                    neighbor_addrs = neighbors
                                else:
                                    neighbor_addrs = []
                                
                                # Set nei_status to a round before recovery_round so neighbors know to send models
                                for neighbor_addr in neighbor_addrs:
                                    new_node.state.nei_status[neighbor_addr] = recovery_round - 2
                                
                                # CRITICAL: Update neighbors' nei_status to include the recovered node
                                # This ensures neighbors know about the recovered node and can send models to it
                                for neighbor_node in nodes:
                                    if neighbor_node.addr != new_node.addr and neighbor_node.addr in neighbor_addrs:
                                        try:
                                            # Set recovered node's status in neighbor's nei_status
                                            # Use recovery_round - 2 to indicate it's behind and needs models
                                            neighbor_node.state.nei_status[new_node.addr] = recovery_round - 2
                                            logger.debug("", f"Updated {neighbor_node.addr}.nei_status[{new_node.addr}] = {recovery_round - 2}")
                                        except Exception as e:
                                            logger.warning("", f"Failed to update nei_status for {neighbor_node.addr}: {e}")
                                
                                logger.info("", f"📡 Node {new_node.addr} initialized nei_status. Neighbors will send latest models via gossip.")
                                
                                # CRITICAL: Wait a bit for gossip to propagate and ensure neighbors know about the recovered node
                                # This allows neighbors to update their nei_status and start sending models
                                time.sleep(1.0)
                                
                                # CRITICAL: The recovered node needs to participate in the current training round
                                # Since the node has round set to recovery_round, it should be able to receive and process
                                # messages (votes, models) from other nodes. However, it needs to actively participate.
                                # The node will participate by:
                                # 1. Receiving vote messages and responding (VoteTrainSetCommand handles this)
                                # 2. Receiving model messages and aggregating (PartialModelCommand/FullModelCommand handle this)
                                # 3. The workflow will be triggered when the node receives appropriate messages
                                
                                # Ensure the node can process messages by making sure model_initialized_lock is released
                                # This allows the node to receive InitModelCommand and other messages
                                try:
                                    # Release the lock so the node can receive model initialization messages
                                    if new_node.state.model_initialized_lock.locked():
                                        new_node.state.model_initialized_lock.release()
                                    logger.info("", f"🔓 Released model_initialized_lock for {new_node.addr} to receive messages.")
                                except Exception:
                                    pass  # Lock may already be released
                                
                                # CRITICAL: Start the learning workflow for the recovered node
                                # The node needs to actively participate in voting and training
                                # We'll start the workflow in a thread, but it will skip initialization since model is already restored
                                import threading
                                def start_recovered_workflow():
                                    try:
                                        # Wait a bit to ensure all connections are established
                                        time.sleep(0.5)
                                        
                                        # Get experiment parameters from an active node
                                        exp = new_node.state.experiment
                                        if exp:
                                            # Ensure aggregator train_set is set before starting workflow
                                            # This prevents "It is not possible to set nodes to aggregate when the aggregation is running" error
                                            if hasattr(new_node.aggregator, 'set_nodes_to_aggregate') and new_node.state.train_set:
                                                try:
                                                    # Check if aggregator is currently running
                                                    if hasattr(new_node.aggregator, '_finish_aggregation_event'):
                                                        if new_node.aggregator._finish_aggregation_event.is_set():
                                                            new_node.aggregator.set_nodes_to_aggregate(new_node.state.train_set)
                                                    else:
                                                        new_node.aggregator.set_nodes_to_aggregate(new_node.state.train_set)
                                                except Exception as e:
                                                    logger.warning("", f"Could not pre-set aggregator train_set for {new_node.addr}: {e}. Will be set in TrainStage.")
                                            
                                            # Start the learning workflow
                                            # The workflow will start from StartLearningStage, but since model is already initialized,
                                            # it should skip initialization and go directly to voting
                                            new_node.learning_workflow.run(
                                                rounds=exp.total_rounds,
                                                epochs=exp.epochs_per_round,
                                                trainset_size=len(new_node.state.train_set) if new_node.state.train_set else 4,
                                                experiment_name=exp.exp_name,
                                                state=new_node.state,
                                                learner=new_node.learner,
                                                communication_protocol=new_node._communication_protocol,
                                                aggregator=new_node.aggregator,
                                                generator=random.Random(),
                                            )
                                    except Exception as e:
                                        logger.error("", f"Failed to start workflow for {new_node.addr}: {e}")
                                        import traceback
                                        logger.error("", traceback.format_exc())
                                
                                # Start workflow in a separate thread
                                workflow_thread = threading.Thread(target=start_recovered_workflow, daemon=True)
                                workflow_thread.start()
                                logger.info("", f"🚀 Started learning workflow for {new_node.addr} to participate in round {recovery_round} training.")
                            else:
                                logger.warning("", f"⚠️ Node {new_node.addr} has no experiment state.")

                        except Exception as e:
                            logger.error("", f"❌ Failed to recover node {node_addr}: {e}")
                            import traceback
                            logger.error("", f"Traceback: {traceback.format_exc()}")

                    failed_node_info.clear()
                    logger.info("", "✅ All nodes recovered successfully.")

            time.sleep(0.5)

        except Exception as e:
            logger.error("", f"Error in monitoring thread: {e}")
            time.sleep(1)

def mnist_with_failure(
    n: int,
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "memory",
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = False,
    topology: TopologyType = TopologyType.LINE,
    batch_size: int = 128,
    failure_round: int = 2,
    recovery_round: int = 4,
    failure_nodes: list[str] = None,
) -> None:
    """
    P2PFL MNIST experiment with node failure simulation.

    Args:
        n: The number of nodes.
        r: The number of rounds.
        e: The number of epochs.
        show_metrics: Show metrics.
        measure_time: Measure time.
        protocol: The protocol to use.
        framework: The framework to use.
        aggregator: The aggregator to use.
        reduced_dataset: Use a reduced dataset just for testing.
        topology: The network topology (star, full, line, ring).
        batch_size: The batch size for training.
        failure_round: Round after which nodes should fail.
        recovery_round: Round after which nodes should recover.
        failure_nodes: List of node addresses to fail.
    """
    if failure_nodes is None:
        failure_nodes = ["node-0"]

    if measure_time:
        start_time = time.time()

    # Check settings
    if n > Settings.gossip.TTL:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes. Otherwise, some messages will not be delivered."
        )

    # Imports
    if framework == "tensorflow":
        from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn  # type: ignore

        model_fn = model_build_fn  # type: ignore
    elif framework == "pytorch":
        from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn  # type: ignore

        model_fn = model_build_fn  # type: ignore
    else:
        raise ValueError(f"Framework {framework} not added on this example.")

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    data.set_batch_size(batch_size)
    partitions = data.generate_partitions(
        n * 50 if reduced_dataset else n,
        RandomIIDPartitionStrategy,  # type: ignore
    )

    # Node Creation
    nodes = []
    node_creation_params = {}  # Store parameters for node recreation during recovery
    for i in range(n):
        address = f"node-{i}" if protocol == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol == "unix" else "127.0.0.1"

        # Create model and data partition
        model = model_fn()
        data_partition = partitions[i]

        # Store creation parameters for potential recovery (save types, not instances)
        node_creation_params[address] = {
            "model": model,
            "data": data_partition,
            "protocol_type": protocol,      # Save type string, not instance
            "aggregator_type": aggregator,   # Save type string, not instance
            "addr": address,
        }

        # Create protocol and aggregator instances for initial node
        comm_protocol = MemoryCommunicationProtocol() if protocol == "memory" else GrpcCommunicationProtocol()
        agg = Scaffold() if aggregator == "scaffold" else None

        # Create and start node
        node = Node(
            model,
            data_partition,
            protocol=comm_protocol,
            addr=address,
            aggregator=agg,
        )
        node.start()
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        wait_convergence(nodes, n - 1, only_direct=False, wait=60)  # type: ignore

        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Set timeouts to shorter values for faster failure detection
        # But increase aggregation timeout after recovery to allow time for model synchronization
        Settings.training.VOTE_TIMEOUT = 10
        Settings.training.AGGREGATION_TIMEOUT = 60  # Increased to 60 seconds to allow time for model synchronization, especially after node recovery
        logger.info("", f"Vote timeout set to 10 seconds, aggregation timeout set to 60 seconds. If not enough nodes found, will proceed with available nodes.")
        
        # Start monitoring thread for node failures
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_rounds_and_control_nodes,
            args=(nodes, failure_nodes, failure_round, recovery_round, adjacency_matrix, stop_event, node_creation_params, protocol, aggregator, model_fn),
            daemon=True,
        )
        monitor_thread.start()
        logger.info("", f"Started monitoring thread. Nodes {failure_nodes} will fail after round {failure_round} completes and recover at round {recovery_round}.")

        # Start Learning
        nodes[0].set_start_learning(rounds=r, epochs=e)

        # Wait and check
        wait_to_finish(nodes, timeout=60 * 60)  # 1 hour

        # Stop monitoring thread
        stop_event.set()
        monitor_thread.join(timeout=5)

        # Local Logs
        if show_metrics:
            local_logs = logger.get_local_logs()
            if local_logs != {}:
                logs_l = list(local_logs.items())[0][1]
                #  Plot experiment metrics
                for round_num, round_metrics in logs_l.items():
                    for node_name, node_metrics in round_metrics.items():
                        for metric, values in node_metrics.items():
                            x, y = zip(*values, strict=False)
                            plt.plot(x, y, label=metric)
                            # Add a red point to the last data point
                            plt.scatter(x[-1], y[-1], color="red")
                            plt.title(f"Round {round_num} - {node_name}")
                            plt.xlabel("Epoch")
                            plt.ylabel(metric)
                            plt.legend()
                            plt.show()

            # Global Logs
            global_logs = logger.get_global_logs()
            if global_logs != {}:
                logs_g = list(global_logs.items())[0][1]  # Accessing the nested dictionary directly
                # Plot experiment metrics
                for node_name, node_metrics in logs_g.items():
                    for metric, values in node_metrics.items():
                        x, y = zip(*values, strict=False)
                        plt.plot(x, y, label=metric)
                        # Add a red point to the last data point
                        plt.scatter(x[-1], y[-1], color="red")
                        plt.title(f"{node_name} - {metric}")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.show()
    except Exception as e:
        raise e
    finally:
        # Stop monitoring thread
        stop_event.set()
        # Stop Nodes
        for node in nodes:
            try:
                node.stop()
            except Exception:
                pass

        if measure_time:
            print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # Parse args
    args = __parse_args()

    set_standalone_settings()

    if args.profiling:
        import os  # noqa: I001
        import yappi  # type: ignore

        # Start profiler
        yappi.start()

    # Set logger
    if args.token != "":
        logger.connect(p2pfl_web_url="http://localhost:3000/api/v1", p2pfl_web_key=args.token)

    # Seed
    if args.seed is not None:
        Settings.general.SEED = args.seed

    # Launch experiment
    try:
        mnist_with_failure(
            args.nodes,
            args.rounds,
            args.epochs,
            show_metrics=args.show_metrics,
            measure_time=args.measure_time,
            protocol=args.protocol,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=args.reduced_dataset,
            topology=args.topology,
            batch_size=args.batch_size,
            failure_round=args.failure_round,
            recovery_round=args.recovery_round,
            failure_nodes=args.failure_nodes,
        )
    finally:
        if args.profiling:
            # Stop profiler
            yappi.stop()
            # Save stats
            profile_dir = os.path.join("profile", "mnist", str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")

