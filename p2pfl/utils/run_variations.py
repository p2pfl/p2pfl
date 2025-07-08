#!/usr/bin/env python3
#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Run experiments with parameter variations."""

# python -m p2pfl.utils.run_variations p2pfl/examples/cifar10/cifar10.yaml \
#   --aggregators FedAvg FedMedian Scaffold FedAdagrad FedAdam FedProx FedYogi Krum \
#   --seeds 42 --rounds 10 --epochs 1 --nodes 10 --param settings.training.ray_actor_pool_size=8

import argparse
import copy
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from p2pfl.management.launch_from_yaml import run_from_yaml
from p2pfl.management.logger import logger


def generate_experiment_id(config: Dict[str, Any], variation_params: Dict[str, Any]) -> str:
    """
    Generate a unique identifier for an experiment configuration.

    Args:
        config: The full configuration dictionary
        variation_params: The parameters that are being varied

    Returns:
        A unique string identifier

    """
    # Create a deterministic string representation of the variation parameters
    param_str = json.dumps(variation_params, sort_keys=True)
    # Use first 8 chars of hash for brevity
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

    # Create a human-readable part from key parameters
    parts = []
    for key, value in sorted(variation_params.items()):
        key_short = key.split(".")[-1]  # Get last part of dot notation
        value_str = value if isinstance(value, str) else str(value)
        parts.append(f"{key_short}_{value_str}")

    readable_part = "_".join(parts[:3])  # Limit to 3 params for readability

    return f"{readable_part}_{param_hash}"


def set_nested_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using dot notation.

    Args:
        config: The configuration dictionary
        path: Dot-separated path (e.g., 'experiment.rounds')
        value: The value to set

    """
    keys = path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def check_results_exist(results_dir: Path) -> bool:
    """
    Check if results already exist for this experiment.

    Args:
        results_dir: Directory where results would be saved

    Returns:
        True if results exist, False otherwise

    """
    if not results_dir.exists():
        return False

    # Check for any CSV files which indicate completed experiment
    csv_files = list(results_dir.glob("*.csv"))
    return len(csv_files) > 0


def save_variation_config(config: Dict[str, Any], results_dir: Path) -> None:
    """Save the configuration used for this variation."""
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_single_experiment(config: Dict[str, Any], results_dir: Path) -> None:
    """
    Run a single experiment with the given configuration.

    Args:
        config: The configuration dictionary
        results_dir: Directory to save results

    """
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration
    save_variation_config(config, results_dir)

    # Record start time
    start_time = time.time()

    try:
        # Run the experiment
        # We need to modify run_from_yaml to accept a config dict
        # For now, we'll save a temporary YAML file
        temp_yaml_path = results_dir / "temp_config.yaml"
        with open(temp_yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        run_from_yaml(str(temp_yaml_path), debug=False)

        # Clean up temp file
        temp_yaml_path.unlink()

        print("Experiment completed successfully.")

    except Exception as e:
        print(f"Error running experiment: {e}")
        raise

    finally:
        # Save logs and metrics
        save_experiment_results(results_dir, start_time)


def save_experiment_results(results_dir: Path, start_time: float) -> None:
    """Save experiment results including logs and metrics."""
    # Save message logs
    all_msgs = logger.get_messages(direction="all")
    if all_msgs:
        try:
            pandas_msgs = pd.DataFrame(all_msgs)
            msg_csv_path = results_dir / "messages.csv"
            pandas_msgs.to_csv(msg_csv_path, index=False)
            print(f"Saved messages log to: {msg_csv_path}")
        except Exception as e:
            print(f"Error saving messages log: {e}")

    # Save global metrics
    global_metrics_data = logger.get_global_logs()
    if global_metrics_data:
        flattened_global_metrics = []
        try:
            for exp, nodes in global_metrics_data.items():
                for node, metrics in nodes.items():
                    for metric_name, values in metrics.items():
                        for round_num, value in values:
                            flattened_global_metrics.append(
                                {"experiment": exp, "node": node, "metric": metric_name, "round": round_num, "value": value}
                            )

            if flattened_global_metrics:
                pandas_global_metrics = pd.DataFrame(flattened_global_metrics)
                global_metrics_csv_path = results_dir / "global_metrics.csv"
                pandas_global_metrics.to_csv(global_metrics_csv_path, index=False)
                print(f"Saved global metrics log to: {global_metrics_csv_path}")
        except Exception as e:
            print(f"Error saving global metrics: {e}")

    # Save system metrics
    system_metrics_data = logger.get_system_metrics()
    if system_metrics_data:
        flattened_system_metrics = []
        try:
            for timestamp, sys_metrics in system_metrics_data.items():
                for sys_metric_name, sys_value in sys_metrics.items():
                    flattened_system_metrics.append(
                        {"timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"), "metric_name": sys_metric_name, "metric_value": sys_value}
                    )

            if flattened_system_metrics:
                pandas_system_metrics = pd.DataFrame(flattened_system_metrics)
                system_metrics_csv_path = results_dir / "system_resources.csv"
                pandas_system_metrics.to_csv(system_metrics_csv_path, index=False)
                print(f"Saved system resource metrics log to: {system_metrics_csv_path}")
        except Exception as e:
            print(f"Error saving system resource metrics: {e}")

    # Save execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.4f} seconds")

    time_csv_path = results_dir / "execution_time.csv"
    try:
        time_df = pd.DataFrame({"Execution Time (s)": [f"{execution_time:.4f}"]})
        time_df.to_csv(time_csv_path, index=False)
        print(f"Saved execution time to: {time_csv_path}")
    except Exception as e:
        print(f"Error saving execution time: {e}")


def parse_custom_params(param_strings: List[str]) -> Dict[str, List[Any]]:
    """
    Parse custom parameter strings in the format 'path.to.param=value1,value2'.

    Args:
        param_strings: List of parameter strings

    Returns:
        Dictionary mapping parameter paths to lists of values

    """
    custom_params = {}

    for param_str in param_strings:
        if "=" not in param_str:
            raise ValueError(f"Invalid parameter format: {param_str}. Expected 'path=value1,value2'")

        path, values_str = param_str.split("=", 1)
        values: List[Any] = []

        for value in values_str.split(","):
            # Try to parse as number
            try:
                if "." in value:
                    values.append(float(value))
                else:
                    values.append(int(value))
            except ValueError:
                # Parse as boolean or string
                if value.lower() == "true":
                    values.append(True)
                elif value.lower() == "false":
                    values.append(False)
                else:
                    values.append(value)

        custom_params[path] = values

    return custom_params


def main() -> int:
    """Execute the main function for running experiments with variations."""
    parser = argparse.ArgumentParser(
        description="Run P2PFL experiments with parameter variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with different aggregators and seeds
  python -m p2pfl.utils.run_variations config.yaml --aggregators FedAvg FedMedian --seeds 42 123

  # Run with different network configurations
  python -m p2pfl.utils.run_variations config.yaml --nodes 5 10 20 --topologies star ring full

  # Run with custom parameters using dot notation
  python -m p2pfl.utils.run_variations config.yaml --param experiment.dataset.batch_size=32,64,128
        """,
    )

    parser.add_argument("yaml_path", help="Path to the base YAML configuration file")
    parser.add_argument("--output-dir", help="Base directory for results (default: results/variations)")

    # Common parameter variations
    parser.add_argument("--aggregators", nargs="+", help="List of aggregator classes")
    parser.add_argument("--seeds", nargs="+", type=int, help="List of random seeds")
    parser.add_argument("--nodes", nargs="+", type=int, help="Number of nodes")
    parser.add_argument("--rounds", nargs="+", type=int, help="Number of rounds")
    parser.add_argument("--epochs", nargs="+", type=int, help="Number of epochs per round")
    parser.add_argument("--topologies", nargs="+", help="Network topologies")
    parser.add_argument("--partitioning", nargs="+", help="Dataset partitioning strategies")
    parser.add_argument("--models", nargs="+", help="Model packages/architectures")
    parser.add_argument("--batch-sizes", nargs="+", type=int, help="Batch sizes")

    # Custom parameters
    parser.add_argument("--param", action="append", dest="custom_params", help="Custom parameter in format 'path.to.param=value1,value2'")

    # Execution options
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip experiments with existing results (default: True)")
    parser.add_argument("--force", action="store_true", help="Force re-run all experiments, ignoring existing results")

    args = parser.parse_args()

    # Load base configuration
    with open(args.yaml_path) as f:
        base_config = yaml.safe_load(f)

    # Set output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        yaml_name = Path(args.yaml_path).stem
        output_base_dir = Path("results") / "variations" / yaml_name

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter variations
    variations = {}

    # Standard parameters
    if args.aggregators:
        # Create a special handling for aggregators to ensure package and name stay paired
        aggregator_packages = {
            "FedAvg": "p2pfl.learning.aggregators.fedavg",
            "FedMedian": "p2pfl.learning.aggregators.fedmedian",
            "Scaffold": "p2pfl.learning.aggregators.scaffold",
            "FedAdagrad": "p2pfl.learning.aggregators.fedadagrad",
            "FedAdam": "p2pfl.learning.aggregators.fedadam",
            "FedProx": "p2pfl.learning.aggregators.fedprox",
            "FedYogi": "p2pfl.learning.aggregators.fedyogi",
            "Krum": "p2pfl.learning.aggregators.krum",
        }
        # We'll handle aggregator variations specially after creating combinations
        variations["_aggregator_config"] = [
            {"name": agg, "package": aggregator_packages.get(agg, f"p2pfl.learning.aggregators.{agg.lower()}")} for agg in args.aggregators
        ]

    if args.seeds:
        variations["experiment.seed"] = args.seeds

    if args.nodes:
        variations["network.nodes"] = args.nodes

    if args.rounds:
        variations["experiment.rounds"] = args.rounds

    if args.epochs:
        variations["experiment.epochs"] = args.epochs

    if args.topologies:
        variations["network.topology"] = args.topologies

    if args.partitioning:
        variations["experiment.dataset.partitioning.strategy"] = args.partitioning

    if args.models:
        variations["experiment.model.package"] = args.models

    if args.batch_sizes:
        variations["experiment.dataset.batch_size"] = args.batch_sizes

    # Custom parameters
    if args.custom_params:
        custom_params = parse_custom_params(args.custom_params)
        variations.update(custom_params)

    # Generate all combinations
    combinations: List[Dict[str, Any]] = []
    if not variations:
        print("No variations specified. Running single experiment with base configuration.")
        combinations = [{}]
    else:
        # Create list of (param_path, values) tuples
        param_items = list(variations.items())
        param_names = [item[0] for item in param_items]
        param_values = [item[1] for item in param_items]

        # Generate cartesian product
        for values in itertools.product(*param_values):
            combo = dict(zip(param_names, values))

            # Handle special case for aggregator config
            if "_aggregator_config" in combo:
                agg_config = combo.pop("_aggregator_config")
                combo["experiment.aggregator.aggregator"] = agg_config["name"]
                combo["experiment.aggregator.package"] = agg_config["package"]

            combinations.append(combo)

    print(f"Total number of experiments to run: {len(combinations)}")
    print(f"Output directory: {output_base_dir}")
    print("-" * 50)

    # Run experiments
    completed = 0
    skipped = 0
    failed = 0

    for i, variation_params in enumerate(combinations, 1):
        print(f"\nExperiment {i}/{len(combinations)}")
        print(f"Parameters: {variation_params}")

        # Create config for this variation
        config = copy.deepcopy(base_config)

        # Apply variations
        for param_path, value in variation_params.items():
            set_nested_value(config, param_path, value)

        # Generate experiment ID and results directory
        exp_id = generate_experiment_id(config, variation_params)
        results_dir = output_base_dir / exp_id

        # Check if already exists
        if not args.force and args.skip_existing and check_results_exist(results_dir):
            print(f"Results already exist in {results_dir}. Skipping.")
            skipped += 1
            continue

        try:
            print(f"Running experiment: {exp_id}")
            print(f"Results directory: {results_dir}")

            # Reset logger state before running the experiment
            logger.reset()

            run_single_experiment(config, results_dir)
            completed += 1

        except Exception as e:
            print(f"ERROR: Experiment failed: {e}")
            failed += 1
            if not args.force:
                print("Stopping due to error. Use --force to continue despite errors.")
                break

        print("-" * 50)

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total experiments: {len(combinations)}")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Results saved in: {output_base_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
