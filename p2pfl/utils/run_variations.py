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
from typing import Any

import pandas as pd
import yaml

from p2pfl.management.launch_from_yaml import run_from_yaml
from p2pfl.management.logger import logger


def generate_experiment_id(variation_params: dict[str, Any], use_short_names: bool = True) -> str:
    """
    Generate a unique identifier for an experiment configuration.

    Args:
        variation_params: The parameters that are being varied
        use_short_names: Whether to use abbreviated parameter names in the identifier

    Returns:
        A unique string identifier including all varied parameters

    """
    # Create a deterministic string representation of the variation parameters
    param_str = json.dumps(variation_params, sort_keys=True)
    # Use first 6 chars of hash for uniqueness (in case of very long names)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:6]

    # Create a human-readable part from ALL parameters
    parts = []
    for key, value in sorted(variation_params.items()):
        # Get meaningful name from the parameter path
        if use_short_names:
            key_parts = key.split(".")
            # Use last 2 parts for long paths, else last part
            key_short = "_".join(key_parts[-2:]) if len(key_parts) > 2 else key_parts[-1]
        else:
            # Use full parameter path with dots replaced by underscores
            key_short = key.replace(".", "_")

        # Format the value appropriately
        if isinstance(value, bool):
            value_str = "T" if value else "F"
        elif isinstance(value, float):
            # Format floats to avoid too many decimals
            value_str = f"{value:.4g}".replace(".", "p")
        elif isinstance(value, str):
            # Truncate long strings and remove special chars
            value_str = value[:15].replace("/", "-").replace(" ", "")
        else:
            value_str = str(value)

        parts.append(f"{key_short}_{value_str}")

    # Join all parts - no limit on number of parameters
    readable_part = "_".join(parts)

    # If the name is too long (>200 chars), truncate and add hash
    readable_part = readable_part[:194] + f"_{param_hash}" if len(readable_part) > 200 else f"{readable_part}_{param_hash}"

    return readable_part


def set_nested_value(config: dict[str, Any], path: str, value: Any) -> None:
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


def save_variation_config(config: dict[str, Any], results_dir: Path) -> None:
    """Save the configuration used for this variation."""
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_single_experiment(config: dict[str, Any], results_dir: Path) -> None:
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


def parse_custom_params(param_strings: list[str]) -> dict[str, list[Any]]:
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
        values: list[Any] = []

        for value in values_str.split(","):
            # Try to parse as number
            try:
                # First try as float (handles scientific notation like 1e-9)
                if "." in value or "e" in value.lower():
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


def run_variations_experiment(
    yaml_path: str,
    aggregators: list[str] | None = None,
    seeds: list[int] | None = None,
    nodes: list[int] | None = None,
    rounds: list[int] | None = None,
    epochs: list[int] | None = None,
    topologies: list[str] | None = None,
    partitioning: list[str] | None = None,
    models: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    custom_params: list[str] | None = None,
    output_dir: str | None = None,
    skip_existing: bool = True,
    force: bool = False,
    full_param_names: bool = False,
    console=None,
) -> int:
    """
    Run experiments with parameter variations.

    Args:
        yaml_path: Path to the base YAML configuration file
        aggregators: List of aggregator classes
        seeds: List of random seeds
        nodes: Number of nodes
        rounds: Number of rounds
        epochs: Number of epochs per round
        topologies: Network topologies
        partitioning: Dataset partitioning strategies
        models: Model packages/architectures
        batch_sizes: Batch sizes
        custom_params: Custom parameters in format 'path.to.param=value1,value2'
        output_dir: Base directory for results
        skip_existing: Skip experiments with existing results
        force: Force re-run all experiments
        full_param_names: Use full parameter paths in folder names
        console: Rich console for output (optional)

    Returns:
        0 if successful, 1 if there were failures

    """
    # Use print if no console provided
    if console is None:
        console = type("Console", (), {"print": print})()

    # Load base configuration
    with open(yaml_path) as f:
        base_config = yaml.safe_load(f)

    # Set output directory
    if output_dir:
        output_base_dir = Path(output_dir)
    else:
        yaml_name = Path(yaml_path).stem
        output_base_dir = Path("results") / "variations" / yaml_name

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter variations
    variations: dict[str, Any] = {}

    # Standard parameters
    if aggregators:
        # Create a special handling for aggregators to ensure package and name stay paired
        aggregator_packages = {
            "FedAvg": "p2pfl.learning.aggregators.fedavg",
            "FedMedian": "p2pfl.learning.aggregators.fedmedian",
            "Scaffold": "p2pfl.learning.aggregators.scaffold",
            "FedAdagrad": "p2pfl.learning.aggregators.fedopt",
            "FedYogi": "p2pfl.learning.aggregators.fedopt",
            "FedProx": "p2pfl.learning.aggregators.fedprox",
            "FedAdam": "p2pfl.learning.aggregators.fedopt",
            "Krum": "p2pfl.learning.aggregators.krum",
        }
        # We'll handle aggregator variations specially after creating combinations
        variations["_aggregator_config"] = [
            {"name": agg, "package": aggregator_packages.get(agg, f"p2pfl.learning.aggregators.{agg.lower()}")} for agg in aggregators
        ]

    if seeds:
        variations["experiment.seed"] = seeds

    if nodes:
        variations["network.nodes"] = nodes

    if rounds:
        variations["experiment.rounds"] = rounds

    if epochs:
        variations["experiment.epochs"] = epochs

    if topologies:
        variations["network.topology"] = topologies

    if partitioning:
        variations["experiment.dataset.partitioning.strategy"] = partitioning

    if models:
        variations["experiment.model.package"] = models

    if batch_sizes:
        variations["experiment.dataset.batch_size"] = batch_sizes

    # Custom parameters
    if custom_params:
        custom_params_parsed = parse_custom_params(custom_params)
        variations.update(custom_params_parsed)

    # Generate all combinations
    combinations: list[dict[str, Any]] = []
    if not variations:
        console.print("No variations specified. Running single experiment with base configuration.")
        combinations = [{}]
    else:
        # Create list of (param_path, values) tuples
        param_items = list(variations.items())
        param_names = [item[0] for item in param_items]
        param_values = [item[1] for item in param_items]

        # Generate cartesian product
        for values in itertools.product(*param_values):
            combo = dict(zip(param_names, values, strict=False))

            # Handle special case for aggregator config
            if "_aggregator_config" in combo:
                agg_config = combo.pop("_aggregator_config")
                combo["experiment.aggregator.aggregator"] = agg_config["name"]
                combo["experiment.aggregator.package"] = agg_config["package"]

            combinations.append(combo)

    console.print(f"Total number of experiments to run: {len(combinations)}")
    console.print(f"Output directory: {output_base_dir}")
    console.print("-" * 50)

    # Run experiments
    completed = 0
    skipped = 0
    failed = 0

    for i, variation_params in enumerate(combinations, 1):
        console.print(f"\nExperiment {i}/{len(combinations)}")
        console.print(f"Parameters: {variation_params}")

        # Create config for this variation
        config = copy.deepcopy(base_config)

        # Apply variations
        for param_path, value in variation_params.items():
            set_nested_value(config, param_path, value)

        # Generate experiment ID and results directory
        exp_id = generate_experiment_id(variation_params, use_short_names=not full_param_names)
        results_dir = output_base_dir / exp_id

        # Check if already exists
        if not force and skip_existing and check_results_exist(results_dir):
            console.print(f"Results already exist in {results_dir}. Skipping.")
            skipped += 1
            continue

        try:
            console.print(f"Running experiment: {exp_id}")
            console.print(f"Results directory: {results_dir}")

            # Reset logger state before running the experiment
            logger.reset()

            run_single_experiment(config, results_dir)
            completed += 1

        except Exception as e:
            console.print(f"ERROR: Experiment failed: {e}")
            failed += 1
            if not force:
                console.print("Stopping due to error. Use --force to continue despite errors.")
                break

        console.print("-" * 50)

    # Summary
    console.print(f"\n{'=' * 50}")
    console.print("SUMMARY")
    console.print(f"{'=' * 50}")
    console.print(f"Total experiments: {len(combinations)}")
    console.print(f"Completed: {completed}")
    console.print(f"Skipped: {skipped}")
    console.print(f"Failed: {failed}")
    console.print(f"Results saved in: {output_base_dir}")

    return 0 if failed == 0 else 1


def main() -> int:
    """Execute the main function for running experiments with variations (CLI entry point)."""
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
    parser.add_argument(
        "--full-param-names", action="store_true", help="Use full parameter paths in folder names (default: use abbreviated names)"
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    return run_variations_experiment(
        yaml_path=args.yaml_path,
        aggregators=args.aggregators,
        seeds=args.seeds,
        nodes=args.nodes,
        rounds=args.rounds,
        epochs=args.epochs,
        topologies=args.topologies,
        partitioning=args.partitioning,
        models=args.models,
        batch_sizes=args.batch_sizes,
        custom_params=args.custom_params,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        force=args.force,
        full_param_names=args.full_param_names,
    )


if __name__ == "__main__":
    sys.exit(main())
