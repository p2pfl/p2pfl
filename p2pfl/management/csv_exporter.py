#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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
"""CSV exporter for experiment metrics, communication logs, and model checkpoints."""

import csv
import os
from datetime import datetime

from p2pfl.management.logger import logger


def export_experiment_csv(nodes: list, output_dir: str = "results", run_dir: str | None = None) -> str:
    """
    Export experiment metrics, communication logs, and model checkpoints to a run directory.

    Writes:
        - ``metrics.csv``: global metrics (accuracy, loss, etc.) per node and round.
        - ``communication.csv``: all messages sent/received with byte sizes.
        - ``model.pt`` / ``model/``: the global model checkpoint (from the first node).

    Args:
        nodes: List of Node instances that participated in the experiment.
        output_dir: Base directory; a timestamped subdir is created when ``run_dir`` is None.
        run_dir: If provided, write directly into this directory (used to share with per-round artifacts).

    Returns:
        The path to the run directory.

    """
    if run_dir is None:
        run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    if not nodes:
        logger.warning("SYSTEM", "No nodes provided for export.")
        return run_dir

    # --- Global metrics ---
    metrics_path = os.path.join(run_dir, "metrics.csv")
    global_logs = logger.get_global_logs()
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "node", "metric", "round", "value"])
        for exp_name, node_logs in global_logs.items():
            for node_addr, metrics in node_logs.items():
                for metric_name, values in metrics.items():
                    for round_num, value in values:
                        writer.writerow([exp_name, node_addr, metric_name, round_num, value])

    # --- Communication logs ---
    comm_path = os.path.join(run_dir, "communication.csv")
    all_messages = logger.get_messages()
    with open(comm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "source", "destination", "direction", "cmd", "package_type", "package_size", "round"])
        for msg in all_messages:
            writer.writerow([
                msg.get("timestamp", ""),
                msg.get("source", ""),
                msg.get("destination", ""),
                msg.get("direction", ""),
                msg.get("cmd", ""),
                msg.get("package_type", ""),
                msg.get("package_size", 0),
                msg.get("round", 0),
            ])

    # --- Global model checkpoint (first node) ---
    try:
        p2pfl_model = nodes[0].model
        framework = p2pfl_model.get_framework()
        if framework == "pytorch":
            import torch

            torch.save(p2pfl_model.model.state_dict(), os.path.join(run_dir, "model.pt"))
        elif framework == "tensorflow":
            p2pfl_model.model.save(os.path.join(run_dir, "model"))
    except Exception as e:
        logger.warning("SYSTEM", f"Could not save global model: {e}")

    # --- Metrics plot (loss curve) ---
    try:
        _plot_metrics(global_logs, run_dir)
    except Exception as e:
        logger.warning("SYSTEM", f"Could not generate metrics plot: {e}")

    # --- Diffusion visualization (if applicable) ---
    try:
        model_pt = os.path.join(run_dir, "model.pt")
        if os.path.exists(model_pt):
            _plot_diffusion_samples(p2pfl_model.model, run_dir)
    except Exception as e:
        logger.warning("SYSTEM", f"Could not generate diffusion visualization: {e}")

    logger.info("SYSTEM", f"Experiment results exported to {os.path.abspath(run_dir)}/")
    return run_dir


def _plot_metrics(global_logs: dict, run_dir: str) -> None:
    """Plot metric curves per node across rounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for exp_name, node_logs in global_logs.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for node_addr, metrics in node_logs.items():
            for metric_name, values in metrics.items():
                rounds = [r for r, _ in values]
                vals = [v for _, v in values]
                ax.plot(rounds, vals, marker="o", markersize=3, label=f"{node_addr}")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Test loss per node across rounds", fontsize=14)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "metrics.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_diffusion_samples(model: object, run_dir: str) -> None:
    """Generate per-class samples for a conditional diffusion model and plot them next to the real distributions."""
    if not hasattr(model, "sample") or not hasattr(model, "num_classes"):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    try:
        from p2pfl.examples.tiny_diffusion.dataset import DISTRIBUTIONS, _normalize
    except ImportError:
        return

    colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12"}
    names = {0: "Moons", 1: "Circles", 2: "Spiral", 3: "Cross"}
    n_per_class = 1000
    rng = np.random.RandomState(42)

    real = {label: _normalize(gen_fn(n_per_class, 0.05, rng)) for label, (_name, gen_fn) in DISTRIBUTIONS.items()}

    model.eval()
    generated = {}
    with torch.no_grad():
        for label in range(model.num_classes):
            generated[label] = model.sample(n_samples=n_per_class, label=label).cpu().numpy()

    n_classes = model.num_classes
    fig, axes = plt.subplots(2, n_classes, figsize=(4 * n_classes, 8))

    for label in range(n_classes):
        # Top row: real
        axes[0, label].scatter(real[label][:, 0], real[label][:, 1], s=4, alpha=0.5, c=colors.get(label, "#444"))
        axes[0, label].set_title(f"Real — {names.get(label, label)}", fontsize=12)
        # Bottom row: generated
        axes[1, label].scatter(generated[label][:, 0], generated[label][:, 1], s=4, alpha=0.5, c=colors.get(label, "#444"))
        axes[1, label].set_title(f"Generated — {names.get(label, label)}", fontsize=12)

    for ax in axes.flat:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "generated_samples.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
