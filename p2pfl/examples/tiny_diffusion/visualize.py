"""Visualize generated samples from a trained tiny diffusion model vs. real distributions."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from p2pfl.examples.tiny_diffusion.dataset import DISTRIBUTIONS, _normalize, make_2d_dataset
from p2pfl.examples.tiny_diffusion.model.diffusion_pytorch import TinyDiffusion


DIST_NAMES = {0: "Moons", 1: "Circles", 2: "Spiral", 3: "Cross"}
COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12"}


def load_model(model_path: str, **model_kwargs) -> TinyDiffusion:
    """Load a TinyDiffusion model from a state_dict checkpoint."""
    model = TinyDiffusion(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def generate_real_points(n_per_dist: int = 1000, noise: float = 0.05, seed: int = 42) -> dict[int, np.ndarray]:
    """Generate reference points from each distribution."""
    rng = np.random.RandomState(seed)
    real = {}
    for label, (_name, gen_fn) in DISTRIBUTIONS.items():
        pts = gen_fn(n_per_dist, noise, rng)
        real[label] = _normalize(pts)
    return real


def visualize(model_path: str, output_path: str | None = None, n_samples: int = 2000, **model_kwargs) -> None:
    """Generate samples and plot them against real distributions."""
    model = load_model(model_path, **model_kwargs)
    real = generate_real_points()

    # Generate samples from the model
    with torch.no_grad():
        generated = model.sample(n_samples=n_samples).cpu().numpy()

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Real distributions
    ax = axes[0]
    for label, pts in real.items():
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, c=COLORS[label], label=DIST_NAMES[label])
    ax.set_title("Real distributions", fontsize=14)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.legend(markerscale=4, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Generated samples
    ax = axes[1]
    ax.scatter(generated[:, 0], generated[:, 1], s=4, alpha=0.5, c="#8e44ad")
    ax.set_title(f"Generated samples (n={n_samples})", fontsize=14)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Panel 3: Overlay
    ax = axes[2]
    for label, pts in real.items():
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.2, c=COLORS[label])
    ax.scatter(generated[:, 0], generated[:, 1], s=4, alpha=0.4, c="#8e44ad", label="Generated")
    ax.set_title("Overlay (real + generated)", fontsize=14)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.legend(markerscale=4, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        # Save next to the model checkpoint
        output_path = os.path.join(os.path.dirname(model_path), "generated_samples.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize tiny diffusion model results")
    parser.add_argument("model_path", help="Path to model.pt checkpoint")
    parser.add_argument("-o", "--output", default=None, help="Output image path (default: next to model.pt)")
    parser.add_argument("-n", "--n-samples", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100)
    args = parser.parse_args()

    visualize(
        args.model_path,
        output_path=args.output,
        n_samples=args.n_samples,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        timesteps=args.timesteps,
    )
