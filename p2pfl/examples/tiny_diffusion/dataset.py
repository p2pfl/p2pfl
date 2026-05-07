"""Synthetic 2D dataset with four distinct distributions for federated diffusion experiments."""

import numpy as np
from datasets import Dataset, DatasetDict

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


def _make_moons(n: int, noise: float, rng: np.random.RandomState) -> np.ndarray:
    """Two interleaving half-circles."""
    t = np.linspace(0, np.pi, n // 2)
    upper = np.column_stack([np.cos(t), np.sin(t)])
    lower = np.column_stack([np.cos(t) + 1, -np.sin(t) + 0.5])
    pts = np.vstack([upper, lower])
    pts += rng.randn(*pts.shape) * noise
    return pts


def _make_circles(n: int, noise: float, rng: np.random.RandomState) -> np.ndarray:
    """Two concentric circles."""
    t = np.linspace(0, 2 * np.pi, n // 2, endpoint=False)
    outer = np.column_stack([np.cos(t), np.sin(t)])
    inner = np.column_stack([np.cos(t) * 0.5, np.sin(t) * 0.5])
    pts = np.vstack([outer, inner])
    pts += rng.randn(*pts.shape) * noise
    return pts


def _make_spiral(n: int, noise: float, rng: np.random.RandomState) -> np.ndarray:
    """Single Archimedean spiral."""
    t = np.linspace(0.5, 4 * np.pi, n)
    r = t / (4 * np.pi)
    pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
    pts += rng.randn(*pts.shape) * noise
    return pts


def _make_cross(n: int, noise: float, rng: np.random.RandomState) -> np.ndarray:
    """Cross shape from four Gaussian blobs at cardinal positions."""
    centers = np.array([[0.0, 0.6], [0.0, -0.6], [0.6, 0.0], [-0.6, 0.0]])
    per_blob = n // 4
    blobs = [rng.randn(per_blob, 2) * 0.1 + c for c in centers]
    return np.vstack(blobs)


DISTRIBUTIONS = {
    0: ("moons", _make_moons),
    1: ("circles", _make_circles),
    2: ("spiral", _make_spiral),
    3: ("cross", _make_cross),
}


def _normalize(pts: np.ndarray) -> np.ndarray:
    """Normalize points to approximately [-1, 1]."""
    center = pts.mean(axis=0)
    scale = np.abs(pts - center).max()
    return (pts - center) / scale


def make_2d_dataset(
    n_samples_per_dist: int = 2000,
    noise: float = 0.05,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> P2PFLDataset:
    """
    Generate a synthetic 2D dataset with four distributions.

    Each sample has two fields:
        - ``x``: a 2D point as ``[x1, x2]`` (floats in ~[-1, 1]).
        - ``label``: integer 0-3 indicating the distribution (moons/circles/spiral/cross).

    Args:
        n_samples_per_dist: Number of points per distribution.
        noise: Gaussian noise added to each distribution.
        seed: Random seed for reproducibility.
        test_ratio: Fraction of data reserved for the test split.

    Returns:
        A ``P2PFLDataset`` with train/test splits ready for partitioning.

    """
    rng = np.random.RandomState(seed)

    all_x = []
    all_labels = []
    for label, (_name, gen_fn) in DISTRIBUTIONS.items():
        pts = gen_fn(n_samples_per_dist, noise, rng)
        pts = _normalize(pts)
        all_x.append(pts.astype(np.float32))
        all_labels.append(np.full(len(pts), label, dtype=np.int64))

    all_x = np.concatenate(all_x)
    all_labels = np.concatenate(all_labels)

    # Shuffle
    idx = rng.permutation(len(all_x))
    all_x = all_x[idx]
    all_labels = all_labels[idx]

    # Train/test split
    n_test = int(len(all_x) * test_ratio)
    train_x, test_x = all_x[:-n_test], all_x[-n_test:]
    train_labels, test_labels = all_labels[:-n_test], all_labels[-n_test:]

    train_ds = Dataset.from_dict({"x": train_x.tolist(), "label": train_labels.tolist()})
    test_ds = Dataset.from_dict({"x": test_x.tolist(), "label": test_labels.tolist()})

    return P2PFLDataset(DatasetDict({"train": train_ds, "test": test_ds}))
