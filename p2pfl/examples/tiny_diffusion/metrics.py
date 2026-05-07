"""Distribution-quality metrics for 2D point clouds.

Sliced Wasserstein-2 and Maximum Mean Discrepancy (Gaussian kernel).
Both are native 2D metrics — FID/KID don't apply here since they require
Inception features over images.
"""

from __future__ import annotations

import numpy as np


def sliced_wasserstein_2(real: np.ndarray, generated: np.ndarray, n_projections: int = 100, seed: int = 0) -> float:
    """
    Sliced Wasserstein-2 distance between two 2D point sets.

    Projects both sets onto random unit directions, computes 1D Wasserstein
    distance per projection (closed-form: sort + L2 of the difference), and
    averages. Stable approximation of W2 in low dimensions.

    Args:
        real: Real samples, shape (N, 2).
        generated: Generated samples, shape (M, 2).
        n_projections: Number of random directions to average over.
        seed: RNG seed for reproducibility.

    Returns:
        Sliced W2 distance (lower is better).

    """
    rng = np.random.default_rng(seed)
    # Random unit directions in 2D
    angles = rng.uniform(0, 2 * np.pi, size=n_projections)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (P, 2)

    # Project: (N, 2) @ (2, P) -> (N, P)
    real_proj = real @ directions.T
    gen_proj = generated @ directions.T

    # 1D Wasserstein-2: sort both, mean squared difference per projection
    real_sorted = np.sort(real_proj, axis=0)
    gen_sorted = np.sort(gen_proj, axis=0)

    # Resample if sizes differ (linear interpolation on quantiles)
    if real_sorted.shape[0] != gen_sorted.shape[0]:
        n = min(real_sorted.shape[0], gen_sorted.shape[0])
        idx_r = np.linspace(0, real_sorted.shape[0] - 1, n).astype(int)
        idx_g = np.linspace(0, gen_sorted.shape[0] - 1, n).astype(int)
        real_sorted = real_sorted[idx_r]
        gen_sorted = gen_sorted[idx_g]

    diff_sq = (real_sorted - gen_sorted) ** 2
    w2_per_projection = np.sqrt(diff_sq.mean(axis=0))
    return float(w2_per_projection.mean())


def gaussian_mmd(real: np.ndarray, generated: np.ndarray, sigma: float = 0.1) -> float:
    """
    Maximum Mean Discrepancy with Gaussian RBF kernel between two 2D point sets.

    MMD² = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)] where k is Gaussian.
    Returns sqrt of the squared MMD (clamped at 0).

    Args:
        real: Real samples, shape (N, 2).
        generated: Generated samples, shape (M, 2).
        sigma: Bandwidth of the Gaussian kernel. Default 0.1 fits points in [-1, 1].

    Returns:
        MMD distance (lower is better).

    """

    def _rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # ||a-b||² = ||a||² + ||b||² - 2 a·b
        sq_a = (a**2).sum(axis=1, keepdims=True)
        sq_b = (b**2).sum(axis=1, keepdims=True).T
        sq = sq_a + sq_b - 2 * (a @ b.T)
        return np.exp(-sq / (2 * sigma**2))

    k_xx = _rbf(real, real).mean()
    k_yy = _rbf(generated, generated).mean()
    k_xy = _rbf(real, generated).mean()

    mmd_sq = float(k_xx + k_yy - 2 * k_xy)
    return float(np.sqrt(max(mmd_sq, 0.0)))
