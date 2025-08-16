#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""
Post-Training Differential Privacy strategy for Local DP.

This implements POST-TRAINING differential privacy, where noise is added once
to the final model update, NOT during training (unlike traditional DP-SGD).

DESIGN RATIONALE FOR POST-TRAINING LOCAL DP IN P2PFL:

In a decentralized federated learning environment like p2pfl, we made the
critical design decision to implement Local Differential Privacy (LDP) as a
"compression" technique rather than a training modification. Here's why:

1. NO TRUSTED AGGREGATOR:
   - In p2pfl, any node can act as an aggregator
   - There's no central trusted server - aggregators are just peers
   - We must assume aggregators could be curious or malicious
   - Therefore, privacy protection MUST happen before data leaves the client

2. WHY COMPRESSION FRAMEWORK?
   - p2pfl converts all models to numpy arrays before transmission
   - This conversion point is the perfect place to add privacy
   - Compression pipeline is already framework-agnostic (works with PyTorch, TF, etc.)
   - No need to modify training code or add framework-specific callbacks

3. LOCAL DP APPROACH:
   - Each client adds noise to their own updates (local privacy)
   - Privacy guarantee holds even if aggregator is malicious
   - No need for secure aggregation or trusted execution environments
   - Simple, clean integration with existing p2pfl architecture

4. IMPLEMENTATION AS "LOSSY COMPRESSION":
   - DP is essentially controlled information loss for privacy
   - Clipping = bounding the information content
   - Noise = lossy transformation that cannot be reversed
   - Fits naturally in the compression pipeline alongside quantization, sparsification

This design ensures that privacy protection is:
- Automatic: Applied during model encoding, no extra steps needed
- Framework-agnostic: Works with any ML framework
- Composable: Can combine with other compressions (quantization, TopK, etc.)
- Untrusted-aggregator-safe: Privacy guaranteed even against malicious aggregators

Refs:

1. "Deep Learning with Differential Privacy" (Abadi et al., 2016)
   https://arxiv.org/abs/1607.00133
   - Introduces DP-SGD with gradient clipping and noise addition
   - Foundation for our clipping + Gaussian noise approach

2. "cpSGD: Communication-efficient and differentially-private distributed SGD"
   (Agarwal et al., 2018) https://arxiv.org/abs/1805.10559
   - Combines compression with differential privacy
   - Shows DP can be viewed as a form of lossy compression

3. "Differentially Private Federated Learning: A Client Level Perspective"
   (Geyer et al., 2017) https://arxiv.org/abs/1712.07557
   - Focuses on client-level DP in federated settings
   - Emphasizes importance of local noise addition for untrusted aggregators

4. "Local Differential Privacy for Federated Learning" (Truex et al., 2020)
   https://arxiv.org/abs/1911.04882
   - Specifically addresses LDP in federated learning with untrusted servers
   - Supports our approach of applying DP before transmission

"""

try:
    import opendp.prelude as dp
except ImportError as err:
    raise ImportError("Please install with `pip install p2pfl[dp]`") from err

import numpy as np

from p2pfl.learning.compression.base_compression_strategy import TensorCompressor

# Allows you to use features added by the community
dp.enable_features("contrib")


class DifferentialPrivacyCompressor(TensorCompressor):
    """
    Apply POST-TRAINING local differential privacy as a compression technique.

    This implements POST-TRAINING DP, where privacy is applied once to the final
    model update after training completes, NOT during training iterations.

    Key differences from traditional DP-SGD:
    - Traditional DP-SGD: Clips/adds noise to gradients at EACH training step
    - Post-Training DP: Clips/adds noise ONCE to the final model update

    This approach:
    1. Requires NO modifications to local training code
    2. Works with any ML framework
    3. Typically has less accuracy loss (noise added once vs. many times)

    Note:
    The `params` argument is expected to be the *model update* (delta)
    after local training, not the full set of model weights.

    """

    def _get_noise_mechanism(
        self, noise_type: str, clip_norm: float, epsilon: float, delta: float, vec_len: int
    ) -> tuple[dp.Measurement, float]:
        """Create an OpenDP noise mechanism."""
        if noise_type == "laplace":
            scale = clip_norm / epsilon
            space = dp.vector_domain(dp.atom_domain(T=float, nan=False), vec_len), dp.l1_distance(T=float)
            mech = space >> dp.m.then_laplace(scale=scale)
        elif noise_type == "gaussian":
            scale = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            space = dp.vector_domain(dp.atom_domain(T=float, nan=False), vec_len), dp.l2_distance(T=float)
            mech = space >> dp.m.then_gaussian(scale=scale)
        else:
            raise ValueError("The parameter 'noise_type' must be 'gaussian' or 'laplace'")
        return mech, scale

    def apply_strategy(
        self,
        params: list[np.ndarray],
        clip_norm: float = 1.0,
        epsilon: float = 3.0,
        delta: float = 1e-5,
        noise_type: str = "gaussian",
        stability_constant: float = 1e-6,
    ) -> tuple[list[np.ndarray], dict]:
        """
        Apply differential privacy to model parameters.

        Args:
            params: Model update (delta) after local training
            clip_norm: Maximum L2 norm for clipping (C)
            epsilon: The privacy budget (ε). Lower values correspond
                to stronger privacy guarantees. Represents the maximum "information leakage"
                allowed for a single data point. A typical value is between 0.1 and 10.0.
            delta: The privacy budget (δ), typically a small number (e.g., 1e-5).
                It represents the probability that the privacy guarantee of epsilon
                is broken. It should be less than 1/N, where N is the number of
                data points in the dataset. This parameter is only used for Gaussian noise.
            noise_type: Type of noise to add ("gaussian" or "laplace")
            stability_constant: A small constant to avoid division by zero when clipping.

        Returns:
            Tuple of (dp_params, dp_info) where dp_info contains privacy parameters

        """
        # Handle empty input
        if not params:
            raise ValueError("DifferentialPrivacyCompressor: list 'params' must not be empty")

        # Step 1: Compute global L2 norm across all parameters
        flat_update = np.concatenate([p.flatten() for p in params])
        total_norm = np.linalg.norm(flat_update)

        # Step 2: Clip if necessary
        if total_norm > clip_norm:
            clip_factor = clip_norm / (total_norm + stability_constant)
            clipped_flat_update = flat_update * clip_factor
        else:
            clipped_flat_update = flat_update.copy()

        # Step 3: Get noise mechanism and add noise
        mech, scale = self._get_noise_mechanism(noise_type, clip_norm, epsilon, delta, clipped_flat_update.size)
        noisy_flat_update = mech(clipped_flat_update.tolist())

        # Unflatten the noisy update
        dp_params = []
        current_pos = 0
        for p in params:
            shape = p.shape
            size = p.size
            dtype = p.dtype
            dp_params.append(np.array(noisy_flat_update[current_pos : current_pos + size], dtype=dtype).reshape(shape))
            current_pos += size

        # Prepare info for privacy accounting
        dp_info = {
            "dp_applied": True,
            "clip_norm": clip_norm,
            "epsilon": epsilon,
            "delta": delta if noise_type == "gaussian" else None,
            "noise_type": noise_type,
            "noise_scale": scale,
            "original_norm": float(total_norm),
            "was_clipped": bool(total_norm > clip_norm),
        }

        return dp_params, dp_info

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Reverse the differential privacy transformation.

        Note: DP is inherently irreversible (that's the point!), so this just returns
        the parameters as-is. The additional_info contains metadata about what DP was applied.

        Args:
            params: The DP-protected parameters
            additional_info: Contains DP metadata (clip_norm, noise_multiplier, etc.)

        Returns:
            The same parameters (DP cannot be reversed)

        """
        # DP is irreversible by design - just return params
        return params


# Alias for backward compatibility
LocalDPCompressor = DifferentialPrivacyCompressor
