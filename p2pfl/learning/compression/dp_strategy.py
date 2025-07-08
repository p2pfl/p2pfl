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

from typing import Optional

import numpy as np

from p2pfl.learning.compression.base_compression_strategy import TensorCompressor


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

    The compressor acts as a state machine:
    - If previous_params is None: treats input as model update directly
    - If previous_params is provided: computes update = params - previous_params
    """

    def apply_strategy(
        self,
        params: list[np.ndarray],
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        previous_params: Optional[list[np.ndarray]] = None,
    ) -> tuple[list[np.ndarray], dict]:
        """
        Apply differential privacy to model parameters.

        Args:
            params: Current model parameters
            clip_norm: Maximum L2 norm for clipping (C)
            noise_multiplier: Noise scale relative to clip_norm (σ = C * noise_multiplier)
            previous_params: Previous round parameters (if None, treats params as update)

        Returns:
            Tuple of (dp_params, dp_info) where dp_info contains privacy parameters

        """
        # State machine: determine if we need to compute update
        if previous_params is None:
            # No previous params - treat input as update directly
            update_params = params
            computed_update = False
        else:
            # Previous params provided - compute update
            update_params = []
            for current, previous in zip(params, previous_params):
                update_params.append(current - previous)
            computed_update = True

        # Step 1: Compute global L2 norm across all parameters
        total_norm = 0.0
        for param in update_params:
            total_norm += np.sum(param**2)
        total_norm = np.sqrt(total_norm)

        # Step 2: Clip if necessary
        clipped_params = []
        if total_norm > clip_norm:
            clip_factor = clip_norm / total_norm
            for param in update_params:
                clipped_params.append(param * clip_factor)
        else:
            clipped_params = [p.copy() for p in update_params]

        # Step 3: Add Gaussian noise
        noise_scale = clip_norm * noise_multiplier
        noisy_updates = []
        for param in clipped_params:
            noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)
            noisy_updates.append(param + noise)

        # Step 4: If we computed update, add it back to previous params
        if computed_update and previous_params is not None:
            dp_params = []
            for dp_update, previous in zip(noisy_updates, previous_params):
                dp_params.append(previous + dp_update)
        else:
            dp_params = noisy_updates

        # Prepare info for privacy accounting
        dp_info = {
            "dp_applied": True,
            "clip_norm": clip_norm,
            "noise_multiplier": noise_multiplier,
            "noise_scale": noise_scale,
            "original_norm": float(total_norm),
            "was_clipped": total_norm > clip_norm,
            "computed_update": computed_update,
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
