"""
koopman/lifting.py

Lifting and decoding networks for the Koopman state space.

LiftingNet : phi (dim=PHI_DIM) → z (dim=koopman_dim)
DecoderNet : z   (dim=koopman_dim) → phi_hat (dim=PHI_DIM)

The lifting is the core non-linear observable map that makes Koopman theory
work on finite-dimensional data: we embed the high-dim probe observations into
a compact space where the dynamics / geometry-to-performance map is approximately
linear (i.e. governed by A(theta)).
"""

import torch
import torch.nn as nn


class LiftingNet(nn.Module):
    """
    phi → z  :  project flow-observable vector to Koopman state space.

    Three-layer MLP with LayerNorm + GELU. The output is NOT normalised —
    the fixed-point loss during training provides the implicit regularisation.
    """

    def __init__(self, phi_dim: int, koopman_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, koopman_dim),
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """phi: (..., phi_dim) → z: (..., koopman_dim)"""
        return self.net(phi)


class DecoderNet(nn.Module):
    """
    z → phi_hat  :  approximate inverse of LiftingNet.

    Used ONLY during training for autoencoder regularisation (encourages the
    lifting to be informative). Not needed at inference time.
    """

    def __init__(self, koopman_dim: int, phi_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(koopman_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, phi_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., koopman_dim) → phi_hat: (..., phi_dim)"""
        return self.net(z)
