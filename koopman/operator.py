"""
koopman/operator.py

Parametric Koopman operator A(theta) and bias b(theta).

Formulation — low-rank dyadic decomposition:

    A(theta) = I + sum_{k=1}^{r} alpha_k(theta) * (u_k ⊗ v_k^T)

where:
    u_k, v_k  ∈ R^K   — LEARNED fixed basis vectors (shared across all
                         geometries; the operator "dictionary")
    alpha_k(theta) ∈ R — geometry-dependent weights from a small theta-MLP

Efficient evaluation: A @ z = z + U^T @ diag(alpha) @ (V @ z)
    Cost: O(r·K) per forward pass vs. O(K²) for a naive dense matrix.

    With K=256, r=32: 32×256×2 = 16,384 basis params + small MLP —
    vs. 256×256 = 65,536 entries if output directly from MLP.

The bias b(theta) ∈ R^K captures the "forcing" side of the fixed-point
equation  A(theta)·z* = z* + b(theta)  ⟺  (A−I)·z* = b.

This is the parametric-DMD / operator-learning formulation:
all geometries share the same operator vocabulary; only the weights vary.
"""

import torch
import torch.nn as nn


class ParametricOperator(nn.Module):
    """
    Geometry-conditioned Koopman operator A(theta) and bias b(theta).

    The forward() method returns (A·z, b) using the efficient low-rank form.
    Call get_full_matrix() only when you need the explicit K×K matrix
    (eigenanalysis, HHL preparation) — it is O(r·K²) and should not be
    called inside the training loop.
    """

    def __init__(
        self,
        theta_dim:   int,
        koopman_dim: int,
        rank:        int = 32,
        hidden_dim:  int = 64,
    ):
        super().__init__()
        self.K = koopman_dim
        self.r = rank

        # Learnable basis vectors — initialise small so A starts near identity
        self.U = nn.Parameter(torch.randn(rank, koopman_dim) * 0.01)  # (r, K)
        self.V = nn.Parameter(torch.randn(rank, koopman_dim) * 0.01)  # (r, K)

        # theta → alpha ∈ R^r  (operator weights)
        self.alpha_net = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, rank),
        )

        # theta → b ∈ R^K  (forcing bias)
        self.b_net = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, koopman_dim),
        )

    def apply_operator(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute A(theta) @ z using the low-rank structure.

            A @ z = z + U^T @ (alpha ⊙ (V @ z))

        Parameters
        ----------
        theta : (batch, theta_dim)
        z     : (batch, K)

        Returns
        -------
        Az    : (batch, K)
        """
        alpha = self.alpha_net(theta)                           # (batch, r)
        Vz    = torch.einsum('rk,bk->br', self.V, z)           # (batch, r)
        aVz   = alpha * Vz                                      # (batch, r)
        delta = torch.einsum('rk,br->bk', self.U, aVz)         # (batch, K)
        return z + delta

    def get_bias(self, theta: torch.Tensor) -> torch.Tensor:
        """b(theta) : (batch, theta_dim) → (batch, K)"""
        return self.b_net(theta)

    def get_full_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Materialise the full K×K matrix A(theta) for a single theta.

            A = I + U^T @ diag(alpha) @ V

        Called for eigenvalue analysis and HHL preparation — NOT during training.

        theta : (theta_dim,) or (1, theta_dim)
        Returns: (K, K) float32
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        alpha = self.alpha_net(theta)   # (1, r)
        A = (torch.eye(self.K, device=theta.device)
             + self.U.T @ torch.diag(alpha[0]) @ self.V)
        return A

    def get_full_matrix_batch(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Materialise A(theta) for a full batch. Returns (B, K, K).

        Used in the DrivAerNet Cd-only training path where we solve
        (A-I)z* = b per sample via torch.linalg.solve (batched).
        Not used in the AhmedML hot loop — that uses apply_operator instead.

        theta : (B, theta_dim)
        Returns: (B, K, K) float32
        """
        B     = theta.shape[0]
        alpha = self.alpha_net(theta)                                   # (B, r)
        I     = torch.eye(self.K, device=theta.device).unsqueeze(0)    # (1, K, K)
        dA    = torch.einsum('rk,br,rj->bkj', self.U, alpha, self.V)  # (B, K, K)
        return I + dA

    def forward(self, theta: torch.Tensor, z: torch.Tensor):
        """
        Returns (Az, b) where Az = A(theta) @ z,  b = b(theta).

        Parameters
        ----------
        theta : (batch, theta_dim)
        z     : (batch, K)

        Returns
        -------
        Az : (batch, K)
        b  : (batch, K)
        """
        return self.apply_operator(theta, z), self.get_bias(theta)
