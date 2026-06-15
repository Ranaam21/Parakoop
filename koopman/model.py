"""
koopman/model.py

Full ParaKoop model: lifting + parametric operator + performance head.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING MODE  (mean-field supervision, no time series)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AhmedML gives (theta, phi_mean, Cd, Cl) — 500 (geometry, mean-field) pairs.
There are no time snapshots, so we use the FIXED-POINT / STEADY-STATE residual:

    z_bar = lift(phi_mean)                   # Koopman state of mean flow
    A(theta) @ z_bar ≈ z_bar + b(theta)     # z_bar is (approx.) the fixed point
    ⟺ (A−I) z_bar ≈ b(theta)

    loss_fp   = || A(theta)@z_bar − z_bar − b(theta) ||² / K
    loss_perf = MSE(w·z_bar, Cd) + MSE(w·z_bar, Cl)      [linear perf head]
    loss_ae   = || decode(z_bar) − phi_mean ||² / PHI_DIM  [AE regularisation]
    loss      = loss_fp + λ_perf·loss_perf + λ_ae·loss_ae

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFERENCE MODE  (frozen weights, optimise theta — inverse design)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. solve_fixed_point(theta):  solve (A(theta)−I) z* = b(theta) → z*
2. Cd, Cl = perf_head(z*)
3. Gradient ∂Cd/∂theta is available for latent refinement (Stage 6b).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS UTILITIES  (called after training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
get_eigenspectrum(theta)  →  complex eigenvalues of A(theta)
                             → Strouhal validation: St = Im(λ)·L / (2π·U∞)
condition_number(theta)   →  κ(A(theta)) — triple-duty diagnostic:
                             flow regime / HHL quantum-readiness / gradient trust
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman.lifting  import LiftingNet, DecoderNet
from koopman.operator import ParametricOperator


class ParaKoopModel(nn.Module):
    """
    End-to-end parametric Koopman model.

    Parameters
    ----------
    phi_dim      : dimension of flow observable vector (default 13824 = 3456×4)
    theta_dim    : dimension of geometry parameter vector (default 8, AhmedML)
    koopman_dim  : dimension of Koopman state space K
    operator_rank: rank r of the low-rank A(theta) decomposition
    hidden_lift  : hidden width of lifting/decoding MLPs
    hidden_op    : hidden width of operator alpha/bias MLPs
    lambda_perf  : weight for Cd/Cl supervision loss
    lambda_ae    : weight for autoencoder reconstruction loss
    """

    def __init__(
        self,
        phi_dim:       int   = 13_824,
        theta_dim:     int   = 8,
        koopman_dim:   int   = 256,
        operator_rank: int   = 32,
        hidden_lift:   int   = 512,
        hidden_op:     int   = 64,
        lambda_perf:   float = 1.0,
        lambda_ae:     float = 0.1,
        lambda_fp_cd:  float = 0.1,
    ):
        super().__init__()
        self.koopman_dim  = koopman_dim
        self.phi_dim      = phi_dim
        self.lambda_perf  = lambda_perf
        self.lambda_ae    = lambda_ae
        self.lambda_fp_cd = lambda_fp_cd   # fixed-point penalty weight for Cd-only mode

        self.lifter   = LiftingNet(phi_dim, koopman_dim, hidden_lift)
        self.decoder  = DecoderNet(koopman_dim, phi_dim, hidden_lift)
        self.operator = ParametricOperator(theta_dim, koopman_dim, operator_rank, hidden_op)

        # Linear performance head: z → (Cd, Cl)
        self.perf_head = nn.Linear(koopman_dim, 2, bias=True)

        # theta → z* network for Cd-only training (DrivAerNet Phase 1).
        # Mirrors the role of lift(phi) in AhmedML training — provides a
        # direct fixed-point estimate that the fp-residual loss then anchors
        # to A(theta).  Not used in the AhmedML forward pass.
        self.z_net = nn.Sequential(
            nn.Linear(theta_dim, hidden_op * 2),
            nn.GELU(),
            nn.Linear(hidden_op * 2, hidden_op * 2),
            nn.GELU(),
            nn.Linear(hidden_op * 2, koopman_dim),
        )

    # ─────────────────────────────────────────────────────────────────
    # Training forward pass
    # ─────────────────────────────────────────────────────────────────

    def forward(self, theta: torch.Tensor, phi: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        theta : (B, theta_dim)
        phi   : (B, phi_dim)

        Returns
        -------
        dict with keys:
            z_bar    : (B, K)    lifted Koopman state
            cd_pred  : (B,)      predicted drag coefficient
            cl_pred  : (B,)      predicted lift coefficient
            loss_fp  : scalar    fixed-point residual loss
            loss_ae  : scalar    autoencoder reconstruction loss
            residual : (B, K)    A@z − z − b  (for debugging)
        """
        # Lift mean-flow observables to Koopman state
        z_bar = self.lifter(phi)                      # (B, K)

        # Parametric operator: Az = A(theta)@z,  b = b(theta)
        Az_bar, b = self.operator(theta, z_bar)       # both (B, K)

        # Fixed-point residual (the core training signal)
        residual = Az_bar - z_bar - b                 # (B, K)
        loss_fp  = (residual ** 2).mean()

        # Performance prediction from lifted state
        perf    = self.perf_head(z_bar)               # (B, 2)
        cd_pred = perf[:, 0]
        cl_pred = perf[:, 1]

        # Autoencoder reconstruction (regularisation)
        phi_hat = self.decoder(z_bar)                 # (B, phi_dim)
        loss_ae = F.mse_loss(phi_hat, phi)

        return dict(
            z_bar    = z_bar,
            cd_pred  = cd_pred,
            cl_pred  = cl_pred,
            loss_fp  = loss_fp,
            loss_ae  = loss_ae,
            residual = residual,
        )

    def compute_loss(
        self,
        out:     dict,
        cd_true: torch.Tensor,
        cl_true: torch.Tensor,
    ) -> tuple:
        """
        Combine all loss terms.

        Returns
        -------
        (total_loss, loss_fp, loss_perf, loss_ae)
        """
        loss_perf = (
            F.mse_loss(out['cd_pred'], cd_true) +
            F.mse_loss(out['cl_pred'], cl_true)
        )
        total = (out['loss_fp']
                 + self.lambda_perf * loss_perf
                 + self.lambda_ae   * out['loss_ae'])
        return total, out['loss_fp'], loss_perf, out['loss_ae']

    # ─────────────────────────────────────────────────────────────────
    # Analysis utilities  (no_grad, for post-training use)
    # ─────────────────────────────────────────────────────────────────

    def forward_cd_only(self, theta: torch.Tensor) -> tuple:
        """
        DrivAerNet Phase 1: theta → (cd_pred, loss_fp) using direct z* parameterization.

        z_net(theta) provides z* directly (no linear solve — avoids near-singular
        blow-up when A ≈ I at init). The fixed-point residual loss then trains
        A(theta) to be consistent with z*(theta), forcing the operator to develop
        real geometry-dependent structure.

        Used by all three data sources in unified training.
        Masked Cl loss handles DrivAerNet (no Cl) vs AhmedML (has Cl) split.

        theta  : (B, theta_dim)
        Returns: (cd_pred: (B,), cl_pred: (B,), loss_fp: scalar)
        """
        z_star  = self.z_net(theta)                   # (B, K)
        Az, b   = self.operator(theta, z_star)        # both (B, K)
        loss_fp = ((Az - z_star - b) ** 2).mean()    # fixed-point residual
        perf    = self.perf_head(z_star)              # (B, 2)
        return perf[:, 0], perf[:, 1], loss_fp        # cd_pred, cl_pred, loss_fp

    @torch.no_grad()
    def get_operator_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Materialise A(theta) as a full (K, K) matrix.
        Used for eigenanalysis and HHL preparation.

        theta : (theta_dim,) or (1, theta_dim)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        return self.operator.get_full_matrix(theta)

    @torch.no_grad()
    def get_eigenspectrum(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Eigenvalues of A(theta) — complex tensor of shape (K,).

        Strouhal validation:
            St = Im(lambda_k) * L_ref / (2 * pi * U_inf)
        Compare against Ahmed-body literature:
            bubble pumping  St ~ 0.07  (Grandemange 2013)
            shedding        St ~ 0.13–0.17  (Evstafyeva 2017)

        Computed on CPU — torch.linalg.eigvals is not implemented on MPS.
        """
        A = self.get_operator_matrix(theta).cpu()
        return torch.linalg.eigvals(A)

    @torch.no_grad()
    def condition_number(self, theta: torch.Tensor) -> float:
        """
        κ(A(theta))  — triple-duty diagnostic:
          • flow regime characterisation
          • HHL quantum-solver speedup bound  O(κ² log N)
          • gradient-refinement trust indicator (smooth landscape ↔ small κ)

        Computed on CPU — torch.linalg.cond is not implemented on MPS.
        """
        A = self.get_operator_matrix(theta).cpu()
        return float(torch.linalg.cond(A).item())

    @torch.no_grad()
    def solve_fixed_point(self, theta: torch.Tensor, reg_eps: float = 1e-4) -> torch.Tensor:
        """
        Solve (A(theta) − I + eps·I) z* = b(theta) for the fixed point z*.

        Used in inference / inverse design — NOT called during training
        (training uses z_bar = lift(phi_mean) as the fixed-point proxy).
        reg_eps matches the small value used at the end of fit_drivaernet
        annealing, keeping training and inference consistent.

        theta : (theta_dim,) or (1, theta_dim)
        Returns: z* : (K,)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        K   = self.koopman_dim
        A   = self.get_operator_matrix(theta)                      # (K, K)
        b   = self.operator.get_bias(theta)                        # (1, K)
        AmI = A - torch.eye(K, device=A.device) + reg_eps * torch.eye(K, device=A.device)
        z_star = torch.linalg.solve(AmI, b.T)                     # (K, 1)
        return z_star.squeeze(-1)                                  # (K,)

    @torch.no_grad()
    def predict_performance(self, theta: torch.Tensor) -> dict:
        """
        Pure geometry → performance prediction (no phi needed).

        Uses z_net(theta) as the fixed-point estimate — consistent with
        DrivAerNet Phase 1 training. This is the inference path for
        inverse design when no flow field data is available.

        theta : (theta_dim,) or (1, theta_dim)
        Returns: {'Cd': float, 'Cl': float, 'z_star': Tensor(K,)}
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        cd, cl, _ = self.forward_cd_only(theta)
        z_star    = self.z_net(theta).squeeze(0)   # (K,) for downstream use
        return {
            'Cd':     cd.squeeze(0).item(),
            'Cl':     cl.squeeze(0).item(),
            'z_star': z_star,
        }

    def param_count(self) -> dict:
        """Return parameter counts by submodule."""
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            'lifter':    n(self.lifter),
            'decoder':   n(self.decoder),
            'operator':  n(self.operator),
            'perf_head': n(self.perf_head),
            'z_net':     n(self.z_net),
            'total':     n(self),
        }
