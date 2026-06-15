"""
hhl/classical.py

Classical baseline solver and κ-analysis for the Koopman linear system.

Two roles:
    1. GROUND TRUTH — solve (A(theta)−I)·z* = b(theta) classically (numpy)
       to verify that the quantum HHL circuit produces the correct answer.

    2. SPEEDUP ANALYSIS — for a trained model, sweep theta across the
       AhmedML/DrivAerNet++ design space and report κ(A(theta)) distributions:
          • by body style (fastback / notchback / estateback)
          • quartile breakdown
          • theoretical HHL speedup at each κ value
       This is the empirical finding the paper reports as the "kappa landscape."

The classical solver is also the practical path when κ is too large for HHL
to be competitive (κ > sqrt(N) roughly). The classifier explicitly flags this.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch


# ── Classical solve ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class ClassicalSolution:
    z_star:  np.ndarray    # (K,) fixed-point solution
    residual: float        # ||(A−I)z* − b||
    kappa:    float        # condition number of M = A − I
    solve_method: str      # 'lstsq' or 'solve'
    hhl_competitive: bool  # True if κ < sqrt(K) (HHL likely beneficial)


def classical_solve(A: np.ndarray, b: np.ndarray) -> ClassicalSolution:
    """
    Solve (A − I) · z* = b  using numpy.

    Parameters
    ----------
    A : (K, K)  Koopman operator matrix
    b : (K,)    bias vector

    Returns
    -------
    ClassicalSolution
    """
    K = A.shape[0]
    M = A - np.eye(K)

    # Condition number of M
    sv = np.linalg.svd(M, compute_uv=False)
    kappa = float(sv[0] / max(sv[-1], 1e-14))

    # Solve
    if kappa < 1e10:
        try:
            z_star = np.linalg.solve(M, b)
            method = 'solve'
        except np.linalg.LinAlgError:
            z_star, *_ = np.linalg.lstsq(M, b, rcond=None)
            method = 'lstsq'
    else:
        z_star, *_ = np.linalg.lstsq(M, b, rcond=None)
        method = 'lstsq'

    residual = float(np.linalg.norm(M @ z_star - b))
    hhl_competitive = kappa < np.sqrt(K)   # rough HHL regime indicator

    return ClassicalSolution(
        z_star=z_star,
        residual=residual,
        kappa=kappa,
        solve_method=method,
        hhl_competitive=hhl_competitive,
    )


# ── κ landscape analysis ──────────────────────────────────────────────────────

@dataclasses.dataclass
class KappaProfile:
    """κ(A(theta)) statistics for a batch of geometries."""
    thetas:     np.ndarray   # (N, theta_dim)
    kappas:     np.ndarray   # (N,) condition numbers
    labels:     list         # N labels (run_id / design_id)
    meta:       dict         # optional metadata per sample

    def summary(self) -> str:
        k = self.kappas
        hhl_frac = float((k < np.sqrt(k.shape[0])).mean())
        return (
            f"κ profile over {len(k)} geometries:\n"
            f"  min={k.min():.2f}  median={np.median(k):.2f}  "
            f"  mean={k.mean():.2f}  max={k.max():.2f}\n"
            f"  p25={np.percentile(k, 25):.2f}  p75={np.percentile(k, 75):.2f}\n"
            f"  HHL-competitive (κ<√K): {hhl_frac*100:.1f}%\n"
            f"  HHL complexity O(κ²logN): "
            f"median={np.median(k)**2 * np.log2(len(k)):.1f}"
        )

    def speedup_table(self, n_bins: int = 5) -> str:
        """
        Print a speedup table: κ range | HHL cost | classical cost | speedup.
        Classical: O(K³) per solve.  HHL: O(κ²·log²K·log(1/ε)).
        """
        k   = self.kappas
        K   = len(k)
        eps = 1e-3   # target precision

        lines = ["  κ range      | HHL O(κ²log²K) | Classical O(K³) | Speedup"]
        lines += ["  " + "-" * 64]

        percentiles = np.percentile(k, np.linspace(0, 100, n_bins + 1))
        for lo, hi in zip(percentiles[:-1], percentiles[1:]):
            mid   = (lo + hi) / 2
            hhl   = mid ** 2 * np.log2(K) ** 2
            cls   = K ** 3
            ratio = cls / hhl
            lines.append(f"  κ~{mid:6.1f}     |  {hhl:12.1f}    |  {cls:12d}    |  {ratio:8.1f}x")

        return "\n".join(lines)


def analyze_kappa_landscape(
    model,
    theta_batch: torch.Tensor,
    labels: Optional[list] = None,
) -> KappaProfile:
    """
    Compute κ(A(theta)−I) for a batch of geometry parameters.

    Parameters
    ----------
    model       : trained ParaKoopModel
    theta_batch : (N, theta_dim) tensor of geometry parameters
    labels      : optional list of N identifiers (run_id / design_id)

    Returns
    -------
    KappaProfile
    """
    model.eval()
    kappas = []

    with torch.no_grad():
        for i in range(theta_batch.shape[0]):
            theta_i = theta_batch[i]
            A = model.get_operator_matrix(theta_i).numpy()
            M = A - np.eye(A.shape[0])
            sv = np.linalg.svd(M, compute_uv=False)
            k  = float(sv[0] / max(sv[-1], 1e-14))
            kappas.append(k)

    kappas_arr = np.array(kappas, dtype=np.float64)
    if labels is None:
        labels = list(range(len(kappas)))

    return KappaProfile(
        thetas = theta_batch.numpy(),
        kappas = kappas_arr,
        labels = labels,
        meta   = {},
    )
