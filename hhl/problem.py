"""
hhl/problem.py

Prepare the HHL linear-system problem from a trained ParaKoop model.

The Koopman fixed-point equation:
    (A(theta) − I) · z* = b(theta)

is exactly the linear system  M · z* = b  that HHL solves, where:
    M = A(theta) − I    ∈ R^{K×K}   (real, generally non-symmetric)
    b = b(theta)        ∈ R^K

For the quantum circuit, M must be:
    1. Hermitian      — use the standard block-encoding:
       M_h = [[0, M], [M^T, 0]]  ∈ R^{2K × 2K}
    2. Size = 2^n    — project to the top-n_qubits_dim principal subspace
                       (captures most of the dynamics via SVD truncation)
    3. Normalised     — scale so ||M_h|| ≤ 1

The projection is physically meaningful: the top singular vectors of M
correspond to the dominant spatial flow modes, and the reduced system
captures the physics in that subspace.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import torch


@dataclasses.dataclass
class HHLProblem:
    """
    A prepared linear system  M_h · x = b_enc  ready for HHL.

    Attributes
    ----------
    A_full     : (K, K) full Koopman operator A(theta)
    M_full     : (K, K) system matrix = A - I
    b_full     : (K,)   RHS bias b(theta)
    M_reduced  : (N, N) hermitian embedding, projected to N=2^n subspace
    b_reduced  : (N,)   RHS in the reduced subspace (normalised)
    b_norm     : float  ||b_full|| (to rescale solution back)
    singular_vals : (n_modes,) kept singular values of M_full (for energy check)
    kappa      : float  condition number of M_full
    kappa_reduced : float  condition number of M_reduced
    n_qubits   : int    log2(N) — number of qubits for the system register
    theta      : np.ndarray (K,)  geometry parameter vector used
    """
    A_full:        np.ndarray
    M_full:        np.ndarray
    b_full:        np.ndarray
    M_reduced:     np.ndarray
    b_reduced:     np.ndarray
    b_norm:        float
    singular_vals: np.ndarray
    kappa:         float
    kappa_reduced: float
    n_qubits:      int
    theta:         np.ndarray

    @property
    def N(self) -> int:
        return self.M_reduced.shape[0]

    def energy_fraction(self) -> float:
        """Fraction of M's spectral energy captured by the reduced system."""
        s_all = np.linalg.svd(self.M_full, compute_uv=False)
        captured = float((self.singular_vals ** 2).sum())
        total    = float((s_all ** 2).sum())
        return captured / total if total > 0 else 0.0

    def hhl_speedup_bound(self) -> dict:
        """
        Theoretical HHL speedup metrics.

        Classical: O(N³)  or O(N·s·κ²·log(1/ε)) for sparse A
        HHL:       O(log(N)·s·κ²·poly(log(1/ε)))  [s = sparsity]

        Speedup in log(N) factor: N / log2(N)  (dimension advantage)
        """
        N = self.N
        k = self.kappa_reduced
        log_speedup = N / np.log2(N)   # naive dimension advantage
        return {
            'N':              N,
            'kappa_full':     self.kappa,
            'kappa_reduced':  self.kappa_reduced,
            'log_N':          np.log2(N),
            'hhl_complexity': f'O(κ²·log²N) = O({k**2:.1f}·{np.log2(N)**2:.1f})',
            'classical_complexity': f'O(N³) = O({N**3})',
            'dimension_speedup': f'~{log_speedup:.1f}x  [N/log₂N = {N}/{np.log2(N):.1f}]',
        }


def prepare_hhl_problem(
    model,
    theta: torch.Tensor,
    n_qubits: int = 3,
) -> HHLProblem:
    """
    Extract the linear system from a trained ParaKoopModel for a single geometry.

    Steps:
        1. Materialise A(theta) and b(theta) from trained model
        2. Form M = A − I
        3. SVD-project M to N = 2^n_qubits principal modes
        4. Build Hermitian embedding M_h for HHL input
        5. Normalise: scale so ||M_h||_op ≤ 1

    Parameters
    ----------
    model    : trained ParaKoopModel
    theta    : (theta_dim,) geometry parameter vector (normalised)
    n_qubits : number of system-register qubits → N = 2^n_qubits modes
                (3 → N=8, 4 → N=16)  [keep ≤ 4 for statevector simulation]

    Returns
    -------
    HHLProblem
    """
    N = 2 ** n_qubits

    # ── Step 1: Get A(theta) and b(theta) from model ──────────────────────────
    model.eval()
    with torch.no_grad():
        if theta.dim() == 1:
            theta_in = theta.unsqueeze(0)
        A_full = model.get_operator_matrix(theta_in).numpy().astype(np.float64)
        b_full = model.operator.get_bias(theta_in).squeeze(0).numpy().astype(np.float64)

    # ── Step 2: System matrix M = A − I ──────────────────────────────────────
    K = A_full.shape[0]
    M_full = A_full - np.eye(K)

    # κ of full M (may be large near fixed point, which is fine — just reported)
    sv_full = np.linalg.svd(M_full, compute_uv=False)
    kappa_full = float(sv_full[0] / max(sv_full[-1], 1e-12))

    # ── Step 3: SVD projection to top N/2 principal modes ─────────────────────
    # We project to the top (N/2) left and right singular vectors.
    # The Hermitian block structure doubles the size: [[0, M_r], [M_r^T, 0]].
    n_modes = N // 2   # M_reduced will be (n_modes × n_modes), M_h is (N × N)

    U, s, Vt = np.linalg.svd(M_full, full_matrices=False)
    U_r  = U[:, :n_modes]         # (K, n_modes) left singular vectors
    Vt_r = Vt[:n_modes, :]        # (n_modes, K) right singular vectors
    M_r  = np.diag(s[:n_modes])   # (n_modes, n_modes) diagonal in SVD basis

    # Project b into the right singular subspace: b_r = Vt_r @ b_full
    b_r      = Vt_r @ b_full      # (n_modes,)
    b_norm   = float(np.linalg.norm(b_full))
    b_r_norm = float(np.linalg.norm(b_r))
    b_r_unit = b_r / max(b_r_norm, 1e-12)   # normalised for state preparation

    # ── Step 4: Hermitian embedding for HHL ───────────────────────────────────
    # Non-symmetric M_r → Hermitian M_h:
    #   M_h = [[0,    M_r  ],    size (N, N)
    #           [M_r^T, 0  ]]
    #
    # RHS encoding:  b_h = [b_r_unit; 0_{n_modes}]   (put b in top half)
    M_h = np.zeros((N, N), dtype=np.float64)
    M_h[:n_modes, n_modes:] = M_r
    M_h[n_modes:, :n_modes] = M_r.T

    b_h = np.zeros(N, dtype=np.float64)
    b_h[:n_modes] = b_r_unit

    # ── Step 5: Normalise M_h so spectral radius ≤ 1 ─────────────────────────
    eigvals = np.linalg.eigvalsh(M_h)
    scale   = max(abs(eigvals).max(), 1e-12)
    M_h_norm = M_h / scale

    kappa_reduced = float(s[0] / max(s[n_modes - 1], 1e-12))

    theta_np = theta.numpy() if hasattr(theta, 'numpy') else np.array(theta)

    return HHLProblem(
        A_full        = A_full,
        M_full        = M_full,
        b_full        = b_full,
        M_reduced     = M_h_norm,
        b_reduced     = b_h,
        b_norm        = b_norm,
        singular_vals = s[:n_modes],
        kappa         = kappa_full,
        kappa_reduced = kappa_reduced,
        n_qubits      = n_qubits,
        theta         = theta_np,
    )
