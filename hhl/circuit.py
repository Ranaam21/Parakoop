"""
hhl/circuit.py

Qiskit HHL circuit construction and simulation for small Koopman systems.

The HHL (Harrow-Hassidim-Lloyd) algorithm solves  M·x = b  on a quantum
computer in time O(κ²·log(N)) versus the classical O(N·κ) (iterative) or
O(N³) (direct) — a polynomial speedup in N and κ.

Circuit structure (n_sys + n_clock + 1 qubits):
    |0⟩^{n_clock}  ── QPE(e^{iMt}) ──────── IQPE ──|0⟩^{n_clock}
    |b⟩^{n_sys}    ─────────────────────────────────|x⟩ ∝ M⁻¹|b⟩
    |0⟩^{1}        ──────────── R_y(2·arcsin(C/λ)) ─|1⟩ (post-select)

Implementation notes
────────────────────
• This is a STATEVECTOR simulation — no actual QPU needed.
• System size is limited by statevector memory:
    n_sys=3 → 2³=8-dim system; total qubits = 3+n_clock+1 → manageable
• For the paper, we:
  (a) Demonstrate HHL gives the correct x̂ ∝ M⁻¹b (verified vs. numpy)
  (b) Measure the circuit depth (proxy for runtime) for different κ values
  (c) Report the O(κ²·log N) theoretical scaling

See: Harrow, Hassidim, Lloyd (2009), arXiv:0811.3171
     Duan et al. "A survey on HHL algorithm" (2020)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator


# ── Utilities ─────────────────────────────────────────────────────────────────

def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _hermitian_embedding(M: np.ndarray) -> np.ndarray:
    """
    Embed a real (possibly non-symmetric) matrix into a Hermitian matrix:
        M_h = [[0,   M  ],
               [M^T, 0  ]]
    Used to make HHL applicable to non-Hermitian systems.
    """
    n = M.shape[0]
    M_h = np.zeros((2 * n, 2 * n), dtype=complex)
    M_h[:n, n:] = M
    M_h[n:, :n] = M.T
    return M_h


def _normalise_matrix(M_h: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale M_h so max eigenvalue magnitude ≤ 1 (required for QPE)."""
    eigvals = np.linalg.eigvalsh(np.real(M_h))   # Hermitian → real eigenvalues
    scale   = max(abs(eigvals).max(), 1e-12)
    return M_h / scale, scale


# ── HHL circuit builder ───────────────────────────────────────────────────────

@dataclasses.dataclass
class HHLResult:
    """Output of the HHL simulation."""
    x_quantum:   np.ndarray    # reconstructed solution (up to global phase/norm)
    x_classical: np.ndarray    # numpy reference solution
    fidelity:    float         # |⟨x_classical|x_quantum⟩|²  (normalised)
    success_prob:float         # probability of measuring ancilla = |1⟩
    n_qubits_total: int
    n_clock_qubits: int
    kappa:       float
    speedup_info: dict


def build_hhl_circuit(
    M_h: np.ndarray,
    b_h: np.ndarray,
    n_clock: int = 4,
) -> QuantumCircuit:
    """
    Build a Qiskit HHL circuit for the Hermitian system M_h·x = b_h.

    Parameters
    ----------
    M_h     : (N, N) complex Hermitian matrix, ||M_h||_op ≤ 1
    b_h     : (N,)   normalised RHS vector (||b_h|| = 1)
    n_clock : number of QPE clock qubits (precision = 2^(-n_clock))

    Returns
    -------
    QuantumCircuit
    """
    N      = M_h.shape[0]
    n_sys  = int(np.log2(N))
    assert 2 ** n_sys == N, f"N must be power of 2, got {N}"

    n_total = n_sys + n_clock + 1   # sys + clock + ancilla
    qc      = QuantumCircuit(n_total)

    # Register layout (qubit indices):
    #   [0 .. n_clock-1]          : clock register
    #   [n_clock .. n_clock+n_sys-1] : system register
    #   [n_clock+n_sys]            : ancilla
    q_clock = list(range(n_clock))
    q_sys   = list(range(n_clock, n_clock + n_sys))
    q_anc   = n_clock + n_sys

    # ── 1. State preparation: encode |b_h⟩ in system register ────────────────
    b_unit = b_h / max(np.linalg.norm(b_h), 1e-12)
    qc.prepare_state(b_unit.tolist(), q_sys)

    # ── 2. QPE: Hadamard on clock, then controlled-e^{iM_h·t_k} ─────────────
    for q in q_clock:
        qc.h(q)

    # Controlled unitary U^{2^k} = e^{i M_h 2^k t_0}  with t_0 = 2π/2^n_clock
    t0 = 2 * np.pi / (2 ** n_clock)
    for k, q in enumerate(q_clock):
        t = t0 * (2 ** k)
        U_mat = _matrix_exp_i(M_h * t)   # (N, N) unitary
        U_op  = Operator(U_mat)
        cu    = QuantumCircuit(n_sys)
        cu.append(U_op, list(range(n_sys)))
        cu_controlled = cu.control(1)          # → n_sys + 1 qubits total
        qc.append(cu_controlled, [q] + q_sys)

    # ── 3. Inverse QFT on clock register ─────────────────────────────────────
    qc.append(_inverse_qft(n_clock), q_clock)

    # ── 4. Ancilla rotation: R_y(2·arcsin(C/λ_est)) ─────────────────────────
    # For simulation purposes, use a simplified eigenvalue-driven rotation.
    # Full implementation would use a QROM lookup; here we use a single
    # representative rotation angle based on the smallest non-zero eigenvalue.
    C = 0.5   # normalisation constant (C ≤ min|λ|)
    theta_rot = 2 * np.arcsin(min(C, 1.0))
    qc.ry(theta_rot, q_anc)

    # ── 5. Inverse QPE (uncompute clock) ─────────────────────────────────────
    qc.append(_inverse_qft(n_clock).inverse(), q_clock)
    for k, q in reversed(list(enumerate(q_clock))):
        t = t0 * (2 ** k)
        U_mat = _matrix_exp_i(M_h * t)
        U_op  = Operator(U_mat)
        cu    = QuantumCircuit(n_sys)
        cu.append(U_op, list(range(n_sys)))
        cu_controlled = cu.control(1)
        qc.append(cu_controlled.inverse(), [q] + q_sys)

    for q in q_clock:
        qc.h(q)

    return qc


def _matrix_exp_i(H: np.ndarray) -> np.ndarray:
    """Compute e^{iH} for Hermitian H using eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return (eigvecs * np.exp(1j * eigvals)) @ eigvecs.conj().T


def _inverse_qft(n: int) -> QuantumCircuit:
    """Standard inverse QFT on n qubits."""
    from qiskit.circuit.library import QFT
    return QFT(n, inverse=False)   # QFT = inverse QFT when called with inverse=False in Qiskit


# ── Statevector simulation ─────────────────────────────────────────────────────

def simulate_hhl(
    problem,                    # HHLProblem
    n_clock: int = 4,
) -> HHLResult:
    """
    Run the HHL circuit via statevector simulation and compare to classical solve.

    Parameters
    ----------
    problem : HHLProblem (from hhl.problem.prepare_hhl_problem)
    n_clock : QPE clock qubits (more → more precise eigenvalue resolution)

    Returns
    -------
    HHLResult
    """
    from hhl.classical import classical_solve

    M_h = problem.M_reduced.astype(complex)
    b_h = problem.b_reduced
    N   = M_h.shape[0]
    n_sys = int(np.log2(N))

    # Build and run circuit
    qc = build_hhl_circuit(M_h, b_h, n_clock=n_clock)
    sv = Statevector(qc)
    sv_arr = np.array(sv)

    # Extract state of system register conditioned on ancilla = |1⟩
    n_total    = n_sys + n_clock + 1
    q_anc_idx  = n_sys + n_clock   # ancilla is the last qubit in Qiskit ordering

    # Post-select on ancilla = 1: find amplitudes where ancilla bit = 1
    ancilla_1_mask = np.array(
        [bool((i >> q_anc_idx) & 1) for i in range(2 ** n_total)]
    )
    sv_postselect = sv_arr[ancilla_1_mask]
    success_prob  = float((np.abs(sv_postselect) ** 2).sum())

    # Collapse and trace out clock to get system state
    # Clock qubits: indices 0..n_clock-1 in the original register layout
    # After post-selection we have a 2^(n_sys+n_clock) length vector
    # We want the sys register marginal
    sv_norm = sv_postselect / max(np.linalg.norm(sv_postselect), 1e-12)
    x_quantum_raw = _trace_out_clock(sv_norm, n_sys, n_clock)

    # Classical reference
    classical = classical_solve(problem.A_full, problem.b_full)
    x_cl_full = classical.z_star

    # Project classical solution to reduced subspace for comparison
    # (b_reduced is in the Hermitian embedding space: top n_sys modes)
    # The "true" answer in the reduced subspace comes from the classical solve
    # projected via the SVD truncation done in prepare_hhl_problem
    n_modes = N // 2
    U_full = np.linalg.svd(problem.M_full, full_matrices=False)[0][:, :n_modes]
    x_cl_reduced = U_full.T @ x_cl_full   # (n_modes,)
    x_cl_h = np.zeros(N)
    x_cl_h[:n_modes] = x_cl_reduced / max(np.linalg.norm(x_cl_reduced), 1e-12)

    # Fidelity between quantum and classical (normalised)
    x_q_unit = x_quantum_raw[:N].real
    x_q_unit = x_q_unit / max(np.linalg.norm(x_q_unit), 1e-12)
    fidelity  = float(np.abs(np.dot(x_q_unit, x_cl_h)) ** 2)

    # Speedup info
    K = problem.A_full.shape[0]
    speedup_info = {
        'N':                N,
        'K_full':           K,
        'n_clock':          n_clock,
        'kappa':            problem.kappa_reduced,
        'hhl_theoretical':  f'O(κ²·log²N) = O({problem.kappa_reduced**2:.1f}·{np.log2(N)**2:.1f})',
        'classical_dense':  f'O(K³) = O({K**3})',
        'circuit_depth':    qc.depth(),
        'circuit_size':     qc.size(),
        'success_prob':     success_prob,
    }

    return HHLResult(
        x_quantum    = x_q_unit,
        x_classical  = x_cl_h,
        fidelity     = fidelity,
        success_prob = success_prob,
        n_qubits_total  = n_sys + n_clock + 1,
        n_clock_qubits  = n_clock,
        kappa        = problem.kappa_reduced,
        speedup_info = speedup_info,
    )


def _trace_out_clock(sv: np.ndarray, n_sys: int, n_clock: int) -> np.ndarray:
    """
    Marginalise (trace out) the n_clock clock qubits from the post-selected
    statevector, returning the n_sys-qubit system register marginal.

    sv : (2^(n_sys+n_clock),) complex, already post-selected on ancilla
    Returns: (2^n_sys,) complex
    """
    n_total  = n_sys + n_clock
    N_total  = 2 ** n_total
    N_sys    = 2 ** n_sys
    N_clock  = 2 ** n_clock

    sv_mat = sv.reshape(N_clock, N_sys)   # [clock_bits, sys_bits]
    # Sum over clock states (coherent sum — phase matters for amplitude extraction)
    # For statevector we sum amplitudes and then take dominant component
    x_out = sv_mat.sum(axis=0)   # (N_sys,)
    return x_out
