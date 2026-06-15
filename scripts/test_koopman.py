"""
scripts/test_koopman.py

Self-test for the ParaKoop Koopman module (no real data needed).

Tests:
    1. Model instantiation and parameter count
    2. Forward pass with synthetic data — loss terms all finite
    3. Eigenspectrum extraction from A(theta)
    4. Condition number κ(A(theta))
    5. Fixed-point solver (A−I)z* = b
    6. predict_performance() — pure theta → Cd/Cl inference
    7. Strouhal proxy computation from eigenvalues

Run from parakoop/:
    PYTHONPATH=/Users/amit21/Desktop/Car_CFD/parakoop python scripts/test_koopman.py
"""

import sys
import torch
import numpy as np

sys.path.insert(0, '/Users/amit21/Desktop/Car_CFD/parakoop')

from koopman import ParaKoopModel, KoopmanTrainer, TrainerConfig

# ── Configuration ─────────────────────────────────────────────────────────────
PHI_DIM      = 13_824   # 3456 probes × 4 fields (as in AhmedML loader)
THETA_DIM    = 8        # AhmedML geometry params
KOOPMAN_DIM  = 256
OPERATOR_RANK = 32
BATCH_SIZE   = 4
L_REF        = 1.044    # Ahmed body length (m) — Strouhal reference
U_INF        = 40.0     # m/s freestream (AhmedML default)

print("=" * 60)
print("ParaKoop Koopman Module — Self Test")
print("=" * 60)

# ── Test 1: Instantiation ─────────────────────────────────────────────────────
print("\n[1] Model instantiation ...")
model = ParaKoopModel(
    phi_dim       = PHI_DIM,
    theta_dim     = THETA_DIM,
    koopman_dim   = KOOPMAN_DIM,
    operator_rank = OPERATOR_RANK,
    hidden_lift   = 256,   # smaller for the test
    hidden_op     = 64,
    lambda_perf   = 1.0,
    lambda_ae     = 0.1,
)
counts = model.param_count()
print(f"  Total params  : {counts['total']:,}")
print(f"  Lifter        : {counts['lifter']:,}")
print(f"  Operator      : {counts['operator']:,}")
print(f"  Decoder       : {counts['decoder']:,}")
print(f"  Perf head     : {counts['perf_head']:,}")

# ── Test 2: Forward pass with synthetic data ───────────────────────────────────
print("\n[2] Forward pass (synthetic data) ...")
torch.manual_seed(0)
theta_fake = torch.randn(BATCH_SIZE, THETA_DIM)       # normalised geometry params
phi_fake   = torch.randn(BATCH_SIZE, PHI_DIM) * 0.3  # flow observables
cd_fake    = torch.rand(BATCH_SIZE) * 0.2 + 0.2      # Cd ~ [0.2, 0.4]
cl_fake    = torch.rand(BATCH_SIZE) * 0.1 - 0.05     # Cl ~ [-0.05, 0.05]

out = model(theta_fake, phi_fake)
total, l_fp, l_perf, l_ae = model.compute_loss(out, cd_fake, cl_fake)

print(f"  loss_total : {total.item():.6f}")
print(f"  loss_fp    : {l_fp.item():.6f}   (fixed-point residual)")
print(f"  loss_perf  : {l_perf.item():.6f}  (Cd/Cl prediction)")
print(f"  loss_ae    : {l_ae.item():.6f}   (autoencoder reconstruction)")
print(f"  cd_pred    : {out['cd_pred'].detach().numpy().round(4)}")
print(f"  cd_true    : {cd_fake.numpy().round(4)}")
print(f"  residual norm (mean over batch): {out['residual'].norm(dim=-1).mean().item():.4f}")

assert all(torch.isfinite(t) for t in [total, l_fp, l_perf, l_ae]), "NaN/Inf in losses!"
print("  ✓ all losses finite")

# ── Test 3: Backward pass (gradient flow) ─────────────────────────────────────
print("\n[3] Backward pass ...")
total.backward()
# z_net is only used in forward_cd_only path (not phi forward) — exclude from check
z_net_params = set(id(p) for p in model.z_net.parameters())
grads_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.parameters() if p.requires_grad and id(p) not in z_net_params
)
assert grads_ok, "Some gradients are None or NaN in phi-path parameters!"

# Also verify z_net gradients flow in the cd-only path
model.zero_grad()
cd_p, cl_p, fp = model.forward_cd_only(theta_fake)
(cd_p.mean() + cl_p.mean() + fp).backward()
z_net_grads_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.z_net.parameters()
)
assert z_net_grads_ok, "z_net gradients missing in forward_cd_only path!"
print("  ✓ phi-path gradients finite  |  z_net cd-only gradients finite")
model.zero_grad()

# ── Test 4: Eigenspectrum of A(theta) ─────────────────────────────────────────
print("\n[4] Eigenspectrum extraction ...")
theta_single = theta_fake[0]   # (theta_dim,)
eigvals = model.get_eigenspectrum(theta_single)   # (K,) complex
print(f"  Eigenvalue count : {len(eigvals)}")
print(f"  |λ| range        : {eigvals.abs().min():.4f} → {eigvals.abs().max():.4f}")
print(f"  Im(λ) range      : {eigvals.imag.min():.4f} → {eigvals.imag.max():.4f}")

# Strouhal proxy: St = Im(λ) * L_ref / (2π * U_inf)
# Keep only positive imaginary parts (positive frequency modes)
pos_imag = eigvals.imag[eigvals.imag > 0]
if len(pos_imag) > 0:
    St_vals = (pos_imag * L_REF / (2 * np.pi * U_INF)).numpy()
    print(f"  St (from Im(λ))  : min={St_vals.min():.4f}  max={St_vals.max():.4f}  "
          f"  (literature: bubble~0.07, shedding~0.13-0.17)")
    print(f"  [Note: St validation meaningful only after training on AhmedML data]")
print("  ✓ eigenspectrum extracted")

# ── Test 5: Condition number κ(A(theta)) ─────────────────────────────────────
print("\n[5] Condition number κ(A(theta)) ...")
kappa = model.condition_number(theta_single)
print(f"  κ(A) = {kappa:.4f}")
print(f"  [Note: untrained model — κ should approach 1.0 due to near-identity init]")
assert np.isfinite(kappa), "κ is NaN!"
print("  ✓ condition number finite")

# ── Test 6: Fixed-point solver ────────────────────────────────────────────────
print("\n[6] Fixed-point solver (A−I)z* = b ...")
z_star = model.solve_fixed_point(theta_single)   # (K,)
print(f"  z* shape : {z_star.shape}")
print(f"  z* norm  : {z_star.norm().item():.4f}")

# Verify: (A−I)z* ≈ b
A   = model.get_operator_matrix(theta_single)
b   = model.operator.get_bias(theta_single.unsqueeze(0)).squeeze(0)
res = (A - torch.eye(KOOPMAN_DIM)) @ z_star - b
print(f"  Residual ||(A−I)z*−b|| : {res.norm().item():.6f}")
print("  ✓ fixed-point solver ran without error")

# ── Test 7: Pure inference (theta → Cd, Cl) ───────────────────────────────────
print("\n[7] Performance inference (theta → Cd, Cl) ...")
perf = model.predict_performance(theta_single)
print(f"  Cd_pred = {perf['Cd']:.4f}   (from fixed-point z*)")
print(f"  Cl_pred = {perf['Cl']:.4f}")
print(f"  z* norm = {perf['z_star'].norm().item():.4f}")
print("  ✓ pure geometry → performance inference works")

# ── Test 8: Operator matrix shape ─────────────────────────────────────────────
print("\n[8] Operator matrix shape ...")
A_mat = model.get_operator_matrix(theta_single)
print(f"  A shape     : {tuple(A_mat.shape)}  (expect [{KOOPMAN_DIM}, {KOOPMAN_DIM}])")
print(f"  A off-diag  : {(A_mat - torch.eye(KOOPMAN_DIM)).abs().max().item():.6f}  "
      f"(small for untrained model)")
assert A_mat.shape == (KOOPMAN_DIM, KOOPMAN_DIM)
print("  ✓ operator matrix shape correct")

print("\n" + "=" * 60)
print("All 8 tests passed. Koopman module is ready.")
print("Next step: train on AhmedML data via KoopmanTrainer.fit()")
print("=" * 60)
