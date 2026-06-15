"""
data_pipeline/ahmedml_loader.py

AhmedML data loader for ParaKoop training.

What AhmedML provides (per run, 500 runs total):
    ahmed_X.stl          3D geometry mesh
    volume_X.vtu         Time-MEAN flow fields: UMean (3D velocity),
                         pMean (pressure), vorticityMean, static(p)_coeffMean
    force_mom_X.csv      Time-mean Cd, Cl
    geo_parameters_X.csv 8 geometry parameters (in mm)

IMPORTANT NOTE — Mean fields, not time snapshots:
    AhmedML stores time-averaged fields (averaged over ~80 convective time
    units of hybrid RANS-LES). Individual time snapshots are NOT stored
    (too large). This means:
    - Parametric Koopman A(theta): ✓ fully compatible — we learn the map
      from geometry parameters to mean flow observables across 500 geometries
    - Temporal Koopman (EDMD on a single run's time series): ✗ not possible
      from this dataset alone
    - Strouhal-number validation: uses SPATIAL vorticity patterns in
      vorticityMean (recirculation structure) rather than temporal frequency
      analysis — this is an approximation; note it explicitly in the paper

Observable extraction strategy:
    Sample the 21.5M-cell VTU at a fixed set of N_PROBES probe points on a
    uniform 3D grid covering the near-wake region. This gives a consistent
    observable vector phi(x) of dimension N_PROBES * N_FIELDS for every run,
    enabling EDMD-style operator fitting across 500 (theta, phi) pairs.

Usage
-----
    from data_pipeline.ahmedml_loader import AhmedMLDataset, load_ahmedml

    ds = load_ahmedml('data/ahmedml/', n_probes=4096)
    sample = ds[0]
    # sample.theta       : np.ndarray (8,)  — normalised geometry params
    # sample.phi         : np.ndarray (N,)  — flow observables
    # sample.cd, sample.cl : float
    # sample.run_id      : int
"""

from __future__ import annotations

import os
import dataclasses
from typing import Optional, List

import numpy as np
import pandas as pd

# ── Probe grid configuration ──────────────────────────────────────────────
# Fixed 3D observation grid in the WAKE REGION behind the car.
# Domain: x=[-4, 6], y=[-1, 1], z=[0, 1.4] (metres, wind-tunnel frame)
# We focus on the near-wake zone: x=[0, 3], y=[-0.6, 0.6], z=[0, 0.6]
# where von Karman vortex shedding and recirculation structures live.
PROBE_GRID = dict(
    x=(-0.5, 3.0, 24),    # (min, max, n_points) — starts slightly behind car rear
    y=(-0.6, 0.6, 12),
    z=( 0.0, 0.6, 12),
)   # total: 24×12×12 = 3456 probes × 4 fields (Ux, Uz, p, Cp) = 13824-dim phi

# Fields to extract from VTU (keep lightweight — skip ReynoldsStress)
FIELDS = ['UMean', 'pMean', 'static(p)_coeffMean']
# UMean is a 3-vector; we keep only x and z components (streamwise + vertical)
# pMean and Cp are scalars
# Resulting phi per probe: Ux, Uz, p, Cp → 4 values
PHI_DIM_PER_PROBE = 4


# ── Geometry parameter mapping ─────────────────────────────────────────────
# AhmedML geo_parameters columns (in mm) → normalised continuous params
# Mapping to CarGeometry-compatible names where possible.
GEO_COLS = [
    'body-length', 'body-height', 'body-width',
    'front-arc-diameter', 'slant-angle-length', 'slant-angle-height',
    'slant-surface-length', 'slant-angle-degrees',
]
MM_TO_M = 1e-3   # AhmedML stores geometry in mm


# ══════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class AhmedSample:
    """One AhmedML run, preprocessed and ready for Koopman training."""
    run_id: int
    theta:  np.ndarray   # (8,) float32 — normalised geometry params
    phi:    np.ndarray   # (N_PROBES * 4,) float32 — flow observables
    cd:     float
    cl:     float
    # Wake recirculation metrics derived from UMean (set by loader)
    bubble_length: float = 0.0    # length of Ux<0 recirculation zone (m)
    wake_deficit:  float = 0.0    # mean velocity deficit in near-wake


# ══════════════════════════════════════════════════════════════════════════
# Probe grid builder
# ══════════════════════════════════════════════════════════════════════════

def build_probe_grid() -> np.ndarray:
    """
    Build the fixed 3D probe array — shape (N_PROBES, 3).
    Same grid for every run, guaranteeing consistent phi dimension.
    """
    xs = np.linspace(*PROBE_GRID['x'])
    ys = np.linspace(*PROBE_GRID['y'])
    zs = np.linspace(*PROBE_GRID['z'])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    probes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    return probes   # (N_PROBES, 3)


PROBES = build_probe_grid()
N_PROBES = len(PROBES)
PHI_DIM  = N_PROBES * PHI_DIM_PER_PROBE


# ══════════════════════════════════════════════════════════════════════════
# Per-run loader
# ══════════════════════════════════════════════════════════════════════════

def load_run(run_dir: str, run_id: int, geo_row: pd.Series,
             theta_mean: np.ndarray, theta_std: np.ndarray) -> Optional[AhmedSample]:
    """
    Load one AhmedML run directory into an AhmedSample.

    Parameters
    ----------
    run_dir     : path to run_X/ directory
    run_id      : integer run number
    geo_row     : geometry parameters row from geo_parameters_all.csv
    theta_mean  : dataset-level mean of geometry params (for normalisation)
    theta_std   : dataset-level std  of geometry params (for normalisation)
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pip install pyvista  (needed for VTU loading)")

    # ── Geometry theta ────────────────────────────────────────────────────
    raw_geo = geo_row[GEO_COLS].values.astype(np.float32)
    raw_geo[:7] *= MM_TO_M           # convert mm → m (first 7 cols are lengths)
    # slant-angle-degrees stays as degrees — already dimensionless
    theta = ((raw_geo - theta_mean) / (theta_std + 1e-8)).astype(np.float32)

    # ── Cd / Cl ───────────────────────────────────────────────────────────
    fm_path = os.path.join(run_dir, f'force_mom_{run_id}.csv')
    if not os.path.exists(fm_path):
        return None
    fm = pd.read_csv(fm_path)
    fm.columns = fm.columns.str.strip()
    cd = float(fm['cd'].iloc[0])
    cl = float(fm['cl'].iloc[0])

    # ── VTU flow field → phi observable vector ────────────────────────────
    vtu_path = os.path.join(run_dir, f'volume_{run_id}.vtu')
    if not os.path.exists(vtu_path):
        return None

    mesh = pv.read(vtu_path)

    # Interpolate fields at fixed probe locations
    probe_mesh = pv.PolyData(PROBES)
    sampled    = probe_mesh.interpolate(mesh, radius=0.05, n_points=4)

    # Extract fields → shape (N_PROBES, PHI_DIM_PER_PROBE)
    U  = sampled['UMean'].astype(np.float32)           # (N, 3)
    p  = sampled['pMean'].astype(np.float32)           # (N,)
    cp = sampled['static(p)_coeffMean'].astype(np.float32)  # (N,)

    Ux = U[:, 0]   # streamwise velocity component
    Uz = U[:, 2]   # vertical velocity component

    phi = np.stack([Ux, Uz, p, cp], axis=1).ravel().astype(np.float32)  # (N*4,)

    # ── Wake recirculation metrics (from full mesh, x > 0 region) ─────────
    # UMean is CELL data — use cell centres for spatial filtering
    full_U   = mesh['UMean']                           # (n_cells, 3)
    cell_pts = np.array(mesh.cell_centers().points)    # (n_cells, 3)

    in_wake = (cell_pts[:, 0] > 0) & (cell_pts[:, 0] < 3.0) & \
              (np.abs(cell_pts[:, 1]) < 0.3) & (cell_pts[:, 2] < 0.5)
    wake_Ux = full_U[in_wake, 0] if in_wake.any() else np.array([0.0])

    recirculating = in_wake & (full_U[:, 0] < 0)
    bubble_length = float(cell_pts[recirculating, 0].max(initial=0.0)) \
                    if recirculating.any() else 0.0
    wake_deficit  = float(np.maximum(-wake_Ux, 0).mean()) if len(wake_Ux) > 0 else 0.0

    return AhmedSample(
        run_id=run_id,
        theta=theta,
        phi=phi,
        cd=cd,
        cl=cl,
        bubble_length=bubble_length,
        wake_deficit=wake_deficit,
    )


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class AhmedMLDataset:
    """
    In-memory dataset of AhmedML samples for Koopman training.

    Provides:
        __len__, __getitem__  — standard dataset interface
        theta_dim             — dimension of geometry parameter vector (8)
        phi_dim               — dimension of flow observable vector (N_PROBES * 4)
        get_arrays()          — return (Theta, Phi, Cd, Cl) as numpy arrays
    """

    def __init__(self, samples: List[AhmedSample]):
        self.samples   = samples
        self.theta_dim = len(samples[0].theta) if samples else 8
        self.phi_dim   = len(samples[0].phi)   if samples else PHI_DIM

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]

    def get_arrays(self):
        """Return (Theta, Phi, Cd, Cl, BubbleLen, WakeDeficit) as float32 arrays."""
        Theta = np.stack([s.theta for s in self.samples])
        Phi   = np.stack([s.phi   for s in self.samples])
        Cd    = np.array([s.cd    for s in self.samples], dtype=np.float32)
        Cl    = np.array([s.cl    for s in self.samples], dtype=np.float32)
        BL    = np.array([s.bubble_length for s in self.samples], dtype=np.float32)
        WD    = np.array([s.wake_deficit  for s in self.samples], dtype=np.float32)
        return Theta, Phi, Cd, Cl, BL, WD

    def summary(self) -> str:
        Theta, Phi, Cd, Cl, BL, WD = self.get_arrays()
        return (
            f"AhmedMLDataset: {len(self)} samples\n"
            f"  theta dim   : {self.theta_dim}\n"
            f"  phi dim     : {self.phi_dim}  ({N_PROBES} probes × {PHI_DIM_PER_PROBE} fields)\n"
            f"  Cd          : {Cd.min():.4f} → {Cd.max():.4f}  mean={Cd.mean():.4f}\n"
            f"  Cl          : {Cl.min():.4f} → {Cl.max():.4f}  mean={Cl.mean():.4f}\n"
            f"  bubble_len  : {BL.min():.3f} → {BL.max():.3f}  mean={BL.mean():.3f} m\n"
            f"  wake_deficit: {WD.min():.3f} → {WD.max():.3f}  mean={WD.mean():.3f} m/s"
        )


# ══════════════════════════════════════════════════════════════════════════
# Top-level loader
# ══════════════════════════════════════════════════════════════════════════

def load_ahmedml(
    data_dir: str = 'data/ahmedml/',
    max_runs:  Optional[int] = None,
    verbose:   bool = True,
) -> AhmedMLDataset:
    """
    Load the AhmedML dataset from disk.

    Parameters
    ----------
    data_dir : path to the ahmedml/ folder
    max_runs : load only this many runs (useful for quick debugging)
    verbose  : print progress

    Returns
    -------
    AhmedMLDataset
    """
    geo_all = pd.read_csv(os.path.join(data_dir, 'geo_parameters_all.csv'))
    fm_all  = pd.read_csv(os.path.join(data_dir, 'force_mom_all.csv'))
    fm_all.columns = fm_all.columns.str.strip()

    # Compute normalisation stats on the FULL geometry CSV (all 499 runs)
    raw = geo_all[GEO_COLS].copy().astype(float)
    raw.iloc[:, :7] *= MM_TO_M
    theta_mean = raw.values.mean(axis=0).astype(np.float32)
    theta_std  = raw.values.std(axis=0).astype(np.float32)

    samples = []
    run_ids = sorted(geo_all['run'].tolist())
    if max_runs:
        run_ids = run_ids[:max_runs]

    for run_id in run_ids:
        run_dir  = os.path.join(data_dir, f'run_{run_id}')
        if not os.path.isdir(run_dir):
            continue

        geo_row = geo_all[geo_all['run'] == run_id]
        if geo_row.empty:
            continue
        geo_row = geo_row.iloc[0]

        if verbose:
            print(f"  Loading run {run_id:3d} / {run_ids[-1]}...", end='\r')

        sample = load_run(run_dir, run_id, geo_row, theta_mean, theta_std)
        if sample is not None:
            samples.append(sample)

    if verbose:
        print(f"\n  Loaded {len(samples)} / {len(run_ids)} runs.")

    return AhmedMLDataset(samples)


# ══════════════════════════════════════════════════════════════════════════
# Quick smoke test (runs on any available run)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/ahmedml/'
    print(f"Smoke test on {data_dir} (loading 2 runs) ...\n")

    ds = load_ahmedml(data_dir, max_runs=2, verbose=True)

    if len(ds) == 0:
        print("No runs loaded yet — download still in progress. Run again once data/ahmedml/run_X/ dirs appear.")
        raise SystemExit(0)

    s = ds[0]
    print(f"\nSample run {s.run_id}:")
    print(f"  theta (normalised, 8): {s.theta.round(3)}")
    print(f"  phi dim              : {len(s.phi)}")
    print(f"  phi range            : {s.phi.min():.4f} → {s.phi.max():.4f}")
    print(f"  Cd={s.cd:.4f}  Cl={s.cl:.4f}")
    print(f"  bubble_length={s.bubble_length:.3f}m  wake_deficit={s.wake_deficit:.3f}m/s")

    if len(ds) >= 2:
        print()
        print(ds.summary())
