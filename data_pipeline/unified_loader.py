"""
data_pipeline/unified_loader.py

Merges all three data sources into a single unified 8-dim geometry theta:

  1. DrivAerNet++ 8K CSV  — Cd only, theta imputed from config-conditional means
  2. DrivAerNet++ 1,163 STL — Cd + real extracted geometry from geometry_features.csv
  3. AhmedML 499 CSV       — Cd + Cl + 8 geometry params from CSV files

Unified theta (8 dims, scale-invariant — works across DrivAerNet ~4700mm and AhmedML ~1000mm):
  [style_fastback, style_notchback, style_estateback,
   height_length_ratio, width_height_ratio, rear_slant_norm,
   cabin_frac, detail_flag]

VTU flow-field phi (optional, 76 AhmedML runs only):
  load_phi_data() returns (Theta_phi, Phi, Cd_phi, Cl_phi) for grounding training.
  Requires pyvista.  Main training works without it.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


# ── Unified theta definition ──────────────────────────────────────────────────

THETA_COLS = [
    'style_fastback',       # 1 if fastback / Ahmed-body, else 0
    'style_notchback',      # 1 if notchback, else 0
    'style_estateback',     # 1 if estateback, else 0
    'height_length_ratio',  # height / length  (DrivAer ~0.27, Ahmed ~0.26)
    'width_height_ratio',   # width / height   (DrivAer ~0.84, Ahmed ~1.40)
    'rear_slant_norm',      # slant_deg / 90.0  (bounded 0–1)
    'cabin_frac',           # cabin_length / body_length
    'detail_flag',          # 0=simplified  0.5=ahmed  1=detailed
]
THETA_DIM = 8

# Physical bounds for each theta slot — used by inverse design clamping
THETA_MIN = np.array([0., 0., 0.,  0.20, 0.50, 0.00, 0.25, 0.0], dtype=np.float32)
THETA_MAX = np.array([1., 1., 1.,  0.45, 1.80, 0.55, 0.70, 1.0], dtype=np.float32)

# Reference lengths by body style for inverse design mm decoding
STYLE_REF_LENGTH_MM = {'fastback': 4850., 'notchback': 4780., 'estateback': 4800.}

# Default CSV paths (relative to parakoop/ root)
_ROOT = os.path.join(os.path.dirname(__file__), '..')
_DA_CD_CSV  = os.path.join(_ROOT, 'data', 'drivaernet', 'DrivAerNetPlusPlus_Cd_8k_Updated.csv')
_DA_GEO_CSV = os.path.join(_ROOT, 'data', 'drivaernet', 'geometry_features.csv')
_AH_FM_CSV  = os.path.join(_ROOT, 'data', 'ahmedml', 'force_mom_all.csv')
_AH_GEO_CSV = os.path.join(_ROOT, 'data', 'ahmedml', 'geo_parameters_all.csv')
_AH_DIR     = os.path.join(_ROOT, 'data', 'ahmedml')


# ── Body-style parsing ────────────────────────────────────────────────────────

def _parse_style(design_id: str) -> str:
    prefix = design_id.split('_')[0]
    return {'F': 'fastback', 'N': 'notchback', 'E': 'estateback'}.get(prefix, 'fastback')

def _style_onehot(style: str) -> np.ndarray:
    return np.array([
        float(style == 'fastback'),
        float(style == 'notchback'),
        float(style == 'estateback'),
    ], dtype=np.float32)

def _is_detailed(design_id: str) -> float:
    """Return 1.0 if design ID indicates detailed mesh, else 0.0."""
    parts = design_id.split('_')
    return 1.0 if len(parts) > 1 and parts[1] == 'D' else 0.0


# ── Per-source theta builders ─────────────────────────────────────────────────

def _stl_row_to_theta(row: pd.Series, design_id: str) -> np.ndarray:
    """DrivAerNet STL row (geometry_features.csv) → 8-dim unified theta."""
    style = _parse_style(design_id)
    length = float(row['length_mm'])
    height = float(row['height_mm'])
    width  = float(row['width_mm'])
    slant  = float(row['rear_slant_deg'])
    cabin  = float(row['cabin_length_frac'])
    h_l = height / length if length > 0 else 0.27
    w_h = width  / height if height > 0 else 0.84
    return np.concatenate([
        _style_onehot(style),
        [h_l, w_h, slant / 90.0, cabin, _is_detailed(design_id)],
    ]).astype(np.float32)


def _ahmed_row_to_theta(geo_row: pd.Series) -> np.ndarray:
    """AhmedML geo_parameters row → 8-dim unified theta."""
    length = float(geo_row['body-length'])
    height = float(geo_row['body-height'])
    width  = float(geo_row['body-width'])
    slant  = float(geo_row['slant-angle-degrees'])
    slant_surf = float(geo_row.get('slant-surface-length', 0.0))
    cabin  = slant_surf / length if length > 0 else 0.45
    h_l = height / length if length > 0 else 0.26
    w_h = width  / height if height > 0 else 1.40
    return np.concatenate([
        [1.0, 0.0, 0.0],               # Ahmed body is fastback-like
        [h_l, w_h, slant / 90.0, cabin, 0.5],  # detail_flag=0.5 (bluff body geometry)
    ]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# UnifiedDataset
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedDataset:
    """
    Merged training dataset from DrivAerNet++ and AhmedML with shared 8-dim theta.

    Usage
    -----
        ds = UnifiedDataset.build()
        Theta, Cd, Cl, has_cl = ds.get_arrays()

        # Inverse design starting point
        theta0 = ds.style_mean_theta('fastback')

        # Decode optimized theta back to geometry
        params = ds.theta_to_named_params(theta_optimized)
        # → {'style': 'fastback', 'rear_slant_deg': 17.4, 'height_mm': 1265, ...}
    """

    def __init__(
        self,
        theta:   np.ndarray,   # (N, 8) float32
        cd:      np.ndarray,   # (N,)   float32
        cl:      np.ndarray,   # (N,)   float32
        has_cl:  np.ndarray,   # (N,)   bool
        sources: np.ndarray,   # (N,)   str — '8k_imputed' | 'stl_real' | 'ahmed_csv'
    ):
        self.theta   = theta
        self.cd      = cd
        self.cl      = cl
        self.has_cl  = has_cl
        self.sources = sources

    def get_arrays(self):
        """Return (Theta, Cd, Cl, has_cl) as float32 / bool arrays."""
        return self.theta, self.cd, self.cl, self.has_cl

    # ── Source-breakdown helpers ──────────────────────────────────────────────

    def _mask(self, source: str) -> np.ndarray:
        return self.sources == source

    def style_mean_theta(self, style: str = 'fastback') -> np.ndarray:
        """
        Mean unified theta for a given body style, computed from real STL designs.
        Used as the starting point for gradient-based inverse design.
        """
        mask = self._mask('stl_real')
        if not mask.any():
            # Fallback: use all sources
            mask = np.ones(len(self.theta), dtype=bool)
        style_idx = ['fastback', 'notchback', 'estateback'].index(style)
        style_mask = mask & (self.theta[:, style_idx] > 0.5)
        if not style_mask.any():
            style_mask = mask
        return self.theta[style_mask].mean(axis=0).astype(np.float32)

    def theta_to_named_params(
        self,
        theta_row: np.ndarray,
        ref_length_mm: Optional[float] = None,
    ) -> dict:
        """
        Convert 8-dim theta to human-readable geometry parameters.

        Parameters
        ----------
        theta_row     : (8,) float32 — unified theta vector
        ref_length_mm : reference car length in mm; if None, inferred from style
        """
        style_scores = theta_row[:3]
        style        = ['fastback', 'notchback', 'estateback'][int(np.argmax(style_scores))]
        h_l   = float(theta_row[3])
        w_h   = float(theta_row[4])
        slant = float(theta_row[5]) * 90.0
        cabin = float(theta_row[6])

        if ref_length_mm is None:
            ref_length_mm = STYLE_REF_LENGTH_MM.get(style, 4800.)

        height_mm = h_l * ref_length_mm
        width_mm  = w_h * height_mm

        return {
            'style':                style,
            'rear_slant_deg':       round(slant, 1),
            'cabin_frac':           round(cabin, 3),
            'height_mm':            round(height_mm, 0),
            'width_mm':             round(width_mm, 0),
            'height_length_ratio':  round(h_l, 4),
            'width_height_ratio':   round(w_h, 4),
            'ref_length_mm':        round(ref_length_mm, 0),
        }

    def summary(self) -> str:
        n_8k  = self._mask('8k_imputed').sum()
        n_stl = self._mask('stl_real').sum()
        n_ah  = self._mask('ahmed_csv').sum()
        cd    = self.cd
        cl_ah = self.cl[self.has_cl]
        lines = [
            f"UnifiedDataset: {len(self.theta):,} samples",
            f"  DrivAerNet 8K  (imputed geo) : {n_8k:4d}  Cd {cd[self._mask('8k_imputed')].min():.3f}–{cd[self._mask('8k_imputed')].max():.3f}",
            f"  DrivAerNet STL (real geo)    : {n_stl:4d}  Cd {cd[self._mask('stl_real')].min():.3f}–{cd[self._mask('stl_real')].max():.3f}",
            f"  AhmedML CSV    (Cd+Cl)       : {n_ah:4d}  Cd {cd[self._mask('ahmed_csv')].min():.3f}–{cd[self._mask('ahmed_csv')].max():.3f}  Cl {cl_ah.min():.3f}–{cl_ah.max():.3f}",
            f"  theta dim: {THETA_DIM}   has_cl: {self.has_cl.sum()} samples",
            f"  Overall Cd: {cd.min():.4f} → {cd.max():.4f}  mean={cd.mean():.4f}",
        ]
        return '\n'.join(lines)

    # ── phi path (76 VTU runs) ────────────────────────────────────────────────

    def load_phi_data(
        self,
        ahmed_dir: str = _AH_DIR,
        verbose: bool = True,
    ) -> Optional[tuple]:
        """
        Load AhmedML VTU flow fields for phi-supervised grounding (76 runs).

        Returns
        -------
        (Theta_phi, Phi, Cd_phi, Cl_phi) — float32 arrays for the runs that have VTU.
        Returns None if pyvista is not available.

        Notes
        -----
        Theta_phi uses the UNIFIED 8-dim theta (not the z-score AhmedML theta).
        Phi is the 13,824-dim wake probe vector from volume_X.vtu.
        """
        try:
            from data_pipeline.ahmedml_loader import (
                load_run, GEO_COLS, MM_TO_M,
            )
        except ImportError:
            return None

        try:
            import pyvista  # noqa: F401
        except ImportError:
            if verbose:
                print("pyvista not installed — phi path unavailable (pip install pyvista)")
            return None

        geo_all = pd.read_csv(os.path.join(ahmed_dir, 'geo_parameters_all.csv'))
        fm_all  = pd.read_csv(os.path.join(ahmed_dir, 'force_mom_all.csv'))
        fm_all.columns = fm_all.columns.str.strip()

        # theta_mean/std needed by load_run() internally (for its own z-score theta)
        raw = geo_all[GEO_COLS].copy().astype(float)
        raw.iloc[:, :7] *= MM_TO_M
        theta_mean = raw.values.mean(axis=0).astype(np.float32)
        theta_std  = raw.values.std(axis=0).astype(np.float32)

        theta_list, phi_list, cd_list, cl_list = [], [], [], []

        for _, geo_row in geo_all.iterrows():
            run_id  = int(geo_row['run'])
            run_dir = os.path.join(ahmed_dir, f'run_{run_id}')
            vtu     = os.path.join(run_dir, f'volume_{run_id}.vtu')

            if not os.path.exists(vtu):
                continue

            if verbose:
                print(f"  Loading VTU for run {run_id:3d}...", end='\r')

            sample = load_run(run_dir, run_id, geo_row, theta_mean, theta_std)
            if sample is None:
                continue

            # Use our unified theta, not the z-score AhmedML theta
            unified_theta = _ahmed_row_to_theta(geo_row)
            theta_list.append(unified_theta)
            phi_list.append(sample.phi)

            fm_row = fm_all[fm_all['run'] == run_id]
            cd_list.append(float(fm_row['cd'].iloc[0]) if not fm_row.empty else sample.cd)
            cl_list.append(float(fm_row['cl'].iloc[0]) if not fm_row.empty else sample.cl)

        if not theta_list:
            if verbose:
                print("No VTU files found.")
            return None

        if verbose:
            print(f"\n  Loaded phi for {len(theta_list)} VTU runs.")

        return (
            np.stack(theta_list).astype(np.float32),
            np.stack(phi_list).astype(np.float32),
            np.array(cd_list, dtype=np.float32),
            np.array(cl_list, dtype=np.float32),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Builder
# ══════════════════════════════════════════════════════════════════════════════

def build(
    drivaernet_csv: str = _DA_CD_CSV,
    geometry_csv:   str = _DA_GEO_CSV,
    ahmed_fm_csv:   str = _AH_FM_CSV,
    ahmed_geo_csv:  str = _AH_GEO_CSV,
    verbose:        bool = True,
) -> UnifiedDataset:
    """
    Build the unified dataset from all three sources.

    Priority:
      1. DrivAerNet STL (real geometry) — highest quality, used as-is
      2. AhmedML CSV — all 499 runs with Cd + Cl
      3. DrivAerNet 8K CSV — remaining designs not in (1), theta imputed from config means
    """
    theta_rows, cd_rows, cl_rows, has_cl_rows, source_rows = [], [], [], [], []

    # ── Source 1: DrivAerNet STL (real geometry) ──────────────────────────────
    if verbose:
        print("Loading DrivAerNet STL geometry features...")
    geo_df = pd.read_csv(geometry_csv).dropna(subset=['length_mm', 'cd'])
    stl_ids = set(geo_df['design_id'].tolist())

    for _, row in geo_df.iterrows():
        did = str(row['design_id'])
        theta_rows.append(_stl_row_to_theta(row, did))
        cd_rows.append(float(row['cd']))
        cl_rows.append(0.0)
        has_cl_rows.append(False)
        source_rows.append('stl_real')

    if verbose:
        print(f"  {len(geo_df)} STL designs loaded.")

    # ── Source 2: AhmedML CSV (Cd + Cl + geometry params) ────────────────────
    if verbose:
        print("Loading AhmedML CSV data...")
    ah_fm  = pd.read_csv(ahmed_fm_csv)
    ah_fm.columns = ah_fm.columns.str.strip()
    ah_geo = pd.read_csv(ahmed_geo_csv)
    ah_df  = ah_geo.merge(ah_fm, on='run').dropna()

    for _, row in ah_df.iterrows():
        theta_rows.append(_ahmed_row_to_theta(row))
        cd_rows.append(float(row['cd']))
        cl_rows.append(float(row['cl']))
        has_cl_rows.append(True)
        source_rows.append('ahmed_csv')

    if verbose:
        print(f"  {len(ah_df)} AhmedML runs loaded (Cd + Cl).")

    # ── Source 3: DrivAerNet 8K CSV (imputed geometry) ────────────────────────
    if verbose:
        print("Loading DrivAerNet 8K CSV (imputing geometry for non-STL designs)...")

    # Compute config-conditional mean theta from STL designs
    config_means = _compute_config_means(geo_df)

    cd_df = pd.read_csv(drivaernet_csv)
    cd_df.columns = ['design_id', 'cd']
    cd_df = cd_df.dropna()

    n_skip, n_add = 0, 0
    for _, row in cd_df.iterrows():
        did = str(row['design_id'])
        if did in stl_ids:
            n_skip += 1
            continue   # already in source 1 with real geometry
        config = _get_config_key(did)
        theta  = config_means.get(config, _fallback_theta(did))
        theta_rows.append(theta)
        cd_rows.append(float(row['cd']))
        cl_rows.append(0.0)
        has_cl_rows.append(False)
        source_rows.append('8k_imputed')
        n_add += 1

    if verbose:
        print(f"  {n_add} imputed designs added ({n_skip} skipped — already in STL set).")

    ds = UnifiedDataset(
        theta   = np.stack(theta_rows).astype(np.float32),
        cd      = np.array(cd_rows,      dtype=np.float32),
        cl      = np.array(cl_rows,      dtype=np.float32),
        has_cl  = np.array(has_cl_rows,  dtype=bool),
        sources = np.array(source_rows),
    )

    if verbose:
        print()
        print(ds.summary())

    return ds


# ── Config-conditional imputation helpers ─────────────────────────────────────

def _get_config_key(design_id: str) -> str:
    """Extract config string from design ID, e.g. 'F_D_WM_WW_0001' → 'F_D_WM_WW'."""
    parts = design_id.split('_')
    return '_'.join(parts[:4]) if len(parts) >= 4 else design_id

def _compute_config_means(geo_df: pd.DataFrame) -> dict:
    """
    Compute mean unified theta per config from real STL designs.
    Returns {config_key: theta_8dim} mapping.
    """
    if 'config' not in geo_df.columns:
        return {}

    means = {}
    for config, grp in geo_df.groupby('config'):
        thetas = []
        for _, row in grp.iterrows():
            did = str(row['design_id'])
            thetas.append(_stl_row_to_theta(row, did))
        means[config] = np.stack(thetas).mean(axis=0).astype(np.float32)
    return means

def _fallback_theta(design_id: str) -> np.ndarray:
    """Fallback theta when config not in means — use style-derived defaults."""
    style  = _parse_style(design_id)
    detail = _is_detailed(design_id)
    defaults = {
        'fastback':   [0.270, 0.840, 0.289, 0.520],  # h/l, w/h, slant/90, cabin
        'notchback':  [0.276, 0.833, 0.094, 0.470],
        'estateback': [0.277, 0.826, 0.056, 0.450],
    }
    ratios = defaults.get(style, defaults['fastback'])
    return np.concatenate([_style_onehot(style), ratios, [detail]]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience top-level function
# ══════════════════════════════════════════════════════════════════════════════

def load_unified(verbose: bool = True) -> UnifiedDataset:
    """Load unified dataset with default paths (run from parakoop/ root)."""
    return build(verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ds = load_unified(verbose=True)

    print("\nTheta stats (unified 8-dim):")
    print(f"  Columns: {THETA_COLS}")
    print(f"  Min:  {ds.theta.min(axis=0).round(3)}")
    print(f"  Max:  {ds.theta.max(axis=0).round(3)}")
    print(f"  Mean: {ds.theta.mean(axis=0).round(3)}")

    print("\nStyle mean theta (fastback → named params):")
    t0 = ds.style_mean_theta('fastback')
    print(f"  theta: {t0.round(3)}")
    print(f"  named: {ds.theta_to_named_params(t0)}")

    print("\nStyle mean theta (notchback → named params):")
    t1 = ds.style_mean_theta('notchback')
    print(f"  named: {ds.theta_to_named_params(t1)}")
