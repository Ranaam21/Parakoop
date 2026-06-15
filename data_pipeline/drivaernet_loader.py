"""
data_pipeline/drivaernet_loader.py

DrivAerNet++ data loader for ParaKoop training.

Current data available:
    DrivAerNetPlusPlus_Cd_8k_Updated.csv   Cd for 8,121 shapes (downloaded)
    STL meshes                              pending Harvard Dataverse access
    VTK CFD fields                          pending Harvard Dataverse access
                                            (see memory: drivaernet-vtk-todo)

The loader handles both phases gracefully:
    Phase 1 (CSV only):  theta = ID-encoded body-style one-hot + normalised
                         Cd label, no phi (used for steady fixed-point
                         output-only supervision of A(theta))
    Phase 2 (full VTK):  adds surface pressure + velocity phi vector once
                         Harvard Dataverse data is placed in data/drivaernet/

ID format: F_D_WM_WW_0001
    F/N/E     = Fastback / Notchback / Estateback
    D/S       = Detailed / Simplified
    WM/NM     = With Mirrors / No Mirrors
    WW/NW     = With Wheels / No Wheels
    0001-XXXX = design index

Usage
-----
    from data_pipeline.drivaernet_loader import load_drivaernet

    ds = load_drivaernet('data/drivaernet/')
    sample = ds[0]
    # sample.design_id   : str    e.g. 'F_D_WM_WW_0001'
    # sample.body_style  : str    'fastback' | 'notchback' | 'estateback'
    # sample.cd          : float
    # sample.theta_discrete : np.ndarray (4,) — one-hot encoded config flags
"""

from __future__ import annotations

import os
import dataclasses
from typing import Optional

import numpy as np
import pandas as pd


CD_CSV = 'DrivAerNetPlusPlus_Cd_8k_Updated.csv'

BODY_STYLE_MAP = {'F': 'fastback', 'N': 'notchback', 'E': 'estateback'}
DETAIL_MAP     = {'D': 1, 'S': 0}
MIRROR_MAP     = {'WM': 1, 'NM': 0}
WHEEL_MAP      = {'WW': 1, 'NW': 0}


@dataclasses.dataclass
class DrivAerSample:
    """One DrivAerNet++ design."""
    design_id:       str
    body_style:      str          # fastback / notchback / estateback
    cd:              float
    theta_discrete:  np.ndarray   # (4,) float32: [body_style_onehot(3), detail, mirrors, wheels]
    phi:             Optional[np.ndarray] = None   # set once VTK data is available


def _parse_id(design_id: str):
    """Parse DrivAerNet++ ID string into component flags."""
    parts = design_id.split('_')
    body  = BODY_STYLE_MAP.get(parts[0], 'fastback')
    detail  = float(DETAIL_MAP.get(parts[1],  1))
    mirrors = float(MIRROR_MAP.get(parts[2],  1))
    wheels  = float(WHEEL_MAP.get(parts[3],   1))
    # One-hot body style: [is_fastback, is_notchback, is_estateback]
    style_onehot = np.array([
        float(body == 'fastback'),
        float(body == 'notchback'),
        float(body == 'estateback'),
    ], dtype=np.float32)
    theta = np.concatenate([style_onehot, [detail, mirrors, wheels]]).astype(np.float32)
    return body, theta


class DrivAerNetDataset:
    """
    DrivAerNet++ dataset.  Phase 1: Cd + discrete config only.
    Phase 2 (VTK): call add_vtk_fields() once data arrives.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

    def get_cd_array(self) -> np.ndarray:
        return np.array([s.cd for s in self.samples], dtype=np.float32)

    def get_theta_array(self) -> np.ndarray:
        return np.stack([s.theta_discrete for s in self.samples])

    def by_style(self, style: str) -> 'DrivAerNetDataset':
        return DrivAerNetDataset([s for s in self.samples if s.body_style == style])

    def summary(self) -> str:
        cd = self.get_cd_array()
        counts = {s: sum(1 for x in self.samples if x.body_style == s)
                  for s in ('fastback', 'notchback', 'estateback')}
        return (
            f"DrivAerNetDataset: {len(self)} designs\n"
            f"  Body styles : fastback={counts['fastback']}  "
            f"notchback={counts['notchback']}  estateback={counts['estateback']}\n"
            f"  Cd range    : {cd.min():.4f} → {cd.max():.4f}  mean={cd.mean():.4f}\n"
            f"  VTK fields  : {'loaded' if self.samples[0].phi is not None else 'pending (Harvard Dataverse)'}"
        )


def load_drivaernet(
    data_dir: str = 'data/drivaernet/',
    max_samples: Optional[int] = None,
) -> DrivAerNetDataset:
    """
    Load DrivAerNet++ from the CSV (Phase 1).
    Handles missing VTK gracefully — warns and continues.
    """
    csv_path = os.path.join(data_dir, CD_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cd CSV not found at {csv_path}.\n"
            f"Download from Dropbox and place in {data_dir}"
        )

    df = pd.read_csv(csv_path)
    df.columns = ['design_id', 'cd']
    df = df.dropna()

    if max_samples:
        df = df.head(max_samples)

    samples = []
    for _, row in df.iterrows():
        body_style, theta_disc = _parse_id(str(row['design_id']))
        samples.append(DrivAerSample(
            design_id=str(row['design_id']),
            body_style=body_style,
            cd=float(row['cd']),
            theta_discrete=theta_disc,
        ))

    return DrivAerNetDataset(samples)


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/drivaernet/'
    ds = load_drivaernet(data_dir)
    print(ds.summary())
    print(f"\nSample[0]: {ds[0].design_id}  Cd={ds[0].cd:.4f}  style={ds[0].body_style}")
    print(f"theta_discrete: {ds[0].theta_discrete}")
    print(f"\nFastback only:")
    print(ds.by_style('fastback').summary())
