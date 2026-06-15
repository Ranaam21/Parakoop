"""
scripts/parse_foam_results.py

Parse OpenFOAM forceCoeffs output for all 5 Ahmed body validation cases
and produce a comparison table: CFD Cd/Cl vs ParaKoop predictions.

Usage
-----
    python scripts/parse_foam_results.py
    python scripts/parse_foam_results.py --cases-dir openfoam/cases --out results/openfoam_cfd_results.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CASES = ['slant10', 'slant12', 'slant15', 'slant17', 'slant19']


def find_coeff_file(case_dir: str) -> str | None:
    """Find forceCoeffs.dat inside postProcessing (OpenFOAM path varies by version)."""
    pattern = os.path.join(case_dir, 'postProcessing', '**', 'forceCoeffs.dat')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        # try coefficient.dat (some versions)
        pattern2 = os.path.join(case_dir, 'postProcessing', '**', 'coefficient.dat')
        matches = glob.glob(pattern2, recursive=True)
    return matches[0] if matches else None


def parse_coeff_file(path: str, n_avg: int = 100) -> tuple[float, float]:
    """
    Read forceCoeffs.dat and return time-averaged Cd, Cl from the last n_avg rows.
    OpenFOAM forceCoeffs.dat columns (v2512):
        Time  Cm  Cd  Cl  Cl(f)  Cl(r)
    or (older):
        Time  Cd  Cs  Cl  CmPitch  CmRoll  CmYaw
    Detect by header comment.
    """
    rows = []
    cd_col = cl_col = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # Parse column header
                parts = line.lstrip('#').split()
                if 'Cd' in parts:
                    cd_col = parts.index('Cd')
                if 'Cl' in parts and cl_col is None:
                    cl_col = parts.index('Cl')
                continue
            vals = line.split()
            if len(vals) >= 3:
                try:
                    rows.append([float(v) for v in vals])
                except ValueError:
                    continue

    if not rows:
        return float('nan'), float('nan')

    arr = np.array(rows)

    # Default column positions if header not parsed
    if cd_col is None:
        cd_col = 2   # typical: Time Cm Cd Cl ...
    if cl_col is None:
        cl_col = 3

    tail = arr[-n_avg:]
    cd = float(np.mean(tail[:, cd_col]))
    cl = float(np.mean(tail[:, cl_col]))
    return cd, cl


def load_pk_prediction(case_dir: str) -> tuple[float | None, float | None, float]:
    """Load ParaKoop predicted Cd/Cl and slant angle from parakoop_geometry.json."""
    jpath = os.path.join(case_dir, 'parakoop_geometry.json')
    if not os.path.exists(jpath):
        return None, None, float('nan')
    with open(jpath) as f:
        d = json.load(f)
    return d.get('pk_cd_pred'), d.get('pk_cl_pred'), d.get('slant_deg', float('nan'))


def print_table(df: pd.DataFrame) -> None:
    print("\n" + "═" * 80)
    print("  ParaKoop — OpenFOAM CFD Validation  (simpleFoam k-ω SST, U∞=40m/s)")
    print("═" * 80)
    print(f"  {'Case':<10}  {'slant°':>7}  "
          f"{'CFD Cd':>8}  {'PK Cd':>8}  {'|ΔCd|':>7}  "
          f"{'CFD Cl':>8}  {'PK Cl':>8}  {'|ΔCl|':>7}")
    print("  " + "─" * 74)
    for _, r in df.iterrows():
        cd_err = abs(r.cd_cfd - r.cd_pk) if not (np.isnan(r.cd_cfd) or np.isnan(r.cd_pk)) else float('nan')
        cl_err = abs(r.cl_cfd - r.cl_pk) if not (np.isnan(r.cl_cfd) or np.isnan(r.cl_pk)) else float('nan')
        cd_cfd_s = f"{r.cd_cfd:.4f}" if not np.isnan(r.cd_cfd) else "  n/a  "
        cl_cfd_s = f"{r.cl_cfd:.4f}" if not np.isnan(r.cl_cfd) else "  n/a  "
        cd_err_s = f"{cd_err:.4f}" if not np.isnan(cd_err) else "  n/a  "
        cl_err_s = f"{cl_err:.4f}" if not np.isnan(cl_err) else "  n/a  "
        print(f"  {r.case:<10}  {r.slant_deg:>7.1f}  "
              f"{cd_cfd_s:>8}  {r.cd_pk:>8.4f}  {cd_err_s:>7}  "
              f"{cl_cfd_s:>8}  {r.cl_pk:>8.4f}  {cl_err_s:>7}")
    print("  " + "─" * 74)

    valid = df.dropna(subset=['cd_cfd'])
    if len(valid):
        cd_mae = (valid.cd_cfd - valid.cd_pk).abs().mean()
        cl_mae = (valid.cl_cfd - valid.cl_pk).abs().mean()
        print(f"  Mean |ΔCd| : {cd_mae:.4f}   Mean |ΔCl| : {cl_mae:.4f}"
              f"   (N={len(valid)} cases with CFD results)")
    else:
        print("  No CFD results yet — run openfoam/run_all_cases.sh first.")
    print("═" * 80)


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('--cases-dir', default='openfoam/cases')
    pa.add_argument('--out', default='results/openfoam_cfd_results.csv')
    pa.add_argument('--n-avg', type=int, default=100,
                    help='Number of final iterations to average for Cd/Cl')
    return pa.parse_args()


def main():
    args = parse_args()

    records = []
    for case in CASES:
        case_dir = os.path.join(args.cases_dir, case)
        pk_cd, pk_cl, slant = load_pk_prediction(case_dir)

        coeff_path = find_coeff_file(case_dir)
        if coeff_path:
            cd_cfd, cl_cfd = parse_coeff_file(coeff_path, n_avg=args.n_avg)
            print(f"  {case}: CFD Cd={cd_cfd:.4f}  Cl={cl_cfd:.4f}  "
                  f"(from {os.path.relpath(coeff_path, args.cases_dir)})")
        else:
            cd_cfd = cl_cfd = float('nan')
            print(f"  {case}: no forceCoeffs.dat found — case not yet run")

        records.append({
            'case':      case,
            'slant_deg': slant,
            'cd_cfd':    cd_cfd,
            'cl_cfd':    cl_cfd,
            'cd_pk':     pk_cd if pk_cd is not None else float('nan'),
            'cl_pk':     pk_cl if pk_cl is not None else float('nan'),
        })

    df = pd.DataFrame(records)
    print_table(df)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n  Saved: {args.out}")


if __name__ == '__main__':
    main()
