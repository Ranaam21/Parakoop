"""
scripts/cfd_validate.py

Step 5 — OpenFOAM CFD Validation.

Two validation modes:

  --mode ahmed (default)
      Hold-out validation: reproduce the same 80/20 seed=42 split used in training,
      isolate the AhmedML rows that ended up in the val set, compare model
      predictions against ground-truth Cd/Cl from AhmedML force_mom CSVs
      (which ARE OpenFOAM k-ω SST RANS results).

  --mode cross
      Cross-scale lookup: run inverse design for target Cd values, find the
      nearest AhmedML run in unified theta space, compare predictions vs CFD.
      Useful for sanity checks but θ-dist can be large across scales.

Usage
-----
    python scripts/cfd_validate.py
    python scripts/cfd_validate.py --mode ahmed --out results/ahmed_val.csv
    python scripts/cfd_validate.py --mode cross --targets 0.25 0.30 0.35
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

from data_pipeline.unified_loader import load_unified, THETA_COLS, _AH_DIR, _ahmed_row_to_theta
from koopman.model   import ParaKoopModel
from koopman.inverse_design import suggest_geometry

CKPT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints',
                    'unified', 'parakoop_unified_best.pt')
VAL_FRAC = 0.15
SEED     = 42


def load_model(ckpt_path: str) -> ParaKoopModel:
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg   = ckpt.get('model_cfg', {})
    model = ParaKoopModel(
        phi_dim       = cfg.get('phi_dim', 1),
        theta_dim     = cfg.get('theta_dim', 8),
        koopman_dim   = cfg.get('koopman_dim', 128),
        operator_rank = cfg.get('operator_rank', 16),
        hidden_lift   = cfg.get('hidden_lift', 64),
        hidden_op     = cfg.get('hidden_op', 64),
        lambda_fp_cd  = cfg.get('lambda_fp_cd', 0.1),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# ── Mode A: AhmedML held-out val validation ───────────────────────────────

def validate_ahmed_holdout(model: ParaKoopModel, dataset, ahmed_dir: str) -> pd.DataFrame:
    """
    Reproduce the same train/val split (seed=42) used in training.
    Identify AhmedML rows in the val set, compare model Cd/Cl vs CFD.
    """
    theta_all, cd_all, cl_all, has_cl_all = dataset.get_arrays()
    n = len(theta_all)
    n_val = int(n * VAL_FRAC)
    n_train = n - n_val

    # Reproduce split
    idx_full = torch.arange(n)
    _, val_idx = random_split(
        idx_full.tolist(), [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_indices = list(val_idx)

    # AhmedML rows: use has_cl flag directly (load order: STL → AhmedML → 8K CSV)
    # This avoids any hardcoded index offset assumptions.
    ahmed_global_indices = [i for i in range(n) if has_cl_all[i]]

    # Load AhmedML ground-truth: run_id → (cd_cfd, cl_cfd, slant_deg)
    geo_all = pd.read_csv(os.path.join(ahmed_dir, 'geo_parameters_all.csv'))
    fm_all  = pd.read_csv(os.path.join(ahmed_dir, 'force_mom_all.csv'))
    fm_all.columns = fm_all.columns.str.strip()
    # Build ordered list matching the unified dataset's AhmedML row order
    # (same iteration order as build(): ah_geo.merge(ah_fm).iterrows())
    ah_fm  = pd.read_csv(os.path.join(ahmed_dir, 'force_mom_all.csv'))
    ah_fm.columns = ah_fm.columns.str.strip()
    ah_geo = pd.read_csv(os.path.join(ahmed_dir, 'geo_parameters_all.csv'))
    ah_df  = ah_geo.merge(ah_fm, on='run').dropna().reset_index(drop=True)

    records = []
    model.eval()
    with torch.no_grad():
        for global_idx in val_indices:
            if not has_cl_all[global_idx]:
                continue   # not an AhmedML row

            # local_idx within AhmedML block
            local_idx = ahmed_global_indices.index(global_idx)
            if local_idx >= len(ah_df):
                continue

            ah_row = ah_df.iloc[local_idx]
            gt = {
                'cd_cfd':    float(ah_row['cd']),
                'cl_cfd':    float(ah_row['cl']),
                'slant_deg': float(ah_row['slant-angle-degrees']),
                'height_mm': float(ah_row['body-height']),
                'length_mm': float(ah_row['body-length']),
            }
            run_id = int(ah_row['run'])

            theta_t = torch.tensor(
                theta_all[global_idx], dtype=torch.float32
            ).unsqueeze(0)
            cd_pred, cl_pred, _ = model.forward_cd_only(theta_t)

            records.append({
                'run_id':    run_id,
                'slant_deg': gt['slant_deg'],
                'height_mm': gt['height_mm'],
                'length_mm': gt['length_mm'],
                'cd_cfd':    gt['cd_cfd'],
                'cl_cfd':    gt['cl_cfd'],
                'cd_pred':   round(cd_pred.item(), 4),
                'cl_pred':   round(cl_pred.item(), 4),
                'err_cd':    round(abs(cd_pred.item() - gt['cd_cfd']), 4),
                'err_cl':    round(abs(cl_pred.item() - gt['cl_cfd']), 4),
            })

    return pd.DataFrame(records)


def print_ahmed_table(df: pd.DataFrame) -> None:
    # Sort by CFD Cd for clean display
    df_show = df.sort_values('cd_cfd').reset_index(drop=True)

    print("\n" + "═" * 80)
    print("  ParaKoop — AhmedML Hold-out Validation  (OpenFOAM k-ω SST RANS)")
    print("═" * 80)
    print(f"  {'run':>6}  {'slant°':>7}  {'h mm':>6}  "
          f"{'CFD Cd':>8}  {'PK Cd':>8}  {'ΔCd':>7}  "
          f"{'CFD Cl':>8}  {'PK Cl':>8}  {'ΔCl':>7}")
    print("  " + "─" * 74)
    for _, r in df_show.iterrows():
        print(f"  {int(r.run_id):>6}  {r.slant_deg:>7.1f}  {r.height_mm:>6.0f}  "
              f"{r.cd_cfd:>8.4f}  {r.cd_pred:>8.4f}  {r.err_cd:>7.4f}  "
              f"{r.cl_cfd:>8.4f}  {r.cl_pred:>8.4f}  {r.err_cl:>7.4f}")
    print("  " + "─" * 74)
    print(f"  Mean |ΔCd| : {df.err_cd.mean():.4f}   "
          f"Median |ΔCd| : {df.err_cd.median():.4f}   "
          f"Max |ΔCd| : {df.err_cd.max():.4f}")
    print(f"  Mean |ΔCl| : {df.err_cl.mean():.4f}   "
          f"Median |ΔCl| : {df.err_cl.median():.4f}   "
          f"Max |ΔCl| : {df.err_cl.max():.4f}")
    print(f"  N val samples (AhmedML) : {len(df)}")
    print("═" * 80)
    print()
    print("  Cd/Cl ground truth from AhmedML OpenFOAM RANS (Lienhart et al. protocol).")
    print("  Val split: 15% held-out, seed=42, matching training split exactly.")


# ── Mode B: cross-scale lookup (original approach) ────────────────────────

def load_ahmed_all(ahmed_dir: str) -> list:
    geo_all = pd.read_csv(os.path.join(ahmed_dir, 'geo_parameters_all.csv'))
    fm_all  = pd.read_csv(os.path.join(ahmed_dir, 'force_mom_all.csv'))
    fm_all.columns = fm_all.columns.str.strip()
    rows = []
    for _, gr in geo_all.iterrows():
        run_id = int(gr['run'])
        fm     = fm_all[fm_all['run'] == run_id]
        if fm.empty:
            continue
        rows.append({
            'run_id':    run_id,
            'theta':     _ahmed_row_to_theta(gr),
            'cd_cfd':    float(fm['cd'].iloc[0]),
            'cl_cfd':    float(fm['cl'].iloc[0]),
            'slant_deg': float(gr['slant-angle-degrees']),
            'height_mm': float(gr['body-height']),
            'length_mm': float(gr['body-length']),
        })
    return rows


def validate_cross(targets, model, dataset, ahmed_rows, lambda_prox=2.0):
    records = []
    styles  = ['fastback', 'notchback', 'estateback']
    for target_cd in targets:
        print(f"\n── Target Cd = {target_cd:.3f} ─────────────────────────────────")
        best, best_err = None, 1e9
        for style in styles:
            r = suggest_geometry(model, dataset, target_cd,
                                 start_style=style, n_steps=500,
                                 lambda_prox=lambda_prox, verbose=False)
            err = abs(r['achieved_cd'] - target_cd)
            if err < best_err:
                best_err = err; best = r; best['start_style'] = style

        # Find nearest AhmedML run
        theta_end = best['theta_end']
        dists = sorted(ahmed_rows, key=lambda x: float(np.linalg.norm(theta_end - x['theta'])))
        nr    = dists[0]
        dist  = float(np.linalg.norm(theta_end - nr['theta']))

        with torch.no_grad():
            t = torch.tensor(nr['theta'], dtype=torch.float32).unsqueeze(0)
            cd_at, cl_at, _ = model.forward_cd_only(t)

        print(f"  Best style: {best['start_style']}  PK Cd={best['achieved_cd']:.4f}")
        print(f"  Nearest AhmedML: run_{nr['run_id']}  θ-dist={dist:.4f}")
        print(f"    slant={nr['slant_deg']:.1f}°  CFD Cd={nr['cd_cfd']:.4f}  "
              f"PK@nearest Cd={cd_at.item():.4f}  |ΔCd|={abs(cd_at.item()-nr['cd_cfd']):.4f}")
        records.append({
            'target_cd': target_cd,
            'start_style': best['start_style'],
            'pred_cd': best['achieved_cd'],
            'nearest_run': nr['run_id'],
            'theta_dist': round(dist, 4),
            'cfd_cd': nr['cd_cfd'], 'cfd_cl': nr['cl_cfd'],
            'model_cd_at_nearest': round(cd_at.item(), 4),
            'abs_err_cd': round(abs(cd_at.item() - nr['cd_cfd']), 4),
        })
    return records


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('--mode', choices=['ahmed', 'cross'], default='ahmed')
    pa.add_argument('--targets', type=float, nargs='+',
                    default=[0.25, 0.28, 0.30, 0.35, 0.40])
    pa.add_argument('--lambda-prox', type=float, default=2.0)
    pa.add_argument('--checkpoint', default=CKPT)
    pa.add_argument('--out', default=None)
    return pa.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  ParaKoop — OpenFOAM CFD Validation")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    model   = load_model(args.checkpoint)
    dataset = load_unified(verbose=False)

    if args.mode == 'ahmed':
        print("\nValidating on held-out AhmedML val set (same split as training)...")
        df = validate_ahmed_holdout(model, dataset, _AH_DIR)
        if df.empty:
            print("  No AhmedML rows found in val split — check split logic.")
        else:
            print_ahmed_table(df)
            if args.out:
                os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
                df.to_csv(args.out, index=False)
                print(f"\nSaved: {args.out}")

    else:  # cross
        print("\nLoading AhmedML ground truth...")
        ahmed_rows = load_ahmed_all(_AH_DIR)
        print(f"  {len(ahmed_rows)} runs loaded")
        records = validate_cross(args.targets, model, dataset, ahmed_rows,
                                 lambda_prox=args.lambda_prox)
        df = pd.DataFrame(records)
        print(f"\nMean |ΔCd| (model vs CFD at nearest): "
              f"{df.abs_err_cd.mean():.4f}")
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            df.to_csv(args.out, index=False)
            print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
