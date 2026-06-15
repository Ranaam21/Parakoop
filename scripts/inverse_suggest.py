"""
scripts/inverse_suggest.py

Gradient-based geometric inverse design via trained ParaKoop.

Given a target Cd (and optionally Cl), finds the geometry parameter values
(rear slant angle, height, cabin fraction, etc.) that achieve the target.

Output is REAL geometric dimensions — NOT a lookup into config flags.

Usage
-----
    python scripts/inverse_suggest.py --target-cd 0.240
    python scripts/inverse_suggest.py --target-cd 0.230 --target-cl 0.10 --style fastback
    python scripts/inverse_suggest.py --target-cd 0.250 --all-styles   # try all 3 styles
    python scripts/inverse_suggest.py --target-cd 0.240 --json result.json
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from data_pipeline.unified_loader import load_unified
from koopman.model          import ParaKoopModel
from koopman.inverse_design import suggest_geometry, batch_suggest


_DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__), '..', 'checkpoints', 'unified', 'parakoop_unified_best.pt'
)


def load_model(ckpt_path: str) -> ParaKoopModel:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['model_cfg']
    model = ParaKoopModel(
        phi_dim       = cfg.get('phi_dim', 1),
        theta_dim     = cfg.get('theta_dim', 8),
        koopman_dim   = cfg.get('koopman_dim', 128),
        operator_rank = cfg.get('operator_rank', 16),
        hidden_lift   = cfg.get('hidden_lift', 64),
        hidden_op     = cfg.get('hidden_op', 64),
        lambda_perf   = cfg.get('lambda_perf', 1.0),
        lambda_ae     = cfg.get('lambda_ae', 0.0),
        lambda_fp_cd  = cfg.get('lambda_fp_cd', 0.1),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description='Geometric inverse design via ParaKoop')
    p.add_argument('--target-cd',   type=float, required=True,
                   help='Target drag coefficient (e.g. 0.240)')
    p.add_argument('--target-cl',   type=float, default=None,
                   help='Target lift coefficient (optional)')
    p.add_argument('--style',       default='fastback',
                   choices=['fastback', 'notchback', 'estateback'],
                   help='Starting body style for optimisation')
    p.add_argument('--all-styles',  action='store_true',
                   help='Run from all 3 body styles and report best')
    p.add_argument('--steps',       type=int,   default=500,
                   help='Gradient descent steps')
    p.add_argument('--lr',          type=float, default=5e-3)
    p.add_argument('--lambda-cl',   type=float, default=0.5)
    p.add_argument('--lambda-prox', type=float, default=2.0,
                   help='Proximity regularisation weight — higher = smaller geometric changes '
                        '(0 = unconstrained). Default 2.0 prevents cross-dataset extrapolation.')
    p.add_argument('--checkpoint',  default=_DEFAULT_CKPT,
                   help='Path to unified ParaKoop checkpoint')
    p.add_argument('--json',        default=None,
                   help='Save result dict to this JSON file')
    args = p.parse_args()

    # ── Check checkpoint ──────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        print("Train first with:  python scripts/train_unified.py")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model   = load_model(args.checkpoint)
    dataset = load_unified(verbose=False)

    # ── Run inverse design ────────────────────────────────────────────────────
    if args.all_styles:
        results = batch_suggest(
            model, dataset,
            target_cd   = args.target_cd,
            target_cl   = args.target_cl,
            n_restarts  = 3,
            n_steps     = args.steps,
            lr          = args.lr,
            lambda_cl   = args.lambda_cl,
            lambda_prox = args.lambda_prox,
        )
        print(f"\n{'='*54}")
        print(f"  All-styles results (sorted by |Cd error|):")
        print(f"{'='*54}")
        for i, r in enumerate(results, 1):
            cd_err = abs(r['achieved_cd'] - args.target_cd)
            print(f"\n  [{i}] start={r['start_params']['style']}  "
                  f"achieved Cd={r['achieved_cd']:.4f}  err={cd_err:.4f}")
            for param, (v0, v1, d) in r['delta'].items():
                sign = '+' if d > 0 else ''
                unit = '°' if 'deg' in param else ('mm' if '_mm' in param else '')
                print(f"      {param:<22}: {v0}{unit} → {v1}{unit}  ({sign}{d:.4f})")
        result = results[0]  # best one for JSON output
    else:
        result = suggest_geometry(
            model, dataset,
            target_cd   = args.target_cd,
            target_cl   = args.target_cl,
            start_style = args.style,
            n_steps     = args.steps,
            lr          = args.lr,
            lambda_cl   = args.lambda_cl,
            lambda_prox = args.lambda_prox,
            verbose     = True,
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.json:
        # Convert numpy arrays to lists for JSON serialisation
        out = {k: v.tolist() if hasattr(v, 'tolist') else v
               for k, v in result.items()}
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResult saved to: {args.json}")


if __name__ == '__main__':
    main()
