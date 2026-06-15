"""
scripts/train_drivaernet.py

Train a ParaKoop model on DrivAerNet++ Phase 1 (Cd-only, no flow fields).

Uses the differentiable fixed-point solve path:
    theta → A(theta) → solve (A−I)z* = b → perf_head → Cd_pred

Two data modes (--mode flag):
    full      8,121 designs from DrivAerNetPlusPlus_Cd_8k_Updated.csv
    balanced    500 designs from balanced_sample_500.csv (faster, all 3 styles)

Model config:
    theta_dim = 6   [is_fastback, is_notchback, is_estateback,
                     is_detailed, has_mirrors, has_wheels]
    phi_dim   = 1   (nominal — lifter/decoder unused in this training mode)
    koopman_dim     controlled by --koopman-dim (default 128)

Usage
-----
    cd parakoop/
    python scripts/train_drivaernet.py
    python scripts/train_drivaernet.py --mode balanced --epochs 200
    python scripts/train_drivaernet.py --koopman-dim 256 --epochs 500 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from parakoop/ root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from data_pipeline.drivaernet_loader import load_drivaernet
from koopman.model   import ParaKoopModel
from koopman.trainer import KoopmanTrainer, TrainerConfig

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'drivaernet')
CKPT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'drivaernet')

DRIVAERNET_THETA_DIM = 6   # [fastback, notchback, estateback, detailed, mirrors, wheels]


def parse_args():
    pa = argparse.ArgumentParser(description='Train ParaKoop on DrivAerNet++ (Phase 1)')
    pa.add_argument('--mode', choices=['full', 'balanced'], default='full',
                    help='full=8121 designs  balanced=500 stratified sample (default: full)')
    pa.add_argument('--epochs',      type=int,   default=300)
    pa.add_argument('--batch-size',  type=int,   default=64)
    pa.add_argument('--lr',          type=float, default=3e-4)
    pa.add_argument('--koopman-dim', type=int,   default=128,
                    help='Koopman state dimension K (default 128; use 256 for full run)')
    pa.add_argument('--rank',        type=int,   default=16,
                    help='Low-rank operator rank r (default 16)')
    pa.add_argument('--log-every',   type=int,   default=10)
    pa.add_argument('--device',      default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    return pa.parse_args()


def main():
    args = parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading DrivAerNet++ ({args.mode} mode) ...")

    if args.mode == 'balanced':
        import pandas as pd
        import numpy as np
        from data_pipeline.drivaernet_loader import (
            DrivAerNetDataset, DrivAerSample, _parse_id, CD_CSV
        )
        csv_path = os.path.join(DATA_DIR, 'balanced_sample_500.csv')
        df = pd.read_csv(csv_path)
        # balanced CSV has columns: design_id, cd, body
        samples = []
        for _, row in df.iterrows():
            body_style, theta_disc = _parse_id(str(row['design_id']))
            samples.append(DrivAerSample(
                design_id=str(row['design_id']),
                body_style=body_style,
                cd=float(row['cd']),
                theta_discrete=theta_disc,
            ))
        ds = DrivAerNetDataset(samples)
    else:
        ds = load_drivaernet(DATA_DIR)

    print(ds.summary())

    # ── Build model ───────────────────────────────────────────────────────────
    model = ParaKoopModel(
        phi_dim      = 1,                    # nominal — unused in this training mode
        theta_dim    = DRIVAERNET_THETA_DIM,
        koopman_dim  = args.koopman_dim,
        operator_rank= args.rank,
        hidden_lift  = 64,                   # minimal lifter (not used)
        hidden_op    = 64,
        lambda_perf  = 1.0,
        lambda_ae    = 0.0,                  # no AE loss (no phi)
    )

    cfg = TrainerConfig(
        n_epochs    = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        log_every   = args.log_every,
        device      = args.device,
        save_best   = True,
    )

    trainer = KoopmanTrainer(model, cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit_drivaernet(ds, checkpoint_dir=CKPT_DIR)

    # ── Post-training: eigenspectrum sample across body styles ────────────────
    print("\n── Eigenspectrum sample (3 body styles) ──")
    model.eval()
    import numpy as np

    style_thetas = {
        'fastback':   [1, 0, 0, 1, 1, 1],
        'notchback':  [0, 1, 0, 1, 1, 1],
        'estateback': [0, 0, 1, 1, 1, 1],
    }
    device = next(model.parameters()).device
    for style, vals in style_thetas.items():
        theta = torch.tensor(vals, dtype=torch.float32).to(device)
        eigs  = model.get_eigenspectrum(theta)
        kappa = model.condition_number(theta)
        top3  = eigs.abs().sort(descending=True).values[:3]
        print(f"  {style:<12}  kappa={kappa:.1f}  "
              f"|eigs| top3: {top3[0]:.4f}  {top3[1]:.4f}  {top3[2]:.4f}")

    print(f"\nCheckpoint saved to: {CKPT_DIR}/parakoop_drivaernet_best.pt")


if __name__ == '__main__':
    main()
