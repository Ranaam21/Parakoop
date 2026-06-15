"""
scripts/train_unified.py

Unified ParaKoop training over all three data sources:
  - DrivAerNet++ 8K CSV   (Cd, theta imputed)
  - DrivAerNet++ 1163 STL (Cd, real geometry theta)
  - AhmedML 499 CSV       (Cd + Cl, geometry theta)
  + AhmedML 76 VTU runs   (phi grounding, optional — requires pyvista)

Usage
-----
    python scripts/train_unified.py
    python scripts/train_unified.py --epochs 500 --lr 1e-3
    python scripts/train_unified.py --no-phi   # skip VTU grounding
    python scripts/train_unified.py --epochs 10  # smoke test
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_pipeline.unified_loader import load_unified, _AH_DIR
from koopman.model   import ParaKoopModel
from koopman.trainer import KoopmanTrainer, TrainerConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=300)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--batch-size',  type=int,   default=64)
    p.add_argument('--koopman-dim', type=int,   default=128)
    p.add_argument('--rank',        type=int,   default=16)
    p.add_argument('--lambda-cl',   type=float, default=0.5,
                   help='Weight for Cl loss on AhmedML rows')
    p.add_argument('--lambda-phi',  type=float, default=0.3,
                   help='Weight for phi-path (VTU grounding) loss')
    p.add_argument('--lambda-fp',   type=float, default=0.1,
                   help='Weight for fixed-point residual loss')
    p.add_argument('--no-phi',      action='store_true',
                   help='Skip VTU phi grounding (useful if pyvista not installed)')
    p.add_argument('--checkpoint',  default='checkpoints/unified/',
                   help='Directory to save best checkpoint')
    p.add_argument('--log-every',   type=int,   default=10)
    args = p.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ParaKoop Unified Training")
    print("=" * 60)
    dataset = load_unified(verbose=True)

    phi_data = None
    if not args.no_phi:
        print("\nLoading AhmedML VTU flow fields (phi grounding)...")
        phi_data = dataset.load_phi_data(ahmed_dir=_AH_DIR, verbose=True)
        if phi_data is None:
            print("  Phi data not available — training without phi grounding.")
        else:
            print(f"  {len(phi_data[0])} VTU runs loaded for phi path.")

    # ── Build model ───────────────────────────────────────────────────────────
    # phi_dim only matters for phi path; use full 13824 so lifter can handle VTU runs
    phi_dim = 13_824 if phi_data is not None else 1
    model = ParaKoopModel(
        phi_dim       = phi_dim,
        theta_dim     = 8,
        koopman_dim   = args.koopman_dim,
        operator_rank = args.rank,
        hidden_lift   = 512 if phi_data is not None else 64,
        hidden_op     = 64,
        lambda_perf   = 1.0,
        lambda_ae     = 0.1  if phi_data is not None else 0.0,
        lambda_fp_cd  = args.lambda_fp,
    )
    print(f"\nModel: theta_dim=8  koopman_dim={args.koopman_dim}  rank={args.rank}")
    counts = model.param_count()
    print(f"  Total params: {counts['total']:,}  "
          f"(z_net={counts.get('z_net', '?')}  op={counts['operator']:,}  "
          f"perf={counts['perf_head']:,})")

    # ── Train ─────────────────────────────────────────────────────────────────
    cfg = TrainerConfig(
        n_epochs   = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        log_every  = args.log_every,
        save_best  = True,
    )
    trainer = KoopmanTrainer(model, cfg)
    trainer.fit_unified(
        unified_dataset = dataset,
        checkpoint_dir  = args.checkpoint,
        lambda_cl       = args.lambda_cl,
        lambda_phi      = args.lambda_phi,
        phi_data        = phi_data,
    )

    print(f"\nCheckpoint saved to: {args.checkpoint}parakoop_unified_best.pt")
    print("Run inverse design with:")
    print(f"  python scripts/inverse_suggest.py --target-cd 0.240 "
          f"--checkpoint {args.checkpoint}parakoop_unified_best.pt")


if __name__ == '__main__':
    main()
