"""
koopman/trainer.py

Training loop for the ParaKoop parametric Koopman model.

Trains on AhmedMLDataset via the fixed-point + performance + AE losses.
DrivAerNet++ (Phase 1, Cd-only) can be mixed in via drivaernet_dataset
for additional performance-head supervision.

Usage
-----
    from koopman.model   import ParaKoopModel
    from koopman.trainer import KoopmanTrainer, TrainerConfig

    model   = ParaKoopModel()
    trainer = KoopmanTrainer(model, TrainerConfig(n_epochs=300))
    trainer.fit(ahmed_ds, checkpoint_dir='checkpoints/')

    # Load a checkpoint
    trainer2 = KoopmanTrainer.load('checkpoints/parakoop_best.pt', model)
"""

from __future__ import annotations

import os
import time
import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclasses.dataclass
class TrainerConfig:
    n_epochs:       int   = 300
    batch_size:     int   = 32
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    val_fraction:   float = 0.15     # held-out fraction for validation
    grad_clip:      float = 1.0
    log_every:      int   = 10       # print every N epochs
    save_best:      bool  = True
    device:         str   = 'auto'   # 'auto' | 'cpu' | 'cuda' | 'mps'


class KoopmanTrainer:
    """Trains ParaKoopModel on AhmedMLDataset."""

    def __init__(self, model: 'ParaKoopModel', cfg: TrainerConfig = TrainerConfig()):
        self.model = model
        self.cfg   = cfg

        if cfg.device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps'  if torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = torch.device(cfg.device)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
        )

        self.best_val_loss = float('inf')
        self.history: dict = {
            'train_loss': [], 'train_fp': [], 'train_perf': [], 'train_ae': [],
            'val_loss': [], 'val_cd_mae': [], 'val_cl_mae': [],
        }

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def fit(self, ahmed_dataset, checkpoint_dir: str = 'checkpoints/') -> None:
        """
        Main training loop.

        Parameters
        ----------
        ahmed_dataset  : AhmedMLDataset (needs .get_arrays())
        checkpoint_dir : directory to save the best model checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        cfg = self.cfg

        train_loader, val_loader, n_train, n_val = self._build_loaders(ahmed_dataset)

        counts = self.model.param_count()
        print(f"\nParaKoop training  [{self.device}]")
        print(f"  Dataset : {n_train} train  /  {n_val} val samples")
        print(f"  Epochs  : {cfg.n_epochs}   Batch: {cfg.batch_size}   LR: {cfg.lr}")
        print(f"  Params  : {counts['total']:,}  "
              f"(lifter={counts['lifter']:,}  op={counts['operator']:,}  "
              f"dec={counts['decoder']:,})")
        print()

        t0 = time.time()

        for epoch in range(1, cfg.n_epochs + 1):
            t_loss, t_fp, t_perf, t_ae = self._train_epoch(train_loader)
            v_loss, v_cd_mae, v_cl_mae = self._val_epoch(val_loader)

            self.history['train_loss'].append(t_loss)
            self.history['train_fp'].append(t_fp)
            self.history['train_perf'].append(t_perf)
            self.history['train_ae'].append(t_ae)
            self.history['val_loss'].append(v_loss)
            self.history['val_cd_mae'].append(v_cd_mae)
            self.history['val_cl_mae'].append(v_cl_mae)

            self.scheduler.step()

            if epoch % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(f"  ep {epoch:4d}/{cfg.n_epochs}  "
                      f"train={t_loss:.5f} (fp={t_fp:.5f} perf={t_perf:.5f} ae={t_ae:.5f})  "
                      f"val={v_loss:.5f}  cd_mae={v_cd_mae:.4f}  [{elapsed:.0f}s]")

            if cfg.save_best and v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                ckpt_path = os.path.join(checkpoint_dir, 'parakoop_best.pt')
                self.save(ckpt_path)
                print(f"    ✓ new best val={v_loss:.5f} → {ckpt_path}")

        elapsed = time.time() - t0
        print(f"\nDone. Best val loss: {self.best_val_loss:.5f}  ({elapsed:.0f}s total)")

    def save(self, path: str) -> None:
        torch.save({
            'model_state':    self.model.state_dict(),
            'history':        self.history,
            'best_val_loss':  self.best_val_loss,
            'model_cfg': {
                'phi_dim':        self.model.phi_dim,
                'theta_dim':      self.model.operator.alpha_net[0].in_features,
                'koopman_dim':    self.model.koopman_dim,
                'operator_rank':  self.model.operator.r,
                'hidden_lift':    self.model.lifter.net[0].out_features,
                'hidden_op':      self.model.operator.alpha_net[0].out_features,
                'lambda_perf':    self.model.lambda_perf,
                'lambda_ae':      self.model.lambda_ae,
                'lambda_fp_cd':   self.model.lambda_fp_cd,
            },
        }, path)

    @classmethod
    def load(
        cls,
        path:  str,
        model: 'ParaKoopModel',
        cfg:   TrainerConfig = TrainerConfig(),
    ) -> 'KoopmanTrainer':
        trainer = cls(model, cfg)
        ckpt    = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        trainer.history       = ckpt.get('history', {})
        trainer.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        return trainer

    def fit_drivaernet(
        self,
        drivaernet_dataset,
        checkpoint_dir: str = 'checkpoints/',
    ) -> None:
        """
        Phase 1 DrivAerNet training: Cd-only supervision via z_net + fixed-point loss.

        z_net(theta) directly parameterizes z* (no linear solve, no instability).
        The fixed-point residual ||(A−I)z* − b||² trains the operator to be
        consistent with z*(theta), forcing A to develop real geometry-dependent
        structure — giving meaningful eigenspectra and κ variation across body styles.

        Loss = MSE(Cd_pred, Cd) + lambda_fp_cd * ||(A−I)z* − b||²

        Parameters
        ----------
        drivaernet_dataset : DrivAerNetDataset  (from data_pipeline.drivaernet_loader)
        checkpoint_dir     : where to save best checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        cfg = self.cfg

        Theta = torch.from_numpy(drivaernet_dataset.get_theta_array()).float()
        Cd    = torch.from_numpy(drivaernet_dataset.get_cd_array()).float()

        full_ds = TensorDataset(Theta, Cd)
        n_val   = max(1, int(len(full_ds) * cfg.val_fraction))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
        )

        counts = self.model.param_count()
        lam    = self.model.lambda_fp_cd
        print(f"\nParaKoop — DrivAerNet Phase 1  [{self.device}]")
        print(f"  Dataset : {n_train} train / {n_val} val  ({len(full_ds)} total designs)")
        print(f"  Epochs  : {cfg.n_epochs}   Batch: {cfg.batch_size}   LR: {cfg.lr}")
        print(f"  Params  : {counts['total']:,}  "
              f"(op={counts['operator']:,}  perf={counts['perf_head']:,})")
        print(f"  Mode    : z_net(theta)→z*  +  fp_loss  (λ={lam})")
        print()

        t0 = time.time()

        for epoch in range(1, cfg.n_epochs + 1):
            # ── train ──────────────────────────────────────────────
            self.model.train()
            train_losses, fp_losses = [], []
            for theta_b, cd_b in train_loader:
                theta_b = theta_b.to(self.device)
                cd_b    = cd_b.to(self.device)
                self.optimizer.zero_grad()
                cd_pred, _, loss_fp = self.model.forward_cd_only(theta_b)
                loss = F.mse_loss(cd_pred, cd_b) + lam * loss_fp
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                train_losses.append(loss.item())
                fp_losses.append(loss_fp.item())

            # ── validate ───────────────────────────────────────────
            self.model.eval()
            val_maes = []
            with torch.no_grad():
                for theta_b, cd_b in val_loader:
                    theta_b = theta_b.to(self.device)
                    cd_b    = cd_b.to(self.device)
                    cd_pred, _, _ = self.model.forward_cd_only(theta_b)
                    val_maes.append(float((cd_pred - cd_b).abs().mean()))

            t_loss = float(np.mean(train_losses))
            t_fp   = float(np.mean(fp_losses))
            v_mae  = float(np.mean(val_maes))

            self.history['train_loss'].append(t_loss)
            self.history['val_cd_mae'].append(v_mae)
            self.scheduler.step()

            if epoch % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(f"  ep {epoch:4d}/{cfg.n_epochs}  "
                      f"train={t_loss:.5f}  fp={t_fp:.5f}  val_cd_mae={v_mae:.5f}  [{elapsed:.0f}s]")

            if cfg.save_best and v_mae < self.best_val_loss:
                self.best_val_loss = v_mae
                ckpt_path = os.path.join(checkpoint_dir, 'parakoop_drivaernet_best.pt')
                self.save(ckpt_path)
                print(f"    ✓ new best val_mae={v_mae:.5f} → {ckpt_path}")

        elapsed = time.time() - t0
        print(f"\nDone. Best val Cd MAE: {self.best_val_loss:.5f}  ({elapsed:.0f}s total)")

    def fit_unified(
        self,
        unified_dataset,
        checkpoint_dir: str = 'checkpoints/unified/',
        lambda_cl:  float = 0.5,
        lambda_phi: float = 0.3,
        phi_data:   Optional[tuple] = None,
    ) -> None:
        """
        Unified multi-source training over DrivAerNet 8K + STL 1163 + AhmedML 499.

        Two interleaved loss paths per epoch:

        z_net path (ALL ~9,600 samples):
            loss_cd  = MSE(cd_pred, cd_true)
            loss_cl  = mean(has_cl * (cl_pred - cl_true)^2)   # AhmedML only
            loss_fp  = lambda_fp_cd * fixed-point residual
            total    = loss_cd + lambda_cl * loss_cl + loss_fp

        phi path (76 VTU runs, optional — grounds A(θ) in real flow physics):
            z_bar       = lifter(phi)
            loss_fp_phi = ||(A(θ)-I)z_bar - b(θ)||²
            loss_ae     = ||decoder(z_bar) - phi||²
            loss_perf   = MSE(perf_head(z_bar), [Cd, Cl])
            total_phi   = loss_fp_phi + loss_ae + loss_perf
            added as: lambda_phi * total_phi

        Parameters
        ----------
        unified_dataset : UnifiedDataset from data_pipeline.unified_loader
        checkpoint_dir  : directory for best-model checkpoint
        lambda_cl       : weight for Cl loss on AhmedML rows
        lambda_phi      : weight for full phi-path loss (VTU grounding)
        phi_data        : (Theta_phi, Phi, Cd_phi, Cl_phi) from dataset.load_phi_data()
                          Pass None to skip phi path (training works without it).
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        cfg = self.cfg

        # ── Build main (z_net) data loaders ──────────────────────────────────
        Theta, Cd, Cl, HasCl = unified_dataset.get_arrays()
        HasCl_f = HasCl.astype(np.float32)

        full_ds = TensorDataset(
            torch.from_numpy(Theta).float(),
            torch.from_numpy(Cd).float(),
            torch.from_numpy(Cl).float(),
            torch.from_numpy(HasCl_f).float(),
        )
        n_val   = max(1, int(len(full_ds) * cfg.val_fraction))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

        # ── Build phi loader (optional) ───────────────────────────────────────
        phi_loader = None
        if phi_data is not None:
            T_phi, Phi, Cd_phi, Cl_phi = phi_data
            phi_ds = TensorDataset(
                torch.from_numpy(T_phi).float(),
                torch.from_numpy(Phi).float(),
                torch.from_numpy(Cd_phi).float(),
                torch.from_numpy(Cl_phi).float(),
            )
            phi_loader = DataLoader(phi_ds, batch_size=min(8, len(phi_ds)), shuffle=True)

        counts = self.model.param_count()
        lam_fp = self.model.lambda_fp_cd
        print(f"\nParaKoop — Unified Training  [{self.device}]")
        print(f"  Dataset : {n_train} train / {n_val} val  ({len(full_ds):,} total)")
        print(f"  Phi VTU : {len(phi_data[0]) if phi_data else 0} runs  (lambda_phi={lambda_phi})")
        print(f"  Epochs  : {cfg.n_epochs}   Batch: {cfg.batch_size}   LR: {cfg.lr}")
        print(f"  Losses  : 0.5×Cd_driv + 0.5×Cd_ahmed(domain-bal) + {lambda_cl}×Cl(masked) + {lam_fp}×fp  [{counts['total']:,} params]")
        print()

        t0 = time.time()
        phi_cycle = iter(phi_loader) if phi_loader else None

        for epoch in range(1, cfg.n_epochs + 1):
            self.model.train()
            train_losses, cd_losses, cl_losses, fp_losses = [], [], [], []

            for theta_b, cd_b, cl_b, hascl_b in train_loader:
                theta_b = theta_b.to(self.device)
                cd_b    = cd_b.to(self.device)
                cl_b    = cl_b.to(self.device)
                hascl_b = hascl_b.to(self.device)

                self.optimizer.zero_grad()

                # z_net path
                cd_pred, cl_pred, loss_fp = self.model.forward_cd_only(theta_b)

                # Domain-balanced Cd loss: AhmedML rows (has_cl=True) are only ~5%
                # of each batch.  Compute per-domain means and average them equally
                # so AhmedML Cd signal is not washed out by DrivAerNet.
                is_ahmed = hascl_b.bool()
                if is_ahmed.any() and (~is_ahmed).any():
                    loss_cd_driv  = F.mse_loss(cd_pred[~is_ahmed], cd_b[~is_ahmed])
                    loss_cd_ahmed = F.mse_loss(cd_pred[ is_ahmed], cd_b[ is_ahmed])
                    loss_cd = 0.5 * loss_cd_driv + 0.5 * loss_cd_ahmed
                else:
                    loss_cd = F.mse_loss(cd_pred, cd_b)

                loss_cl = (hascl_b * (cl_pred - cl_b) ** 2).mean()
                loss    = loss_cd + lambda_cl * loss_cl + lam_fp * loss_fp

                # phi path (optional, interleaved)
                if phi_loader is not None:
                    try:
                        phi_batch = next(phi_cycle)
                    except StopIteration:
                        phi_cycle = iter(phi_loader)
                        phi_batch = next(phi_cycle)
                    tp, phip, cdp, clp = [x.to(self.device) for x in phi_batch]
                    out_phi  = self.model(tp, phip)
                    _, l_fp_phi, l_perf_phi, l_ae_phi = self.model.compute_loss(
                        out_phi, cdp, clp
                    )
                    loss = loss + lambda_phi * (l_fp_phi + l_perf_phi + l_ae_phi)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()

                train_losses.append(loss.item())
                cd_losses.append(loss_cd.item())
                cl_losses.append(loss_cl.item())
                fp_losses.append(loss_fp.item())

            # ── Validation (z_net path only) ──────────────────────────────────
            self.model.eval()
            val_cd_maes, val_cl_maes = [], []
            with torch.no_grad():
                for theta_b, cd_b, cl_b, hascl_b in val_loader:
                    theta_b = theta_b.to(self.device)
                    cd_b    = cd_b.to(self.device)
                    cl_b    = cl_b.to(self.device)
                    hascl_b = hascl_b.to(self.device)
                    cd_pred, cl_pred, _ = self.model.forward_cd_only(theta_b)
                    val_cd_maes.append(float((cd_pred - cd_b).abs().mean()))
                    if hascl_b.any():
                        val_cl_maes.append(
                            float((cl_pred[hascl_b.bool()] - cl_b[hascl_b.bool()]).abs().mean())
                        )

            t_loss  = float(np.mean(train_losses))
            v_cd    = float(np.mean(val_cd_maes))
            v_cl    = float(np.mean(val_cl_maes)) if val_cl_maes else float('nan')

            self.history['train_loss'].append(t_loss)
            self.history['val_cd_mae'].append(v_cd)
            self.scheduler.step()

            if epoch % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(f"  ep {epoch:4d}/{cfg.n_epochs}  "
                      f"train={t_loss:.5f}  "
                      f"cd_mae={float(np.mean(cd_losses)):.5f}  "
                      f"cl_loss={float(np.mean(cl_losses)):.5f}  "
                      f"fp={float(np.mean(fp_losses)):.5f}  "
                      f"val_cd={v_cd:.5f}  val_cl={v_cl:.4f}  [{elapsed:.0f}s]")

            if cfg.save_best and v_cd < self.best_val_loss:
                self.best_val_loss = v_cd
                ckpt_path = os.path.join(checkpoint_dir, 'parakoop_unified_best.pt')
                self.save(ckpt_path)
                print(f"    ✓ new best val_cd={v_cd:.5f} → {ckpt_path}")

        elapsed = time.time() - t0
        print(f"\nDone. Best val Cd MAE: {self.best_val_loss:.5f}  ({elapsed:.0f}s total)")

    # ─────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────

    def _build_loaders(self, ahmed_dataset):
        Theta, Phi, Cd, Cl, _, _ = ahmed_dataset.get_arrays()
        full_ds = TensorDataset(
            torch.from_numpy(Theta).float(),
            torch.from_numpy(Phi).float(),
            torch.from_numpy(Cd).float(),
            torch.from_numpy(Cl).float(),
        )
        n_val   = max(1, int(len(full_ds) * self.cfg.val_fraction))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False
        )
        return train_loader, val_loader, n_train, n_val

    def _train_epoch(self, loader) -> tuple:
        self.model.train()
        losses, fps, perfs, aes = [], [], [], []
        for theta_b, phi_b, cd_b, cl_b in loader:
            theta_b = theta_b.to(self.device)
            phi_b   = phi_b.to(self.device)
            cd_b    = cd_b.to(self.device)
            cl_b    = cl_b.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(theta_b, phi_b)
            loss, l_fp, l_perf, l_ae = self.model.compute_loss(out, cd_b, cl_b)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            losses.append(loss.item())
            fps.append(l_fp.item())
            perfs.append(l_perf.item())
            aes.append(l_ae.item())

        return (float(np.mean(losses)), float(np.mean(fps)),
                float(np.mean(perfs)), float(np.mean(aes)))

    @torch.no_grad()
    def _val_epoch(self, loader) -> tuple:
        self.model.eval()
        losses, cd_maes, cl_maes = [], [], []
        for theta_b, phi_b, cd_b, cl_b in loader:
            theta_b = theta_b.to(self.device)
            phi_b   = phi_b.to(self.device)
            cd_b    = cd_b.to(self.device)
            cl_b    = cl_b.to(self.device)

            out = self.model(theta_b, phi_b)
            loss, *_ = self.model.compute_loss(out, cd_b, cl_b)
            losses.append(loss.item())
            cd_maes.append(float((out['cd_pred'] - cd_b).abs().mean()))
            cl_maes.append(float((out['cl_pred'] - cl_b).abs().mean()))

        return (float(np.mean(losses)),
                float(np.mean(cd_maes)),
                float(np.mean(cl_maes)))
