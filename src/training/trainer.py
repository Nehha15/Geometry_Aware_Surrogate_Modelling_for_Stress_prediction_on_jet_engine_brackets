"""
trainer.py
==========
Training loop — v3.

Changes from v2:
  - Unpacks 6-element batch tuple (added sample weight)
  - Passes weights tensor to StressLoss.forward
  - EarlyStopping on EMA val loss, patience=40
  - Periodic checkpoint every save_every epochs
"""

import os, time, json
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

from src.training.losses import StressLoss
from src.models.tnet import tnet_regularization_loss


class EarlyStopping:
    def __init__(self, patience=40, min_delta=1e-4, ema_alpha=0.3):
        self.patience  = patience
        self.min_delta = min_delta
        self.alpha     = ema_alpha
        self.best      = float("inf")
        self.ema       = None
        self.counter   = 0

    def step(self, val_loss):
        self.ema = val_loss if self.ema is None \
                   else self.alpha * val_loss + (1 - self.alpha) * self.ema
        if self.ema < self.best - self.min_delta:
            self.best    = self.ema
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        tc = cfg["training"]
        lc = cfg.get("loss", {})

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tc.get("scheduler_T_max", 150))

        self.criterion = StressLoss(
            primary=lc.get("primary", "huber"),
            delta=lc.get("huber_delta", 1.0),
            physics_weight=lc.get("physics_weight", 0.05),
        )

        self.tnet_weight = 0.001
        self.grad_clip   = tc.get("grad_clip", 1.0)
        self.save_every  = tc.get("save_every", 10)
        self.ckpt_dir    = tc.get("checkpoint_dir", "checkpoints")
        self.log_dir     = tc.get("log_dir", "logs")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir,  exist_ok=True)

        self.early_stopping = EarlyStopping(
            patience  = tc.get("early_stopping_patience", 40),
            min_delta = tc.get("early_stopping_min_delta", 1e-4),
            ema_alpha = tc.get("early_stopping_ema_alpha", 0.3),
        )
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def _train_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        for pcs, lcs, feats, ys, wts, _ in tqdm(
                self.train_loader, desc="Train", leave=False):
            pcs   = pcs.to(self.device)
            lcs   = lcs.to(self.device)
            feats = feats.to(self.device)
            ys    = ys.to(self.device)
            wts   = wts.to(self.device)

            self.optimizer.zero_grad()
            pred, trans = self.model(pcs, lcs, feats)
            loss, _     = self.criterion(pred, ys, weights=wts)

            if trans is not None:
                loss = loss + self.tnet_weight * tnet_regularization_loss(trans)

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total += loss.item(); n += 1
        return total / max(n, 1)

    def _val_epoch(self):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for pcs, lcs, feats, ys, wts, _ in self.val_loader:
                pcs   = pcs.to(self.device)
                lcs   = lcs.to(self.device)
                feats = feats.to(self.device)
                ys    = ys.to(self.device)
                wts   = wts.to(self.device)
                pred, _ = self.model(pcs, lcs, feats)
                loss, _ = self.criterion(pred, ys, weights=wts)
                total += loss.item(); n += 1
        return total / max(n, 1)

    def train(self, epochs):
        best_val = float("inf")
        t0_total = time.time()

        for epoch in range(1, epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            self.scheduler.step()

            lr  = self.optimizer.param_groups[0]["lr"]
            ema = self.early_stopping.ema or val_loss

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                  f"EMA {ema:.4f} | LR {lr:.2e} | {time.time()-t0:.1f}s")

            if val_loss < best_val:
                best_val = val_loss
                self._save("best_model.pth", epoch, val_loss)
                print(f"  ✓ Best saved (val={val_loss:.4f})")

            if epoch % self.save_every == 0:
                self._save(f"ckpt_epoch{epoch:03d}.pth", epoch, val_loss)

            if self.early_stopping.step(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nDone in {(time.time()-t0_total)/60:.1f} min | best val: {best_val:.4f}")
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def _save(self, name, epoch, val_loss):
        torch.save({
            "epoch": epoch, "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }, os.path.join(self.ckpt_dir, name))