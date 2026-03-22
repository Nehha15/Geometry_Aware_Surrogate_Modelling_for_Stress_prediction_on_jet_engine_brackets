"""
train_tabular.py
================
Complete end-to-end training, evaluation and visualization
for the feature-based DeepJEB stress surrogate.

No xyzdmlc.npz needed — runs entirely from bracket_labels.csv.

What this script does:
  1. Loads bracket_labels.csv  (2138 brackets x 4 directions = 8552 samples)
  2. Builds 23-dim feature vectors (19 geometry + 4 one-hot load direction)
  3. Trains a deep MLP with residual connections, SiLU activations, BN, Dropout
  4. Evaluates on held-out test set: R2, RMSE, MAE, MAPE per load direction
  5. Generates all 7 visualization plots to outputs/tabular/

Usage:
    python scripts/train_tabular.py
    python scripts/train_tabular.py --csv data/raw/bracket_labels.csv --epochs 200
    python scripts/train_tabular.py --model residual --epochs 150
"""

import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Try torch import ────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data_tabular.csv_loader import (
    load_csv_dataset, YIELD_MPa, LOAD_DIR_IDX
)

if HAS_TORCH:
    from src.data_tabular.stress_mlp import StressMLP, ResidualStressMLP


# ─────────────────────────────────────────────────────────────────
#  Training loop (PyTorch)
# ─────────────────────────────────────────────────────────────────

def train_model(splits, cfg, device):
    """Train the MLP and return (model, history)."""

    train = splits["train"]
    val   = splits["val"]

    X_tr = torch.tensor(train["X_scaled"], dtype=torch.float32)
    y_tr = torch.tensor(train["y"],        dtype=torch.float32).unsqueeze(1)
    X_v  = torch.tensor(val["X_scaled"],   dtype=torch.float32)
    y_v  = torch.tensor(val["y"],          dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          shuffle=True, drop_last=True)

    # Model
    if cfg["model"] == "residual":
        model = ResidualStressMLP(
            n_features=X_tr.shape[1],
            width=cfg["width"],
            n_blocks=cfg["n_blocks"],
            dropout=cfg["dropout"],
        ).to(device)
    else:
        model = StressMLP(
            n_features=X_tr.shape[1],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {cfg['model']}  |  {n_params:,} parameters")

    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    best_val   = float("inf")
    patience_c = 0
    history    = {"train_loss": [], "val_loss": [], "lr": []}
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            loss += 0.05 * torch.mean(F.relu(-pred) ** 2)  # physics reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        train_loss = ep_loss / len(train_dl)

        # Validate
        model.eval()
        with torch.no_grad():
            vp   = model(X_v.to(device))
            vloss = F.huber_loss(vp, y_v.to(device)).item()

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vloss)
        history["lr"].append(lr)

        if epoch % 20 == 0 or epoch == cfg["epochs"]:
            print(f"  Epoch {epoch:4d}/{cfg['epochs']}  "
                  f"train={train_loss:.4f}  val={vloss:.4f}  lr={lr:.2e}")

        if vloss < best_val - 1e-5:
            best_val   = vloss
            patience_c = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
            if patience_c >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_model(model, splits, device):
    """Full evaluation on test set. Returns report dict."""
    model.eval()
    test = splits["test"]
    X_t  = torch.tensor(test["X_scaled"], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_log = model(X_t).cpu().numpy().flatten()

    true_log = test["y"]
    true_mpa = np.expm1(true_log)
    pred_mpa = np.expm1(pred_log)
    rel_err  = 100 * np.abs((pred_mpa - true_mpa) / (true_mpa + 1e-6))

    def _metrics(pl, tl):
        pm = np.expm1(pl); tm = np.expm1(tl)
        ss_res = np.sum((tl - pl)**2); ss_tot = np.sum((tl - tl.mean())**2)
        return {
            "r2":         float(1 - ss_res / (ss_tot + 1e-10)),
            "rmse_log":   float(np.sqrt(np.mean((pl-tl)**2))),
            "rmse_mpa":   float(np.sqrt(np.mean((pm-tm)**2))),
            "mae_mpa":    float(np.mean(np.abs(pm-tm))),
            "mape_pct":   float(100*np.mean(np.abs((pm-tm)/(tm+1e-6)))),
            "max_err_mpa":float(np.max(np.abs(pm-tm))),
            "within_5pct": float(np.mean(100*np.abs((pm-tm)/(tm+1e-6)) < 5)),
            "within_10pct":float(np.mean(100*np.abs((pm-tm)/(tm+1e-6)) < 10)),
        }

    overall = _metrics(pred_log, true_log)
    dirs     = np.array(test["dirs"])
    per_dir  = {}
    for d in ("ver","hor","dia","tor"):
        mask = dirs == d
        if mask.sum() > 0:
            per_dir[d] = _metrics(pred_log[mask], true_log[mask])

    # Safety factor classification
    true_fos = YIELD_MPa / (true_mpa + 1e-6)
    pred_fos = YIELD_MPa / (pred_mpa + 1e-6)
    fos_acc  = float(np.mean((true_fos >= 1.0) == (pred_fos >= 1.0)))

    _print_report(overall, per_dir, fos_acc, len(pred_log))

    return {
        "metrics":       overall,
        "per_dir":       per_dir,
        "fos_accuracy":  fos_acc,
        "pred_log":      pred_log.tolist(),
        "true_log":      true_log.tolist(),
        "dirs":          dirs.tolist(),
        "meta":          test["meta"],
    }


def _print_report(m, per_dir, fos_acc, n):
    DIR_NAMES = {"ver":"Vertical","hor":"Horizontal","dia":"Diagonal","tor":"Torsional"}
    print(f"\n{'='*55}")
    print(f"  TEST SET RESULTS  ({n} samples)")
    print(f"{'='*55}")
    print(f"  R²            : {m['r2']:.4f}")
    print(f"  RMSE (MPa)    : {m['rmse_mpa']:.2f}")
    print(f"  MAE  (MPa)    : {m['mae_mpa']:.2f}")
    print(f"  MAPE          : {m['mape_pct']:.2f}%")
    print(f"  Max error     : {m['max_err_mpa']:.2f} MPa")
    print(f"  Within  5%    : {m['within_5pct']*100:.1f}%")
    print(f"  Within 10%    : {m['within_10pct']*100:.1f}%")
    print(f"  FoS accuracy  : {fos_acc*100:.1f}%")
    print(f"\n  Per load direction:")
    for d, dm in sorted(per_dir.items()):
        print(f"    {DIR_NAMES.get(d,d):<12}: R²={dm['r2']:.3f}  "
              f"MAPE={dm['mape_pct']:.2f}%  RMSE={dm['rmse_mpa']:.1f} MPa")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────
#  Visualizations
# ─────────────────────────────────────────────────────────────────

DIR_COLORS = {"ver":"#2E75B6","hor":"#E87722","dia":"#2EAA4F","tor":"#C00000"}
DIR_FULL   = {"ver":"Vertical","hor":"Horizontal","dia":"Diagonal","tor":"Torsional"}
DIR_ORDER  = ["ver","hor","dia","tor"]


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_all(report, history, splits, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    pred_mpa = np.expm1(np.array(report["pred_log"]))
    true_mpa = np.expm1(np.array(report["true_log"]))
    dirs     = np.array(report["dirs"])
    rel_err  = 100 * np.abs((pred_mpa - true_mpa) / (true_mpa + 1e-6))
    failures = rel_err > 10.0

    # ── 1. Dataset overview ─────────────────────────────────────
    meta_all = splits["train"]["meta"] + splits["val"]["meta"] + splits["test"]["meta"]
    all_vm   = np.array([m["vm_mpa"]  for m in meta_all])
    all_dir  = np.array([m["dir"]     for m in meta_all])
    all_mass = np.array([m["mass_kg"] for m in meta_all])
    all_f1   = np.array([m["freq1_hz"]for m in meta_all])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"DeepJEB Dataset Overview — {len(meta_all)} samples\n"
                 f"Ti-6Al-4V  |  Real Nonlinear FEA  |  4 Load Directions",
                 fontsize=13, fontweight="bold")
    ax = axes[0,0]
    data_v = [all_vm[all_dir==d].tolist() for d in DIR_ORDER]
    parts  = ax.violinplot(data_v, showmedians=True)
    for pc, d in zip(parts["bodies"], DIR_ORDER):
        pc.set_facecolor(DIR_COLORS[d]); pc.set_alpha(0.7)
    ax.axhline(YIELD_MPa, color="red", ls="--", lw=1.5, label=f"Yield {YIELD_MPa} MPa")
    ax.set_xticks(range(1,5)); ax.set_xticklabels([DIR_FULL[d] for d in DIR_ORDER])
    ax.set_ylabel("Von Mises Stress (MPa)"); ax.set_title("Stress distribution")
    ax.legend(fontsize=9)

    ax = axes[0,1]
    for d in DIR_ORDER:
        m_ = all_dir==d
        ax.scatter(all_mass[m_], all_vm[m_], c=DIR_COLORS[d],
                   label=DIR_FULL[d], alpha=0.4, s=8, edgecolors="none")
    ax.axhline(YIELD_MPa, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Mass (kg)"); ax.set_ylabel("Stress (MPa)")
    ax.set_title("Stress vs bracket mass"); ax.legend(fontsize=8)

    ax = axes[1,0]
    ax.hist(all_vm, bins=60, color="#2E75B6", alpha=0.7, edgecolor="white")
    ax.axvline(YIELD_MPa, color="red", ls="--", lw=1.5, label=f"Yield {YIELD_MPa} MPa")
    ax.set_xlabel("Von Mises Stress (MPa)"); ax.set_title("Overall distribution")
    ax.legend()

    ax = axes[1,1]
    for d in DIR_ORDER:
        m_ = all_dir==d
        ax.scatter(all_f1[m_], all_vm[m_], c=DIR_COLORS[d],
                   label=DIR_FULL[d], alpha=0.4, s=8, edgecolors="none")
    ax.axhline(YIELD_MPa, color="red", ls="--", lw=1.5)
    ax.set_xlabel("1st modal frequency (Hz)"); ax.set_ylabel("Stress (MPa)")
    ax.set_title("Stress vs 1st modal frequency"); ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/1_dataset_overview.png")

    # ── 2. Training curves ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Curves — Feature-Based MLP Surrogate",
                 fontsize=12, fontweight="bold")
    ep = range(1, len(history["train_loss"])+1)
    ax = axes[0]
    ax.plot(ep, history["train_loss"], "#2E75B6", lw=2, label="Train loss")
    ax.plot(ep, history["val_loss"],   "#E87722", lw=2, ls="--", label="Val loss")
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color="gray", ls=":", lw=1, label=f"Best epoch {best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Loss curves"); ax.legend(); ax.set_yscale("log")
    ax = axes[1]
    ax.plot(ep, history["lr"], "#2EAA4F", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("Cosine annealing LR schedule"); ax.set_yscale("log")
    plt.tight_layout()
    _save(fig, f"{plots_dir}/2_training_curves.png")

    # ── 3. Predicted vs actual ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Predicted vs Actual Von Mises Stress\n"
                 "Feature-Based MLP  |  DeepJEB  |  Ti-6Al-4V",
                 fontsize=12, fontweight="bold")
    ax = axes[0]
    for d in DIR_ORDER:
        mask = dirs==d
        ax.scatter(true_mpa[mask], pred_mpa[mask], c=DIR_COLORS[d],
                   label=DIR_FULL[d], alpha=0.4, s=10, edgecolors="none")
    lim = max(true_mpa.max(), pred_mpa.max())*1.05
    ax.plot([0,lim],[0,lim],"k--",lw=1.5,label="Perfect")
    ax.fill_between([0,lim],[0,0.9*lim],[0,1.1*lim],alpha=0.07,color="green",label="±10%")
    ax.axvline(YIELD_MPa,color="red",ls=":",lw=1); ax.axhline(YIELD_MPa,color="red",ls=":",lw=1)
    ax.set_xlabel("True (MPa)"); ax.set_ylabel("Predicted (MPa)")
    ax.set_title("All load directions"); ax.legend(fontsize=8)
    m_ = report["metrics"]
    ax.text(0.05,0.95,f"R²={m_['r2']:.4f}\nRMSE={m_['rmse_mpa']:.1f} MPa\nMAPE={m_['mape_pct']:.2f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white",alpha=0.8,boxstyle="round,pad=0.3"))

    ax = axes[1]
    pd_  = report["per_dir"]
    dav  = [d for d in DIR_ORDER if d in pd_]
    r2s  = [pd_[d]["r2"]       for d in dav]
    mps  = [pd_[d]["mape_pct"] for d in dav]
    x    = np.arange(len(dav)); w=0.35
    ax.bar(x-w/2, r2s, w, color=[DIR_COLORS[d] for d in dav], alpha=0.85, edgecolor="white")
    ax2  = ax.twinx()
    ax2.bar(x+w/2, mps, w, color=[DIR_COLORS[d] for d in dav], alpha=0.5, edgecolor="white", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels([DIR_FULL[d] for d in dav])
    ax.set_ylabel("R²"); ax2.set_ylabel("MAPE (%)"); ax.set_ylim(0,1.1)
    ax.set_title("Per-direction accuracy")
    for i,(v,d) in enumerate(zip(r2s,dav)):
        ax.text(i-w/2, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/3_predicted_vs_actual.png")

    # ── 4. Error distribution ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Relative Error Distribution by Load Direction",
                 fontsize=12, fontweight="bold")
    ax = axes[0,0]
    ax.hist(rel_err, bins=60, color="#2E75B6", alpha=0.75, edgecolor="white")
    ax.axvline(10, color="red", ls="--", lw=1.5, label="10% threshold")
    ax.axvline(np.median(rel_err), color="green", ls="--", lw=1.5,
               label=f"Median {np.median(rel_err):.1f}%")
    ax.set_xlabel("Relative error (%)"); ax.set_title("All directions")
    ax.legend(fontsize=8)

    for ax, d in zip(axes.flat[1:], DIR_ORDER):
        mask = dirs==d
        e    = rel_err[mask]
        ax.hist(e, bins=40, color=DIR_COLORS[d], alpha=0.75, edgecolor="white")
        ax.axvline(10, color="red", ls="--", lw=1.5)
        ax.axvline(np.median(e), color="black", ls="--", lw=1)
        ax.set_xlabel("Relative error (%)")
        ax.set_title(f"{DIR_FULL[d]} (n={mask.sum()})\n"
                     f"Median={np.median(e):.1f}%  p90={np.percentile(e,90):.1f}%")

    ax = axes[1,2]
    for d in DIR_ORDER:
        mask = dirs==d
        e    = np.sort(rel_err[mask])
        ax.plot(e, np.arange(1,len(e)+1)/len(e), color=DIR_COLORS[d], lw=2, label=DIR_FULL[d])
    ax.axvline(10, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Relative error (%)"); ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of relative error"); ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, f"{plots_dir}/4_error_distribution.png")

    # ── 5. Failure analysis ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"Failure Analysis (>10% error)\n"
                 f"{failures.sum()}/{len(rel_err)} samples flagged ({100*failures.mean():.1f}%)",
                 fontsize=12, fontweight="bold")
    ax = axes[0]
    lim = max(true_mpa.max(), pred_mpa.max())*1.05
    ax.scatter(true_mpa[~failures], pred_mpa[~failures],
               c="#2E75B6", alpha=0.3, s=8, label="OK", edgecolors="none")
    ax.scatter(true_mpa[failures],  pred_mpa[failures],
               c="red", alpha=0.7, s=20, label="Failure",
               edgecolors="darkred", linewidths=0.3)
    ax.plot([0,lim],[0,lim],"k--",lw=1.5)
    ax.fill_between([0,lim],[0,0.9*lim],[0,1.1*lim],alpha=0.07,color="green")
    ax.axvline(YIELD_MPa,color="red",ls=":",lw=1)
    ax.set_xlabel("True (MPa)"); ax.set_ylabel("Predicted (MPa)")
    ax.set_title("Failures in red"); ax.legend(fontsize=9)

    ax = axes[1]
    f_rates = []
    for d in DIR_ORDER:
        mask = dirs==d
        f_rates.append(100*failures[mask].mean() if mask.sum() else 0)
    bars = ax.bar([DIR_FULL[d] for d in DIR_ORDER], f_rates,
                  color=[DIR_COLORS[d] for d in DIR_ORDER], edgecolor="white")
    ax.axhline(10, color="red", ls="--", lw=1.5, label="10% threshold")
    for bar, v in zip(bars, f_rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Failure rate (%)"); ax.set_title("Failure rate per direction")
    ax.legend()
    plt.tight_layout()
    _save(fig, f"{plots_dir}/5_failure_analysis.png")

    # ── 6. Safety factor ────────────────────────────────────────
    true_fos = YIELD_MPa / (true_mpa + 1e-6)
    pred_fos = YIELD_MPa / (pred_mpa + 1e-6)
    fos_acc  = np.mean((true_fos >= 1.0) == (pred_fos >= 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"Safety Factor Analysis  (Ti-6Al-4V yield = {YIELD_MPa} MPa)",
                 fontsize=12, fontweight="bold")
    ax = axes[0]
    lim = min(max(true_fos.max(), pred_fos.max())*1.05, 2.5)
    for d in DIR_ORDER:
        mask = dirs==d
        ax.scatter(true_fos[mask], pred_fos[mask], c=DIR_COLORS[d],
                   label=DIR_FULL[d], alpha=0.4, s=10, edgecolors="none")
    ax.plot([0,lim],[0,lim],"k--",lw=1.5)
    ax.axvline(1.0,color="red",ls="--",lw=1.5,label="Yield (FoS=1)")
    ax.axhline(1.0,color="red",ls="--",lw=1.5)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel("True FoS"); ax.set_ylabel("Predicted FoS")
    ax.set_title("Safety factor prediction"); ax.legend(fontsize=8)

    ax = axes[1]
    bins = np.linspace(0, 2.5, 60)
    ax.hist(np.clip(true_fos,0,2.5), bins=bins, alpha=0.55, color="#2E75B6",
            label="True FoS", edgecolor="white")
    ax.hist(np.clip(pred_fos,0,2.5), bins=bins, alpha=0.55, color="#E87722",
            label="Predicted FoS", edgecolor="white")
    ax.axvline(1.0, color="red", lw=2, label="Yield (FoS=1)")
    ax.set_xlabel("Safety Factor"); ax.set_ylabel("Count")
    ax.set_title("Safety factor distribution"); ax.legend(fontsize=9)
    ax.text(0.65, 0.92, f"Yield classification\naccuracy: {fos_acc*100:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white",alpha=0.8,boxstyle="round,pad=0.3"))
    plt.tight_layout()
    _save(fig, f"{plots_dir}/6_safety_factor.png")

    # ── 7. Feature importance (permutation) ─────────────────────
    # For tabular models we can compute permutation importance
    # (not possible for PointNet++ with raw point clouds)
    meta_t   = report["meta"]
    masses   = np.array([m["mass_kg"]  for m in meta_t])
    freqs    = np.array([m["freq1_hz"] for m in meta_t])
    vols     = np.array([m["volume"]   for m in meta_t]) / 1e3
    nodes    = np.array([m["nodes"]    for m in meta_t]) / 1e3

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("True Stress vs Geometric Properties\n"
                 "Color = relative prediction error (%)",
                 fontsize=12, fontweight="bold")
    for (vals, label, ax) in [
        (masses, "Bracket mass (kg)",     axes[0,0]),
        (vols,   "Volume (cm\u00B3)",     axes[0,1]),
        (nodes,  "Node count (k)",        axes[1,0]),
        (freqs,  "1st modal freq (Hz)",   axes[1,1]),
    ]:
        n_min = min(len(vals), len(true_mpa), len(rel_err))
        sc = ax.scatter(vals[:n_min], true_mpa[:n_min],
                        c=rel_err[:n_min], cmap="RdYlGn_r",
                        vmin=0, vmax=30, alpha=0.6, s=10, edgecolors="none")
        ax.axhline(YIELD_MPa, color="red", ls="--", lw=1.5)
        ax.set_xlabel(label); ax.set_ylabel("True Stress (MPa)")
        ax.set_title(f"Stress vs {label}")
        plt.colorbar(sc, ax=ax, label="Rel. error (%)")
    plt.tight_layout()
    _save(fig, f"{plots_dir}/7_stress_vs_geometry.png")

    print(f"\nAll 7 plots saved to: {plots_dir}/")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train feature-based DeepJEB stress surrogate"
    )
    parser.add_argument("--csv",      default="data/raw/bracket_labels.csv")
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--model",    default="residual",
                        choices=["mlp","residual"])
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out",      default="outputs/tabular")
    parser.add_argument("--ckpt",     default="checkpoints/tabular_best.pth")
    args = parser.parse_args()

    print("\n" + "="*58)
    print("  DeepJEB Feature-Based Stress Surrogate")
    print("  No xyzdmlc.npz needed — runs from bracket_labels.csv")
    print("="*58)

    # ── 1. Load data ────────────────────────────────────────────
    print(f"\n[1/4] Loading dataset: {args.csv}")
    splits, scaler, feat_names = load_csv_dataset(
        args.csv, seed=args.seed,
    )

    # ── 2. Train ────────────────────────────────────────────────
    if not HAS_TORCH:
        print("\nERROR: PyTorch not installed.")
        print("Install with: pip install torch")
        print("\nRunning numpy-only baseline (no training) for demo ...\n")
        _numpy_baseline(splits, args.out)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/4] Training  (device={device})")

    cfg = {
        "model":       args.model,
        "epochs":      args.epochs,
        "lr":          args.lr,
        "wd":          1e-4,
        "batch_size":  args.batch,
        "dropout":     0.3,
        "patience":    30,
        # mlp config
        "hidden_dims": (256, 256, 128, 128, 64),
        # residual config
        "width":       256,
        "n_blocks":    4,
    }
    torch.manual_seed(args.seed)
    model, history = train_model(splits, cfg, device)

    # Save checkpoint
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_scale":     scaler.scale_.tolist(),
        "feat_names":       feat_names,
        "cfg":              cfg,
    }, args.ckpt)
    print(f"  Checkpoint saved: {args.ckpt}")

    # ── 3. Evaluate ─────────────────────────────────────────────
    print(f"\n[3/4] Evaluating on test set ...")
    report = evaluate_model(model, splits, device)

    # Save JSON report
    os.makedirs("data/results", exist_ok=True)
    rpt = {k:v for k,v in report.items() if k not in ("pred_log","true_log","dirs","meta")}
    with open("data/results/tabular_report.json","w") as f:
        json.dump(rpt, f, indent=2)

    # ── 4. Visualize ────────────────────────────────────────────
    print(f"\n[4/4] Generating visualizations ...")
    plot_all(report, history, splits, args.out)

    print("\nDone. Summary:")
    m = report["metrics"]
    print(f"  R²    = {m['r2']:.4f}")
    print(f"  MAPE  = {m['mape_pct']:.2f}%")
    print(f"  RMSE  = {m['rmse_mpa']:.1f} MPa")
    print(f"  FoS accuracy = {report['fos_accuracy']*100:.1f}%")


def _numpy_baseline(splits, out_dir):
    """Median-per-direction baseline when torch is unavailable."""
    import numpy as np

    test  = splits["test"]
    train = splits["train"]
    dirs  = np.array(test["dirs"])
    true  = np.expm1(test["y"])
    preds = np.zeros_like(true)

    for d in ("ver","hor","dia","tor"):
        tr_mask = np.array(train["dirs"]) == d
        te_mask = dirs == d
        if tr_mask.sum() > 0 and te_mask.sum() > 0:
            preds[te_mask] = np.median(np.expm1(train["y"][tr_mask]))

    rel_err = 100 * np.abs((preds-true)/(true+1e-6))
    print(f"Median baseline MAPE: {rel_err.mean():.2f}%")
    print(f"(Train the MLP for much better results)")

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(true, preds, alpha=0.4, s=10)
    lim = max(true.max(), preds.max())*1.05
    ax.plot([0,lim],[0,lim],"k--")
    ax.set_xlabel("True (MPa)"); ax.set_ylabel("Predicted (MPa)")
    ax.set_title(f"Median baseline  MAPE={rel_err.mean():.1f}%")
    fig.savefig(f"{out_dir}/baseline_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Baseline plot saved: {out_dir}/baseline_scatter.png")


if __name__ == "__main__":
    main()
