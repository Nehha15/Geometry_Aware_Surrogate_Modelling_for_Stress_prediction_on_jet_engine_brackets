"""
visualize.py
============
End-to-end visualization for the DeepJEB stress surrogate.

Produces 8 publication-quality plots:

  1.  dataset_overview.png        — stress distribution per load direction
  2.  training_curves.png         — loss + LR curves over epochs
  3.  predicted_vs_actual.png     — scatter plot coloured by load direction
  4.  error_distribution.png      — relative error histogram per direction
  5.  failure_analysis.png        — worst-case bracket predictions
  6.  per_direction_metrics.png   — R² / MAPE bar charts by direction
  7.  stress_vs_mass.png          — stress vs bracket mass coloured by direction
  8.  safety_factor_dist.png      — safety factor distribution + yield line

Usage:
    from src.evaluation.visualize import VisualizationSuite
    viz = VisualizationSuite(cfg, results_dir="outputs")
    viz.plot_dataset_overview(meta_list)
    viz.plot_training_curves(history)
    viz.plot_all(report, history, meta_list)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

MATERIAL_YIELD = 227.6   # Ti-6Al-4V yield stress (MPa)

DIR_COLORS  = {"ver": "#2E75B6", "hor": "#E87722",
               "dia": "#2EAA4F", "tor": "#C00000"}
DIR_LABELS  = {"ver": "Vertical", "hor": "Horizontal",
               "dia": "Diagonal", "tor": "Torsional"}
DIR_ORDER   = ["ver", "hor", "dia", "tor"]


def _save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")


def denormalize(log_v):
    return float(np.expm1(log_v))


# ─────────────────────────────────────────────────────────────────
#  1. Dataset Overview
# ─────────────────────────────────────────────────────────────────

def plot_dataset_overview(meta_list: list, out_path: str):
    """
    4-panel overview of the dataset stress distribution:
      Top-left:  violin plot — stress distribution per load direction
      Top-right: scatter — stress vs mass coloured by direction
      Bottom-left: histogram — stress (MPa) all directions combined
      Bottom-right: bar — sample count per direction
    """
    by_dir = {d: [] for d in DIR_ORDER}
    masses = {d: [] for d in DIR_ORDER}

    for m in meta_list:
        d  = m.get("load_direction", "ver")
        vm = m.get("von_mises_mpa", 0.0)
        ms = m.get("mass_kg", 0.0)
        if d in by_dir and vm > 0:
            by_dir[d].append(vm)
            masses[d].append(ms)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "DeepJEB Dataset — Stress Distribution Overview\n"
        "Ti–6Al–4V  |  Nonlinear FEA  |  4 Load Directions",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Panel 1: Violin
    ax = axes[0, 0]
    data_v = [by_dir[d] for d in DIR_ORDER if by_dir[d]]
    parts  = ax.violinplot(data_v, showmedians=True, showextrema=True)
    for pc, d in zip(parts["bodies"], DIR_ORDER):
        pc.set_facecolor(DIR_COLORS[d])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.axhline(MATERIAL_YIELD, color="red", ls="--", lw=1.5,
               label=f"Yield = {MATERIAL_YIELD} MPa")
    ax.set_xticks(range(1, len(DIR_ORDER) + 1))
    ax.set_xticklabels([DIR_LABELS[d] for d in DIR_ORDER])
    ax.set_ylabel("Von Mises Stress (MPa)")
    ax.set_title("Stress distribution per load direction")
    ax.legend(fontsize=9)

    # Panel 2: Stress vs mass scatter
    ax = axes[0, 1]
    for d in DIR_ORDER:
        if by_dir[d]:
            ax.scatter(masses[d], by_dir[d], c=DIR_COLORS[d],
                       label=DIR_LABELS[d], alpha=0.5, s=8, edgecolors="none")
    ax.axhline(MATERIAL_YIELD, color="red", ls="--", lw=1.5,
               label=f"Yield = {MATERIAL_YIELD} MPa")
    ax.set_xlabel("Bracket mass (kg)")
    ax.set_ylabel("Von Mises Stress (MPa)")
    ax.set_title("Stress vs bracket mass")
    ax.legend(fontsize=8)

    # Panel 3: Combined histogram
    ax = axes[1, 0]
    all_vm = [v for d in DIR_ORDER for v in by_dir[d]]
    ax.hist(all_vm, bins=60, color="#2E75B6", alpha=0.7, edgecolor="white")
    ax.axvline(MATERIAL_YIELD, color="red", ls="--", lw=1.5,
               label=f"Yield = {MATERIAL_YIELD} MPa")
    ax.set_xlabel("Von Mises Stress (MPa)")
    ax.set_ylabel("Count")
    ax.set_title("Overall stress distribution")
    ax.legend(fontsize=9)

    # Panel 4: Sample counts
    ax = axes[1, 1]
    counts = [len(by_dir[d]) for d in DIR_ORDER]
    bars   = ax.bar([DIR_LABELS[d] for d in DIR_ORDER],
                    counts,
                    color=[DIR_COLORS[d] for d in DIR_ORDER],
                    edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of samples")
    ax.set_title("Samples per load direction")

    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────
#  2. Training Curves
# ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, out_path: str):
    """Loss + LR curves over training epochs."""
    epochs     = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    lr_hist    = history.get("lr", [None]*len(epochs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Curves — DeepJEB Stress Surrogate",
                 fontsize=12, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, color="#2E75B6", lw=2, label="Training loss")
    ax.plot(epochs, val_loss,   color="#E87722", lw=2, label="Validation loss",
            linestyle="--")
    best_ep = int(np.argmin(val_loss)) + 1
    ax.axvline(best_ep, color="gray", ls=":", lw=1,
               label=f"Best epoch {best_ep}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.set_yscale("log")

    # Learning rate
    ax = axes[1]
    if any(lr is not None for lr in lr_hist):
        ax.plot(epochs, lr_hist, color="#2EAA4F", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning rate schedule (cosine annealing)")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "LR history not available",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────
#  3. Predicted vs Actual
# ─────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(report: dict, out_path: str):
    """Scatter of predicted vs actual Von Mises stress (MPa)."""
    pred_log = np.array(report["pred_log"])
    true_log = np.array(report["true_log"])
    dirs     = np.array(report["dirs"])

    pred_mpa = np.array([denormalize(v) for v in pred_log])
    true_mpa = np.array([denormalize(v) for v in true_log])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Predicted vs Actual Von Mises Stress\n"
                 "DeepJEB  |  Ti–6Al–4V  |  Nonlinear FEA",
                 fontsize=12, fontweight="bold")

    # Panel 1: All directions
    ax = axes[0]
    for d in DIR_ORDER:
        mask = dirs == d
        if mask.sum() == 0:
            continue
        ax.scatter(true_mpa[mask], pred_mpa[mask],
                   c=DIR_COLORS[d], label=DIR_LABELS[d],
                   alpha=0.5, s=10, edgecolors="none")

    lim = max(true_mpa.max(), pred_mpa.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="Perfect prediction")
    ax.fill_between([0, lim], [0, 0.9*lim], [0, 1.1*lim],
                    alpha=0.08, color="green", label="±10% band")
    ax.axvline(MATERIAL_YIELD, color="red", ls=":", lw=1,
               label=f"Yield = {MATERIAL_YIELD} MPa")
    ax.axhline(MATERIAL_YIELD, color="red", ls=":", lw=1)
    ax.set_xlabel("True Von Mises Stress (MPa)")
    ax.set_ylabel("Predicted Von Mises Stress (MPa)")
    ax.set_title("All load directions")
    ax.legend(fontsize=8, markerscale=2)

    m = report["metrics"]
    textstr = (f"R² = {m['r2']:.4f}\n"
               f"RMSE = {m['rmse_mpa']:.1f} MPa\n"
               f"MAPE = {m['mape_pct']:.2f}%")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Panel 2: Per-direction R² bars
    ax = axes[1]
    pd   = report.get("per_dir", {})
    dirs_avail = [d for d in DIR_ORDER if d in pd]
    r2s  = [pd[d]["r2"]       for d in dirs_avail]
    mapes= [pd[d]["mape_pct"] for d in dirs_avail]

    x    = np.arange(len(dirs_avail))
    w    = 0.35
    b1   = ax.bar(x - w/2, r2s,  w, color=[DIR_COLORS[d] for d in dirs_avail],
                  label="R²", alpha=0.85, edgecolor="white")
    ax2  = ax.twinx()
    b2   = ax2.bar(x + w/2, mapes, w, color=[DIR_COLORS[d] for d in dirs_avail],
                   label="MAPE (%)", alpha=0.5, edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([DIR_LABELS[d] for d in dirs_avail])
    ax.set_ylabel("R²")
    ax2.set_ylabel("MAPE (%)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-direction accuracy")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)

    for bar, val in zip(b1, r2s):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────
#  4. Error Distribution
# ─────────────────────────────────────────────────────────────────

def plot_error_distribution(report: dict, out_path: str):
    """Relative error histogram per load direction."""
    pred_log = np.array(report["pred_log"])
    true_log = np.array(report["true_log"])
    dirs     = np.array(report["dirs"])

    pred_mpa = np.array([denormalize(v) for v in pred_log])
    true_mpa = np.array([denormalize(v) for v in true_log])
    rel_err  = 100 * np.abs((pred_mpa - true_mpa) / (true_mpa + 1e-6))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Relative Error Distribution by Load Direction",
                 fontsize=12, fontweight="bold")

    # Combined
    ax = axes[0, 0]
    ax.hist(rel_err, bins=60, color="#2E75B6", alpha=0.75, edgecolor="white")
    ax.axvline(10, color="red", ls="--", lw=1.5, label="10% threshold")
    ax.axvline(np.median(rel_err), color="green", ls="--", lw=1.5,
               label=f"Median {np.median(rel_err):.1f}%")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Count")
    ax.set_title("All directions")
    ax.legend(fontsize=8)

    # Per direction
    for ax, d in zip(axes.flat[1:], DIR_ORDER):
        mask = dirs == d
        if mask.sum() == 0:
            ax.set_visible(False)
            continue
        e = rel_err[mask]
        ax.hist(e, bins=40, color=DIR_COLORS[d], alpha=0.75, edgecolor="white")
        ax.axvline(10, color="red", ls="--", lw=1.5)
        ax.axvline(np.median(e), color="black", ls="--", lw=1)
        ax.set_xlabel("Relative error (%)")
        ax.set_title(f"{DIR_LABELS[d]}  (n={mask.sum()})\n"
                     f"Median={np.median(e):.1f}%  p90={np.percentile(e,90):.1f}%")

    # CDF panel
    ax = axes[1, 2]
    for d in DIR_ORDER:
        mask = dirs == d
        if mask.sum() == 0:
            continue
        e      = np.sort(rel_err[mask])
        cdf    = np.arange(1, len(e)+1) / len(e)
        ax.plot(e, cdf, color=DIR_COLORS[d], lw=2, label=DIR_LABELS[d])
    ax.axvline(10, color="red", ls="--", lw=1.5, label="10%")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Cumulative error distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────
#  5. Failure Analysis
# ─────────────────────────────────────────────────────────────────

def plot_failure_analysis(report: dict, out_path: str,
                          threshold_pct: float = 10.0):
    """Identifies and visualises worst-case predictions."""
    pred_log = np.array(report["pred_log"])
    true_log = np.array(report["true_log"])
    dirs     = np.array(report["dirs"])

    pred_mpa = np.array([denormalize(v) for v in pred_log])
    true_mpa = np.array([denormalize(v) for v in true_log])
    rel_err  = 100 * np.abs((pred_mpa - true_mpa) / (true_mpa + 1e-6))
    failures = rel_err > threshold_pct

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"Failure Analysis  (threshold = {threshold_pct}%)\n"
                 f"{failures.sum()}/{len(rel_err)} samples flagged "
                 f"({100*failures.mean():.1f}%)",
                 fontsize=12, fontweight="bold")

    # Scatter: true vs pred, failures highlighted
    ax = axes[0]
    lim = max(true_mpa.max(), pred_mpa.max()) * 1.05
    ax.scatter(true_mpa[~failures], pred_mpa[~failures],
               c="#2E75B6", alpha=0.35, s=8,  label="OK",      edgecolors="none")
    ax.scatter(true_mpa[failures],  pred_mpa[failures],
               c="red",    alpha=0.7,  s=20, label="Failure",  edgecolors="darkred",
               linewidths=0.3)
    ax.plot([0, lim], [0, lim], "k--", lw=1.5)
    ax.fill_between([0, lim], [0, (1-threshold_pct/100)*lim],
                    [0, (1+threshold_pct/100)*lim],
                    alpha=0.08, color="green")
    ax.axvline(MATERIAL_YIELD, color="red", ls=":", lw=1)
    ax.axhline(MATERIAL_YIELD, color="red", ls=":", lw=1)
    ax.set_xlabel("True Stress (MPa)")
    ax.set_ylabel("Predicted Stress (MPa)")
    ax.set_title("Predicted vs actual (failures in red)")
    ax.legend(fontsize=9)

    # Failure rate per direction
    ax = axes[1]
    f_rates = []
    dir_names = []
    for d in DIR_ORDER:
        mask = dirs == d
        if mask.sum() == 0:
            continue
        rate = 100 * failures[mask].mean()
        f_rates.append(rate)
        dir_names.append(DIR_LABELS[d])
    bars = ax.bar(dir_names, f_rates,
                  color=[DIR_COLORS[d] for d in DIR_ORDER[:len(dir_names)]],
                  edgecolor="white")
    ax.axhline(threshold_pct, color="red", ls="--", lw=1.5,
               label=f"Threshold {threshold_pct}%")
    for bar, v in zip(bars, f_rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"Failure rate (% samples > {threshold_pct}% error)")
    ax.set_title("Failure rate per load direction")
    ax.legend()

    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────
#  6. Safety Factor Distribution
# ─────────────────────────────────────────────────────────────────

def plot_safety_factor_distribution(report: dict, meta_list: list,
                                    out_path: str):
    """
    Shows true and predicted safety factors (FoS = yield / sigma_vm).
    FoS < 1 means the bracket is predicted to yield.
    """
    pred_log = np.array(report["pred_log"])
    true_log = np.array(report["true_log"])
    dirs     = np.array(report["dirs"])

    pred_mpa = np.array([denormalize(v) for v in pred_log])
    true_mpa = np.array([denormalize(v) for v in true_log])

    true_fos = MATERIAL_YIELD / (true_mpa + 1e-6)
    pred_fos = MATERIAL_YIELD / (pred_mpa + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Safety Factor Analysis  "
                 f"(yield = {MATERIAL_YIELD} MPa, Ti–6Al–4V)",
                 fontsize=12, fontweight="bold")

    # FoS scatter
    ax = axes[0]
    for d in DIR_ORDER:
        mask = dirs == d
        if mask.sum() == 0:
            continue
        ax.scatter(true_fos[mask], pred_fos[mask],
                   c=DIR_COLORS[d], label=DIR_LABELS[d],
                   alpha=0.5, s=10, edgecolors="none")
    lim = min(max(true_fos.max(), pred_fos.max()) * 1.05, 20)
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="Perfect")
    ax.axvline(1.0, color="red", ls="--", lw=1.5, label="FoS = 1 (yield)")
    ax.axhline(1.0, color="red", ls="--", lw=1.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True Safety Factor")
    ax.set_ylabel("Predicted Safety Factor")
    ax.set_title("True vs predicted safety factor")
    ax.legend(fontsize=8)

    # FoS distribution
    ax = axes[1]
    bins = np.linspace(0, min(true_fos.max(), 15), 60)
    ax.hist(true_fos, bins=bins, alpha=0.55, color="#2E75B6",
            label="True FoS", edgecolor="white")
    ax.hist(pred_fos, bins=bins, alpha=0.55, color="#E87722",
            label="Predicted FoS", edgecolor="white")
    ax.axvline(1.0, color="red", lw=2, label="Yield (FoS = 1)")
    ax.axvline(1.5, color="green", ls="--", lw=1.5,
               label="Design threshold (FoS = 1.5)")
    ax.set_xlabel("Safety Factor")
    ax.set_ylabel("Count")
    ax.set_title("Safety factor distribution")
    ax.legend(fontsize=9)

    # Annotation: classification accuracy
    true_safe = true_fos >= 1.0
    pred_safe = pred_fos >= 1.0
    acc = (true_safe == pred_safe).mean()
    ax.text(0.65, 0.92,
            f"Yield classification\naccuracy: {acc*100:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.3"))

    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────
#  7. Stress vs Geometric Properties
# ─────────────────────────────────────────────────────────────────

def plot_stress_vs_geometry(report: dict, meta_list: list, out_path: str):
    """
    Scatter plots of true stress vs key geometric properties:
      mass, volume, num_nodes, 1st modal frequency
    """
    pred_log = np.array(report["pred_log"])
    true_log = np.array(report["true_log"])

    true_mpa = np.array([denormalize(v) for v in true_log])
    pred_mpa = np.array([denormalize(v) for v in pred_log])
    rel_err  = 100 * np.abs((pred_mpa - true_mpa) / (true_mpa + 1e-6))

    # Trim meta_list to match report length
    n = len(true_mpa)
    metas = meta_list[:n] if len(meta_list) >= n else meta_list

    masses  = np.array([m.get("mass_kg",    0.0) for m in metas])
    vols    = np.array([m.get("volume_mm3", 0.0) for m in metas]) / 1e3  # cm³
    nodes   = np.array([m.get("num_nodes",  0.0) for m in metas]) / 1e3  # k-nodes
    freqs   = np.array([m.get("freq_1st_hz", 0.0) for m in metas])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("True Stress vs Geometric Properties\n"
                 "Color = relative prediction error (%)",
                 fontsize=12, fontweight="bold")

    geom_data = [
        (masses[:n],  "Bracket mass (kg)",    axes[0,0]),
        (vols[:n],    "Volume (cm³)",          axes[0,1]),
        (nodes[:n],   "Node count (k)",        axes[1,0]),
        (freqs[:n],   "1st modal freq (Hz)",   axes[1,1]),
    ]

    for x_vals, x_label, ax in geom_data:
        sc = ax.scatter(x_vals, true_mpa, c=rel_err, cmap="RdYlGn_r",
                        vmin=0, vmax=30, alpha=0.6, s=10, edgecolors="none")
        ax.axhline(MATERIAL_YIELD, color="red", ls="--", lw=1.5,
                   label=f"Yield = {MATERIAL_YIELD} MPa")
        ax.set_xlabel(x_label)
        ax.set_ylabel("True Von Mises Stress (MPa)")
        ax.set_title(f"Stress vs {x_label}")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="Relative error (%)")

    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────
#  Master Suite
# ─────────────────────────────────────────────────────────────────

class VisualizationSuite:
    """
    Produces all plots for the DeepJEB stress surrogate project.

    Usage:
        viz = VisualizationSuite(plots_dir="outputs")
        viz.plot_all(report, history, meta_list)
    """

    def __init__(self, plots_dir: str = "outputs"):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def _path(self, name): return os.path.join(self.plots_dir, name)

    def plot_dataset_overview(self, meta_list: list):
        plot_dataset_overview(meta_list, self._path("1_dataset_overview.png"))

    def plot_training_curves(self, history: dict):
        plot_training_curves(history, self._path("2_training_curves.png"))

    def plot_predicted_vs_actual(self, report: dict):
        plot_predicted_vs_actual(report, self._path("3_predicted_vs_actual.png"))

    def plot_error_distribution(self, report: dict):
        plot_error_distribution(report, self._path("4_error_distribution.png"))

    def plot_failure_analysis(self, report: dict, threshold_pct: float = 10.0):
        plot_failure_analysis(report, self._path("5_failure_analysis.png"),
                              threshold_pct)

    def plot_safety_factor(self, report: dict, meta_list: list):
        plot_safety_factor_distribution(
            report, meta_list, self._path("6_safety_factor_dist.png"))

    def plot_stress_vs_geometry(self, report: dict, meta_list: list):
        plot_stress_vs_geometry(
            report, meta_list, self._path("7_stress_vs_geometry.png"))

    def plot_all(self, report: dict, history: dict, meta_list: list,
                 threshold_pct: float = 10.0):
        print("\nGenerating all visualizations ...")
        self.plot_training_curves(report.get("_history", history))
        self.plot_predicted_vs_actual(report)
        self.plot_error_distribution(report)
        self.plot_failure_analysis(report, threshold_pct)
        self.plot_safety_factor(report, meta_list)
        self.plot_stress_vs_geometry(report, meta_list)
        print(f"\nAll plots saved to: {self.plots_dir}/")
