"""
run_demo.py
===========
End-to-end demonstration using the real DeepJEB bracket_labels.csv.

This script runs WITHOUT PyTorch or xyzdmlc.npz:
  1. Loads bracket_labels.csv → reads real FEA stress labels
  2. Generates geometry proxy point clouds from inertia tensors
  3. Saves processed samples to data/processed/
  4. Generates the dataset overview visualization
  5. Prints full dataset statistics

This proves the complete data pipeline works before committing
to full training. Run this first to verify your setup.

Usage:
    python run_demo.py
    python run_demo.py --csv data/raw/bracket_labels.csv --n_samples 50
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.deepjeb_loader import (
    load_deepjeb_fallback, STRESS_COLUMNS, LOAD_DIRECTION_IDX,
    YIELD_STRESS_MPA, normalize_stress_mpa, denormalize_stress
)

def denormalize(v): return denormalize_stress(v)


def run_demo(csv_path: str, out_dir: str = "data/processed_demo",
             n_pts: int = 512, seed: int = 42):

    print("\n" + "="*60)
    print("  DeepJEB Stress Surrogate — End-to-End Demo")
    print("  Real FEA labels from bracket_labels.csv")
    print("="*60)

    # ── Step 1: Load CSV, show statistics ──────────────────────
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n[1/4] Dataset loaded")
    print(f"  Brackets:        {len(df)}")
    print(f"  Mass range:      {df['mass(kg)'].min():.2f} – "
          f"{df['mass(kg)'].max():.2f} kg")
    print(f"  Node count:      {int(df['num_nodes'].min()):,} – "
          f"{int(df['num_nodes'].max()):,}")
    print()

    dirs = ["ver", "hor", "dia", "tor"]
    for d in dirs:
        col = STRESS_COLUMNS[d]
        vals = df[col].dropna()
        yielded = (vals > YIELD_STRESS_MPA).sum()
        print(f"  {d}: stress [{vals.min():.1f}, {vals.max():.1f}] MPa  "
              f"mean={vals.mean():.1f}  yielded={yielded}/{len(vals)}")

    # ── Step 2: Build proxy dataset ─────────────────────────────
    print(f"\n[2/4] Building geometry-proxy dataset ...")
    stats = load_deepjeb_fallback(
        csv_path, out_dir, n_points=n_pts,
        load_directions=dirs, seed=seed, verbose=False
    )
    total = stats["total_samples"]
    print(f"  {total} samples saved to {out_dir}/")

    # ── Step 3: Verify saved samples ────────────────────────────
    print(f"\n[3/4] Verifying samples ...")
    pc_files = sorted([f for f in os.listdir(out_dir) if f.endswith("_pc.npy")])
    print(f"  Files: {len(pc_files)} point clouds found")

    sample_errors = 0
    vms_all = []
    for fname in pc_files[:20]:   # spot-check first 20
        base = fname.replace("_pc.npy", "")
        pc   = np.load(os.path.join(out_dir, f"{base}_pc.npy"))
        y    = float(np.load(os.path.join(out_dir, f"{base}_y.npy")))
        meta = np.load(os.path.join(out_dir, f"{base}_meta.npy"),
                       allow_pickle=True).item()

        # Assertions
        assert pc.shape[1] == 3,                    f"Bad PC shape: {pc.shape}"
        assert pc.dtype == np.float32,              "PC must be float32"
        max_d = float(np.max(np.linalg.norm(pc, axis=1)))
        assert abs(max_d - 1.0) < 0.02,            f"Not unit sphere: {max_d:.4f}"
        assert y > 0,                               "Negative normalized stress"
        # Roundtrip
        vm_back = denormalize(y)
        assert vm_back > 0,                         "Roundtrip failed"
        assert "von_mises_mpa" in meta,             "Missing meta key"

        vms_all.append(meta["von_mises_mpa"])

    print(f"  Spot-check 20 samples: all assertions passed")
    print(f"  Stress range in spot-check: "
          f"{min(vms_all):.1f} – {max(vms_all):.1f} MPa")

    # ── Step 4: Dataset overview plot ───────────────────────────
    print(f"\n[4/4] Generating dataset overview plots ...")

    # Read all metadata
    all_meta = []
    for fname in pc_files:
        base      = fname.replace("_pc.npy", "")
        meta_path = os.path.join(out_dir, f"{base}_meta.npy")
        if os.path.exists(meta_path):
            all_meta.append(
                np.load(meta_path, allow_pickle=True).item()
            )

    os.makedirs("outputs", exist_ok=True)
    _plot_demo_overview(df, all_meta, "outputs/demo_overview.png")
    _plot_stress_heatmap(df, "outputs/demo_stress_heatmap.png")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n" + "="*60)
    print(f"  Demo Complete")
    print(f"="*60)
    print(f"  Samples prepared : {total}")
    print(f"  Output directory : {out_dir}/")
    print(f"  Plots generated  : outputs/demo_overview.png")
    print(f"                     outputs/demo_stress_heatmap.png")
    print(f"\n  Next steps:")
    print(f"  1. Place xyzdmlc.npz in data/raw/ (real point clouds)")
    print(f"  2. python scripts/prepare_dataset.py")
    print(f"  3. python scripts/train.py")
    print(f"  4. python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print(f"="*60 + "\n")


def _plot_demo_overview(df, all_meta, out_path):
    """4-panel dataset overview from real CSV data."""
    DIR_COLORS = {"ver":"#2E75B6","hor":"#E87722","dia":"#2EAA4F","tor":"#C00000"}
    DIR_LABELS = {"ver":"Vertical","hor":"Horizontal","dia":"Diagonal","tor":"Torsional"}
    DIR_ORDER  = ["ver","hor","dia","tor"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DeepJEB Dataset Overview — {len(df)} Jet Engine Brackets\n"
        f"Ti–6Al–4V  |  Nonlinear FEA  |  OptiStruct",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Stress distributions (violin)
    ax = axes[0, 0]
    cols = [f"max_{d}_stress(MPa)" for d in DIR_ORDER]
    data = [df[c].dropna().values for c in cols]
    parts = ax.violinplot(data, showmedians=True)
    for pc, d in zip(parts["bodies"], DIR_ORDER):
        pc.set_facecolor(DIR_COLORS[d]); pc.set_alpha(0.7)
    ax.axhline(YIELD_STRESS_MPA, color="red", ls="--", lw=1.5,
               label=f"Yield = {YIELD_STRESS_MPA} MPa")
    ax.set_xticks(range(1,5))
    ax.set_xticklabels([DIR_LABELS[d] for d in DIR_ORDER])
    ax.set_ylabel("Von Mises Stress (MPa)")
    ax.set_title("Stress distribution per load direction")
    ax.legend(fontsize=9)

    # Mass distribution
    ax = axes[0, 1]
    ax.hist(df["mass(kg)"].dropna(), bins=50, color="#2E75B6",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Bracket mass (kg)")
    ax.set_ylabel("Count")
    ax.set_title(f"Mass distribution  "
                 f"[{df['mass(kg)'].min():.2f} – {df['mass(kg)'].max():.2f} kg]")
    ax.axvline(df["mass(kg)"].mean(), color="red", ls="--",
               label=f"Mean={df['mass(kg)'].mean():.2f} kg")
    ax.legend()

    # Stress vs mass per direction
    ax = axes[1, 0]
    for d in DIR_ORDER:
        col = f"max_{d}_stress(MPa)"
        ax.scatter(df["mass(kg)"], df[col], c=DIR_COLORS[d],
                   label=DIR_LABELS[d], alpha=0.4, s=6, edgecolors="none")
    ax.axhline(YIELD_STRESS_MPA, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Mass (kg)"); ax.set_ylabel("Von Mises Stress (MPa)")
    ax.set_title("Stress vs mass"); ax.legend(fontsize=8)

    # Yielded fraction per direction
    ax = axes[1, 1]
    yielded = []
    totals  = []
    for d in DIR_ORDER:
        col = f"max_{d}_stress(MPa)"
        v   = df[col].dropna()
        yielded.append((v > YIELD_STRESS_MPA).sum())
        totals.append(len(v))

    x = range(len(DIR_ORDER))
    ax.bar([DIR_LABELS[d] for d in DIR_ORDER], yielded,
           color=[DIR_COLORS[d] for d in DIR_ORDER], edgecolor="white")
    for xi, (y_n, t) in enumerate(zip(yielded, totals)):
        ax.text(xi, y_n + 2, f"{y_n}\n({100*y_n/t:.1f}%)",
                ha="center", fontsize=8)
    ax.set_ylabel("Samples where FoS < 1")
    ax.set_title("Yielded samples per direction")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white"); plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_stress_heatmap(df, out_path):
    """Correlation heatmap between stress across load directions."""
    import matplotlib.cm as cm

    cols = {
        "Vertical":   "max_ver_stress(MPa)",
        "Horizontal": "max_hor_stress(MPa)",
        "Diagonal":   "max_dia_stress(MPa)",
        "Torsional":  "max_tor_stress(MPa)",
    }
    data  = df[[v for v in cols.values()]].dropna()
    corr  = data.corr()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stress Correlation Analysis — DeepJEB",
                 fontsize=12, fontweight="bold")

    # Heatmap
    ax = axes[0]
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    labels = list(cols.keys())
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if corr.values[i,j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Cross-direction stress correlation")

    # Scatter: vertical vs horizontal stress
    ax = axes[1]
    x = df["max_ver_stress(MPa)"].values
    y = df["max_hor_stress(MPa)"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[mask], y[mask], alpha=0.4, s=8, c="#2E75B6", edgecolors="none")
    ax.axvline(YIELD_STRESS_MPA, color="red", ls="--", lw=1, label="Yield")
    ax.axhline(YIELD_STRESS_MPA, color="red", ls="--", lw=1)
    ax.set_xlabel("Vertical load stress (MPa)")
    ax.set_ylabel("Horizontal load stress (MPa)")
    ax.set_title("Vertical vs horizontal stress")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white"); plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",       default="data/raw/bracket_labels.csv")
    p.add_argument("--out",       default="data/processed_demo")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--n_pts",     type=int, default=512)
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()
    run_demo(args.csv, args.out, args.n_pts, args.seed)
