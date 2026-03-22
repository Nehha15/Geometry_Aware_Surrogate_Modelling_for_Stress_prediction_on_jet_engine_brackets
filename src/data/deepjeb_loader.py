"""
deepjeb_loader.py
=================
Loads and prepares the Point-DeepONet / DeepJEB bracket dataset.

Dataset structure (place files in data/raw/):
  bracket_labels.csv  — per-bracket metadata and scalar FEA results
  xyzdmlc.npz         — point cloud xyz coordinates per bracket
  targets.npz         — full-field stress and displacement arrays

What this file does:
  1. Reads bracket_labels.csv → extracts Von Mises stress labels
     for each of the 4 load directions (ver, hor, dia, tor)
  2. Reads xyzdmlc.npz       → extracts point cloud for each bracket
  3. Subsamples each point cloud to n_points (FPS or random)
  4. Normalizes point clouds to unit sphere
  5. Applies log1p normalization to stress labels (MPa)
  6. Saves processed samples: {idx:05d}_{dir}_pc.npy
                              {idx:05d}_{dir}_y.npy
                              {idx:05d}_{dir}_meta.npy

The saved format is identical to the existing StressDataset API,
so all downstream code (dataset.py, trainer.py, evaluate.py)
works without modification.

Material: Ti-6Al-4V
  E = 113.8 GPa, nu = 0.342, yield = 227.6 MPa
  Nonlinear elastic-plastic with isotropic hardening
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):      # silent fallback
        return iterable


# Stress label column mapping: load direction → CSV column name
STRESS_COLUMNS = {
    "ver": "max_ver_stress(MPa)",
    "hor": "max_hor_stress(MPa)",
    "dia": "max_dia_stress(MPa)",
    "tor": "max_tor_stress(MPa)",
}

# Displacement label columns (also saved in metadata)
DISP_COLUMNS = {
    "ver": ("abs_max_ver_xdisp(mm)", "abs_max_ver_ydisp(mm)",
            "abs_max_ver_zdisp(mm)", "abs_max_ver_magdisp(mm)"),
    "hor": ("abs_max_hor_xdisp(mm)", "abs_max_hor_ydisp(mm)",
            "abs_max_hor_zdisp(mm)", "abs_max_hor_magdisp(mm)"),
    "dia": ("abs_max_dia_xdisp(mm)", "abs_max_dia_ydisp(mm)",
            "abs_max_dia_zdisp(mm)", "abs_max_dia_magdisp(mm)"),
    "tor": ("abs_max_tor_xdisp(mm)", "abs_max_tor_ydisp(mm)",
            "abs_max_tor_zdisp(mm)", "abs_max_tor_magdisp(mm)"),
}

LOAD_DIRECTION_IDX = {"ver": 0, "hor": 1, "dia": 2, "tor": 3}

YIELD_STRESS_MPA = 227.6   # Ti-6Al-4V


def normalize_pointcloud(pc: np.ndarray) -> np.ndarray:
    """Center + scale to unit sphere."""
    pc = pc.astype(np.float32)
    pc -= pc.mean(axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    if scale > 1e-8:
        pc /= scale
    return pc


def normalize_stress_mpa(stress_mpa: float) -> float:
    """log1p(MPa) — invertible, compresses wide dynamic range."""
    return float(np.log1p(max(stress_mpa, 0.0)))


def denormalize_stress(log_stress: float) -> float:
    """Invert normalize_stress_mpa → returns MPa."""
    return float(np.expm1(log_stress))


def fps_subsample(pc: np.ndarray, n_points: int,
                  rng: np.random.Generator = None) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) to subsample a large point cloud.

    FPS gives more uniform coverage than random sampling —
    important for complex bracket geometries where surface area
    varies a lot across different regions.

    Falls back to random sampling if the point cloud is smaller
    than n_points (pads by repeating random points).
    """
    N = pc.shape[0]
    if N <= n_points:
        # Pad by repeating random points
        _rng = rng or np.random.default_rng()
        idx  = _rng.integers(0, N, n_points - N)
        return np.vstack([pc, pc[idx]])

    _rng = rng or np.random.default_rng()

    selected = np.zeros(n_points, dtype=int)
    selected[0] = int(_rng.integers(0, N))
    dists = np.full(N, np.inf)

    for i in range(1, n_points):
        last  = pc[selected[i-1]]
        d     = np.sum((pc - last)**2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = int(np.argmax(dists))

    return pc[selected]


def random_subsample(pc: np.ndarray, n_points: int,
                     rng: np.random.Generator = None) -> np.ndarray:
    """Faster alternative to FPS — uniform random sampling."""
    _rng = rng or np.random.default_rng()
    N    = pc.shape[0]
    if N <= n_points:
        idx = _rng.integers(0, N, n_points - N)
        return np.vstack([pc, pc[idx]])
    idx = _rng.choice(N, n_points, replace=False)
    return pc[idx]


def load_deepjeb(
    labels_csv:     str,
    pointcloud_npz: str,
    out_dir:        str,
    n_points:       int  = 2048,
    load_directions = ("ver", "hor", "dia", "tor"),
    use_fps:        bool = False,
    seed:           int  = 42,
    verbose:        bool = True,
) -> dict:
    """
    Full DeepJEB data loading and preprocessing pipeline.

    Reads the CSV labels and the point cloud NPZ, creates one processed
    sample per (bracket, load_direction) pair, and saves to out_dir.

    Args:
        labels_csv:     path to bracket_labels.csv
        pointcloud_npz: path to xyzdmlc.npz
        out_dir:        directory to write processed samples
        n_points:       target point cloud size after subsampling
        load_directions: which load directions to include
        use_fps:        use Farthest Point Sampling (slower but better)
        seed:           random seed
        verbose:        print progress

    Returns:
        stats dict with dataset summary
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── 1. Load CSV labels ──────────────────────────────────────
    if verbose:
        print(f"\nLoading labels: {labels_csv}")
    df = pd.read_csv(labels_csv)
    df = df.dropna(subset=list(STRESS_COLUMNS.values()))
    n_brackets = len(df)
    if verbose:
        print(f"  {n_brackets} brackets loaded")
        for d in load_directions:
            col = STRESS_COLUMNS[d]
            print(f"  {d}: stress [{df[col].min():.1f}, "
                  f"{df[col].max():.1f}] MPa  mean={df[col].mean():.1f} MPa")

    # ── 2. Load point cloud NPZ ─────────────────────────────────
    if verbose:
        print(f"\nLoading point clouds: {pointcloud_npz}")

    pc_data = np.load(pointcloud_npz, allow_pickle=True)

    # xyzdmlc.npz typically stores arrays keyed by bracket item_name
    # Try common key formats
    pc_keys = list(pc_data.keys())
    if verbose:
        print(f"  NPZ keys (first 5): {pc_keys[:5]}")
        print(f"  Total keys: {len(pc_keys)}")

    # ── 3. Build sample index ───────────────────────────────────
    # Match CSV item_name to NPZ keys
    matched = []
    for _, row in df.iterrows():
        name = str(row["item_name"])
        # Try exact match, then common variants
        key = None
        for candidate in [name, name.replace("_", "-"),
                          name.lower(), name.upper()]:
            if candidate in pc_data:
                key = candidate
                break
        if key is not None:
            matched.append((row, key))
        elif verbose:
            pass  # silently skip unmatched

    if verbose:
        print(f"\n  Matched {len(matched)}/{n_brackets} brackets to point clouds")
        if len(matched) == 0:
            print("  WARNING: No matches found. Check that item_name values")
            print("  in the CSV match the keys in the NPZ file.")
            print(f"  CSV sample names:  {list(df['item_name'][:3])}")
            print(f"  NPZ sample keys:   {pc_keys[:3]}")

    # ── 4. Process and save ─────────────────────────────────────
    sample_idx    = 0
    skipped       = 0
    stats         = {d: {"count": 0, "vm_min": np.inf, "vm_max": 0.0,
                         "fos_below1": 0} for d in load_directions}

    subsample_fn = fps_subsample if use_fps else random_subsample
    sample_fn_name = "FPS" if use_fps else "random"

    if verbose:
        print(f"\nProcessing samples (subsampling={sample_fn_name}, "
              f"n_points={n_points}) ...")

    iterable = tqdm(matched) if verbose else matched

    for row, pc_key in iterable:
        try:
            raw_pc = pc_data[pc_key]  # shape: (N, 3) or (N, C)

            # Handle various NPZ array shapes
            if raw_pc.ndim == 1:
                raw_pc = raw_pc.reshape(-1, 3)
            if raw_pc.shape[1] > 3:
                raw_pc = raw_pc[:, :3]   # take xyz only
            if raw_pc.shape[1] < 3:
                skipped += 1
                continue

            # Subsample and normalize
            pc_sub  = subsample_fn(raw_pc.astype(np.float32), n_points, rng)
            pc_norm = normalize_pointcloud(pc_sub)

        except Exception as e:
            skipped += 1
            continue

        name = str(row["item_name"])

        for direction in load_directions:
            stress_col = STRESS_COLUMNS[direction]
            vm_mpa     = float(row[stress_col])

            if vm_mpa <= 0 or np.isnan(vm_mpa):
                skipped += 1
                continue

            # Normalized label
            y_norm = normalize_stress_mpa(vm_mpa)

            # Safety factor
            fos    = YIELD_STRESS_MPA / vm_mpa

            # Displacement metadata
            dx_col, dy_col, dz_col, dmag_col = DISP_COLUMNS[direction]
            meta = {
                "item_name":       name,
                "load_direction":  direction,
                "load_dir_idx":    LOAD_DIRECTION_IDX[direction],
                "von_mises_mpa":   vm_mpa,
                "safety_factor":   float(fos),
                "yielded":         bool(fos < 1.0),
                "disp_x_mm":       float(row.get(dx_col, 0.0)),
                "disp_y_mm":       float(row.get(dy_col, 0.0)),
                "disp_z_mm":       float(row.get(dz_col, 0.0)),
                "disp_mag_mm":     float(row.get(dmag_col, 0.0)),
                "mass_kg":         float(row.get("mass(kg)", 0.0)),
                "num_nodes":       int(row.get("num_nodes", 0)),
                "volume_mm3":      float(row.get("volume(mm3)", 0.0)),
                "freq_1st_hz":     float(row.get("1st_mode_freq(Hz)", 0.0)),
                "freq_2nd_hz":     float(row.get("2nd_mode_freq(Hz)", 0.0)),
            }

            # Save
            base = f"{sample_idx:05d}"
            np.save(os.path.join(out_dir, f"{base}_pc.npy"),   pc_norm)
            np.save(os.path.join(out_dir, f"{base}_y.npy"),
                    np.array(y_norm, dtype=np.float32))
            np.save(os.path.join(out_dir, f"{base}_meta.npy"), meta)

            # Update stats
            s = stats[direction]
            s["count"]      += 1
            s["vm_min"]      = min(s["vm_min"], vm_mpa)
            s["vm_max"]      = max(s["vm_max"], vm_mpa)
            if fos < 1.0:
                s["fos_below1"] += 1

            sample_idx += 1

    # ── 5. Summary ──────────────────────────────────────────────
    total = sample_idx
    if verbose:
        print(f"\n{'='*58}")
        print(f"  DeepJEB Dataset Preparation Complete")
        print(f"{'='*58}")
        print(f"  Total samples saved : {total}")
        print(f"  Skipped             : {skipped}")
        print(f"  Saved to            : {out_dir}")
        print(f"\n  Per load direction:")
        print(f"  {'Dir':<6}  {'n':>5}  {'Min MPa':>9}  "
              f"{'Max MPa':>9}  {'Yielded':>8}")
        print(f"  {'-'*50}")
        for d, s in stats.items():
            if s["count"] == 0:
                continue
            print(f"  {d:<6}  {s['count']:>5}  "
                  f"{s['vm_min']:>9.1f}  {s['vm_max']:>9.1f}  "
                  f"  {s['fos_below1']}/{s['count']}")
        print(f"{'='*58}\n")

    return {"total_samples": total, "skipped": skipped, "per_direction": stats}


def load_deepjeb_fallback(
    labels_csv: str,
    out_dir:    str,
    n_points:   int = 2048,
    load_directions=("ver", "hor", "dia", "tor"),
    seed:       int = 42,
    verbose:    bool = True,
) -> dict:
    """
    Fallback loader when xyzdmlc.npz is not available.

    Uses geometry descriptors from the CSV (volume, mass, inertia tensors,
    surface area, CG coordinates) to generate a synthetic feature vector.

    The point cloud is replaced with a structured set of feature points
    derived from the bracket's physical properties. This lets the rest
    of the pipeline run for testing purposes, but should not be used
    for final model training — the real xyzdmlc.npz is required for that.

    Each 'point cloud' is a (n_points, 3) array where coordinates are
    sampled from a Gaussian distribution shaped by the inertia tensor
    principal axes, approximating the bracket's 3D mass distribution.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\nFallback loader (no xyzdmlc.npz): generating geometry proxies")
        print(f"Loading labels: {labels_csv}")

    df = pd.read_csv(labels_csv)
    df = df.dropna(subset=list(STRESS_COLUMNS.values()))

    sample_idx = 0
    skipped    = 0
    stats      = {d: {"count": 0, "vm_min": np.inf, "vm_max": 0.0}
                  for d in load_directions}

    iterable = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()

    for _, row in iterable:
        # Build geometry proxy point cloud from inertia tensor
        # Principal moments → ellipsoid semi-axes
        I1 = max(float(row.get("I_1(kg*mm2)", 1000)), 1.0)
        I2 = max(float(row.get("I_2(kg*mm2)", 1000)), 1.0)
        I3 = max(float(row.get("I_3(kg*mm2)", 1000)), 1.0)

        # Ellipsoid semi-axes proportional to sqrt of inertia differences
        # (from I = 2/5 * m * r^2 → r ~ sqrt(I/m))
        mass = max(float(row.get("mass(kg)", 1.0)), 0.01)
        ax   = np.sqrt(max((I2 + I3 - I1) / (2 * mass), 0.01))
        ay   = np.sqrt(max((I1 + I3 - I2) / (2 * mass), 0.01))
        az   = np.sqrt(max((I1 + I2 - I3) / (2 * mass), 0.01))

        # Sample points on ellipsoid surface (Monte Carlo rejection)
        pts = rng.normal(0, 1, (n_points * 3, 3))
        pts[:, 0] *= ax
        pts[:, 1] *= ay
        pts[:, 2] *= az
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts   = pts / (norms + 1e-8)
        pts   = pts[:n_points]
        pts   = pts.astype(np.float32)
        pc_norm = normalize_pointcloud(pts)

        name = str(row["item_name"])

        for direction in load_directions:
            stress_col = STRESS_COLUMNS[direction]
            vm_mpa     = float(row[stress_col])
            if vm_mpa <= 0 or np.isnan(vm_mpa):
                skipped += 1
                continue

            y_norm = normalize_stress_mpa(vm_mpa)
            fos    = YIELD_STRESS_MPA / vm_mpa

            dx_col, dy_col, dz_col, dmag_col = DISP_COLUMNS[direction]
            meta = {
                "item_name":      name,
                "load_direction": direction,
                "load_dir_idx":   LOAD_DIRECTION_IDX[direction],
                "von_mises_mpa":  vm_mpa,
                "safety_factor":  float(fos),
                "yielded":        bool(fos < 1.0),
                "disp_x_mm":      float(row.get(dx_col, 0.0)),
                "disp_y_mm":      float(row.get(dy_col, 0.0)),
                "disp_z_mm":      float(row.get(dz_col, 0.0)),
                "disp_mag_mm":    float(row.get(dmag_col, 0.0)),
                "mass_kg":        float(row.get("mass(kg)", 0.0)),
                "num_nodes":      int(row.get("num_nodes", 0)),
                "volume_mm3":     float(row.get("volume(mm3)", 0.0)),
                "proxy_geometry": True,
            }

            base = f"{sample_idx:05d}"
            np.save(os.path.join(out_dir, f"{base}_pc.npy"),   pc_norm)
            np.save(os.path.join(out_dir, f"{base}_y.npy"),
                    np.array(y_norm, dtype=np.float32))
            np.save(os.path.join(out_dir, f"{base}_meta.npy"), meta)

            s = stats[direction]
            s["count"] += 1
            s["vm_min"]  = min(s["vm_min"], vm_mpa)
            s["vm_max"]  = max(s["vm_max"], vm_mpa)
            sample_idx  += 1

    if verbose:
        print(f"\n  Proxy dataset: {sample_idx} samples saved to {out_dir}")
        print(f"  NOTE: Use real xyzdmlc.npz for actual training.\n")

    return {"total_samples": sample_idx, "skipped": skipped,
            "per_direction": stats, "is_proxy": True}
