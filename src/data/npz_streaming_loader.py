"""
npz_streaming_loader.py
=======================
Streams xyzdmlc.npz key-by-key and saves fully preprocessed samples.

PREPROCESSING APPLIED (v3 — EDA-informed)
==========================================

Point cloud
-----------
  1. Random subsample → 2048 points
  2. Center: pc -= mean(pc)
  3. Scale:  pc /= max(||pc_i||)   unit sphere

Physics features  (13, down from raw 16)
-----------------------------------------
  Kept: mass_kg, volume_mm3, surface_area_mm2,
        I_1/I_2/I_3 (principal moments, r=-0.74 to -0.77 with stress),
        freq_1st_hz, freq_2nd_hz, cg_x/y/z, num_nodes, min_jac

  Dropped: I_xx, I_yy, I_zz, I_xy, I_yz, I_zx
    — redundant linear combos of I_1/I_2/I_3, confirmed by r>0.99

Target normalisation
--------------------
  y_raw  = log1p(vm_mpa)
  y      = (y_raw - 6.484) / 0.386   zero-mean unit-std in log space
  Un-normalise: vm_mpa = expm1(y * 0.386 + 6.484)

Outlier handling
----------------
  11 brackets with ver_stress > 2000 MPa → sample_weight = 0.3
  All others → sample_weight = 1.0
  Trainer applies weighted loss.

Saved per sample i:
  {i:05d}_pc.npy      (2048,3) normalised point cloud
  {i:05d}_y.npy       scalar   normalised target
  {i:05d}_dir.npy     (3,)     one-hot direction
  {i:05d}_weight.npy  scalar   sample weight
  {i:05d}_meta.npy    dict     all metadata + raw physics values
"""

import os
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(x, **kw): return x


STRESS_LOG_MEAN    = 6.484
STRESS_LOG_STD     = 0.386
OUTLIER_STRESS_MPa = 2000.0
OUTLIER_WEIGHT     = 0.3
NORMAL_WEIGHT      = 1.0

STRESS_COLS = {
    "ver": "max_ver_stress(MPa)",
    "hor": "max_hor_stress(MPa)",
    "dia": "max_dia_stress(MPa)",
}
DISP_COLS = {
    "ver": "abs_max_ver_magdisp(mm)",
    "hor": "abs_max_hor_magdisp(mm)",
    "dia": "abs_max_dia_magdisp(mm)",
}
DIR_MAP = {"ver": 0, "hor": 1, "dia": 2}

PHYS_FEAT_KEYS = [
    "mass_kg", "volume_mm3", "surface_area_mm2",
    "I_1_kg_mm2", "I_2_kg_mm2", "I_3_kg_mm2",
    "freq_1st_hz", "freq_2nd_hz",
    "cg_x_mm", "cg_y_mm", "cg_z_mm",
    "num_nodes", "min_jac",
]
N_PHYS_FEATURES = len(PHYS_FEAT_KEYS)   # 13


def normalize_pointcloud(pc):
    pc = pc.astype(np.float32)
    pc -= pc.mean(axis=0)
    d = np.max(np.linalg.norm(pc, axis=1))
    if d > 1e-8:
        pc /= d
    return pc


def random_subsample(pc, n, rng):
    N = len(pc)
    if N <= n:
        return np.vstack([pc, pc[rng.integers(0, N, n - N)]])
    return pc[rng.choice(N, n, replace=False)]


def process_pointcloud(raw_pc, n_points, rng):
    return normalize_pointcloud(
        random_subsample(raw_pc[:, :3], n_points, rng)
    ).astype(np.float32)


def normalise_target(vm_mpa):
    y_raw = float(np.log1p(vm_mpa))
    return float((y_raw - STRESS_LOG_MEAN) / STRESS_LOG_STD), y_raw


def denormalise_target(y):
    return float(np.expm1(y * STRESS_LOG_STD + STRESS_LOG_MEAN))


def extract_phys(row):
    return np.array([
        float(row.get("mass(kg)",          0.0)),
        float(row.get("volume(mm3)",        0.0)),
        float(row.get("surface_area(mm2)",  0.0)),
        float(row.get("I_1(kg*mm2)",        0.0)),
        float(row.get("I_2(kg*mm2)",        0.0)),
        float(row.get("I_3(kg*mm2)",        0.0)),
        float(row.get("1st_mode_freq(Hz)",  0.0)),
        float(row.get("2nd_mode_freq(Hz)",  0.0)),
        float(row.get("CG_x(mm)",           0.0)),
        float(row.get("CG_y(mm)",           0.0)),
        float(row.get("CG_z(mm)",           0.0)),
        float(row.get("num_nodes",          0.0)),
        float(row.get("min_jac",            0.0)),
    ], dtype=np.float32)


def stream_process_npz(labels_csv, npz_path, out_dir,
                       n_points=2048, seed=42, max_samples=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("\n" + "="*60)
    print("  DEEPJEB PREPROCESSING  v3  (EDA-informed)")
    print("="*60)
    print(f"  Target : y = (log1p(MPa) - {STRESS_LOG_MEAN}) / {STRESS_LOG_STD}")
    print(f"  Feats  : {N_PHYS_FEATURES} physics features")
    print(f"  Outlier: > {OUTLIER_STRESS_MPa} MPa → weight={OUTLIER_WEIGHT}")

    df = pd.read_csv(labels_csv)
    df["item_name"] = df["item_name"].astype(str)
    df_map = {r["item_name"]: r for _, r in df.iterrows()}

    outlier_names = {
        str(r["item_name"]) for _, r in df.iterrows()
        if float(r.get("max_ver_stress(MPa)", 0)) > OUTLIER_STRESS_MPa
    }
    print(f"\n  CSV rows       : {len(df)}")
    print(f"  Outlier brackets: {len(outlier_names)} → {sorted(outlier_names)}")

    npz  = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    keys = list(npz.keys())
    print(f"  NPZ keys       : {len(keys)}\n")

    sample_idx = 0
    skipped    = 0
    dir_counts = {d: 0 for d in DIR_MAP}

    for pc_key in (_tqdm(keys, desc="Processing") if verbose else keys):
        parts = pc_key.split("_")
        if len(parts) < 3:
            skipped += 1; continue

        direction = parts[0]
        item_id   = "_".join(parts[1:])

        if direction not in DIR_MAP or item_id not in df_map:
            skipped += 1; continue

        row = df_map[item_id]

        try:
            vm_mpa = float(row[STRESS_COLS[direction]])
        except (KeyError, ValueError):
            skipped += 1; continue
        if vm_mpa <= 0 or np.isnan(vm_mpa):
            skipped += 1; continue

        try:
            raw_pc = np.array(npz[pc_key])
        except Exception:
            skipped += 1; continue
        if raw_pc.ndim == 1:
            raw_pc = raw_pc.reshape(-1, 3)
        if raw_pc.shape[0] < 10 or raw_pc.shape[1] < 3:
            skipped += 1; continue

        pc = process_pointcloud(raw_pc, n_points, rng)
        del raw_pc

        y_norm, y_raw = normalise_target(vm_mpa)

        dir_vec = np.zeros(3, dtype=np.float32)
        dir_vec[DIR_MAP[direction]] = 1.0

        is_outlier = item_id in outlier_names
        weight     = OUTLIER_WEIGHT if is_outlier else NORMAL_WEIGHT

        disp_mm = float(row.get(DISP_COLS.get(direction, ""), 0.0)) \
                  if DISP_COLS.get(direction, "") in (row.index if hasattr(row, 'index') else {}) else 0.0

        meta = {
            "item_name":       item_id,
            "npz_key":         pc_key,
            "direction":       direction,
            "von_mises_mpa":   vm_mpa,
            "y_raw_log1p":     y_raw,
            "disp_mm":         disp_mm,
            "is_outlier":      is_outlier,
            "sample_weight":   weight,
            # 13 physics features (raw)
            "mass_kg":          float(row.get("mass(kg)",          0.0)),
            "volume_mm3":       float(row.get("volume(mm3)",        0.0)),
            "surface_area_mm2": float(row.get("surface_area(mm2)",  0.0)),
            "I_1_kg_mm2":       float(row.get("I_1(kg*mm2)",        0.0)),
            "I_2_kg_mm2":       float(row.get("I_2(kg*mm2)",        0.0)),
            "I_3_kg_mm2":       float(row.get("I_3(kg*mm2)",        0.0)),
            "freq_1st_hz":      float(row.get("1st_mode_freq(Hz)",  0.0)),
            "freq_2nd_hz":      float(row.get("2nd_mode_freq(Hz)",  0.0)),
            "cg_x_mm":          float(row.get("CG_x(mm)",           0.0)),
            "cg_y_mm":          float(row.get("CG_y(mm)",           0.0)),
            "cg_z_mm":          float(row.get("CG_z(mm)",           0.0)),
            "num_nodes":        int(row.get("num_nodes",            0)),
            "min_jac":          float(row.get("min_jac",            0.0)),
        }

        base = f"{sample_idx:05d}"
        np.save(os.path.join(out_dir, f"{base}_pc.npy"),     pc)
        np.save(os.path.join(out_dir, f"{base}_y.npy"),      np.float32(y_norm))
        np.save(os.path.join(out_dir, f"{base}_dir.npy"),    dir_vec)
        np.save(os.path.join(out_dir, f"{base}_weight.npy"), np.float32(weight))
        np.save(os.path.join(out_dir, f"{base}_meta.npy"),   meta)

        dir_counts[direction] += 1
        sample_idx += 1
        if max_samples and sample_idx >= max_samples:
            break

    print("\n" + "="*60)
    print(f"  Saved   : {sample_idx}  Skipped: {skipped}")
    for d, n in dir_counts.items():
        print(f"    {d}: {n}")
    print("="*60 + "\n")
    return sample_idx


def verify_processed_sample(out_dir, idx=0):
    base = f"{idx:05d}"
    pc   = np.load(os.path.join(out_dir, f"{base}_pc.npy"))
    y    = float(np.load(os.path.join(out_dir, f"{base}_y.npy")))
    dv   = np.load(os.path.join(out_dir, f"{base}_dir.npy"))
    wt   = float(np.load(os.path.join(out_dir, f"{base}_weight.npy")))
    meta = np.load(os.path.join(out_dir, f"{base}_meta.npy"), allow_pickle=True).item()

    vm  = meta["von_mises_mpa"]
    y_expected = (np.log1p(vm) - STRESS_LOG_MEAN) / STRESS_LOG_STD

    print(f"\nSample {idx:05d}")
    print(f"  direction      : {meta['direction']}")
    print(f"  von_mises_mpa  : {vm:.2f}")
    print(f"  y (normalised) : {y:.4f}  (expected {y_expected:.4f})")
    print(f"  dir one-hot    : {dv}")
    print(f"  weight         : {wt}  (outlier={meta['is_outlier']})")
    print(f"  pc shape       : {pc.shape}")
    print(f"  centroid ~0    : {pc.mean(axis=0)}")
    print(f"  max_dist ~1    : {np.max(np.linalg.norm(pc, axis=1)):.6f}")
    print(f"\n  Physics features:")
    for k in PHYS_FEAT_KEYS:
        v = meta.get(k, "MISSING")
        flag = " *** MISSING ***" if v == "MISSING" or v == 0.0 else ""
        print(f"    {k:<22}: {v}{flag}")

    assert pc.shape == (2048, 3)
    assert abs(y - y_expected) < 1e-3, f"Target mismatch: {y} vs {y_expected}"
    assert dv.sum() == 1.0
    assert meta.get("I_2_kg_mm2", 0) > 0, "I_2 missing"
    print("\n  [OK] all checks passed")
    return True