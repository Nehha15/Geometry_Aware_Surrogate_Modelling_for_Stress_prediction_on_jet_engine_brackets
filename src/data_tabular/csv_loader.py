"""
csv_loader.py
=============
Loads DeepJEB bracket_labels.csv and builds the tabular feature matrix.

No xyzdmlc.npz required — all 19 features come directly from the CSV.

Input features (19 total):
  Geometry:    num_nodes, volume, mass, surface_area
  Inertia:     I_1, I_2, I_3 (principal), I_xx, I_yy, I_zz, I_xy, I_yz, I_zx
  CG:          CG_x, CG_y, CG_z
  Mesh:        min_jac
  Dynamics:    1st_mode_freq, 2nd_mode_freq
  Load:        one-hot direction (4 dims, added at runtime)

Target (1 per sample):
  log1p(Von Mises stress in MPa)   -- invertible via expm1

Why these features are physically meaningful:
  - Inertia tensor encodes the 3D mass distribution shape of the bracket.
    Two brackets with identical mass but different inertia tensors have
    fundamentally different geometries and different stress distributions.
  - Modal frequencies directly encode structural stiffness-to-mass ratio.
    f1 = (1/2pi)*sqrt(k/m). High f1 => stiff bracket => higher stress
    concentrations under the same applied load.
  - CG position encodes the lever arm from the mounting bolts to the
    effective load application point — longer lever arm => higher bending
    moment => higher stress.
  - Volume and surface_area jointly encode the geometric complexity ratio
    (surface_area / volume^(2/3)) which correlates with geometric
    irregularity and stress concentration potential.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

YIELD_MPa = 227.6   # Ti-6Al-4V

FEATURE_COLS = [
    "num_nodes",
    "volume(mm3)",
    "mass(kg)",
    "surface_area(mm2)",
    "CG_x(mm)",
    "CG_y(mm)",
    "CG_z(mm)",
    "I_xx(kg*mm2)",
    "I_yy(kg*mm2)",
    "I_zz(kg*mm2)",
    "I_xy(kg*mm2)",
    "I_yz(kg*mm2)",
    "I_zx(kg*mm2)",
    "I_1(kg*mm2)",
    "I_2(kg*mm2)",
    "I_3(kg*mm2)",
    "min_jac",
    "1st_mode_freq(Hz)",
    "2nd_mode_freq(Hz)",
]

STRESS_COLS = {
    "ver": "max_ver_stress(MPa)",
    "hor": "max_hor_stress(MPa)",
    "dia": "max_dia_stress(MPa)",
    "tor": "max_tor_stress(MPa)",
}

DISP_COLS = {
    "ver": "abs_max_ver_magdisp(mm)",
    "hor": "abs_max_hor_magdisp(mm)",
    "dia": "abs_max_dia_magdisp(mm)",
    "tor": "abs_max_tor_magdisp(mm)",
}

LOAD_DIR_IDX = {"ver": 0, "hor": 1, "dia": 2, "tor": 3}


def load_csv_dataset(
    csv_path: str,
    load_directions=("ver", "hor", "dia", "tor"),
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
):
    """
    Load bracket_labels.csv and build train/val/test arrays.

    Returns
    -------
    splits : dict with keys 'train', 'val', 'test'
        Each value is a dict:
            X       — (N, 23) float32  [19 geo features + 4 one-hot load dir]
            y       — (N,)    float32  log1p(stress_MPa)
            vm_mpa  — (N,)    float32  raw stress in MPa
            fos     — (N,)    float32  safety factor
            dirs    — list[str] load direction label per sample
            names   — list[str] bracket item_name per sample
    scaler : fitted StandardScaler on training X (geo features only, not one-hot)
    feature_names : list[str] of all 23 input feature names
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLS + list(STRESS_COLS.values()))

    # Build flat sample list: one row per (bracket, direction)
    rows = []
    for _, bracket in df.iterrows():
        geo = bracket[FEATURE_COLS].values.astype(np.float32)  # (19,)
        for direction in load_directions:
            vm_mpa = float(bracket[STRESS_COLS[direction]])
            if vm_mpa <= 0 or np.isnan(vm_mpa):
                continue
            # One-hot load direction
            lc_oh = np.zeros(4, dtype=np.float32)
            lc_oh[LOAD_DIR_IDX[direction]] = 1.0

            rows.append({
                "geo":       geo,
                "lc_oh":     lc_oh,
                "lc_idx":    LOAD_DIR_IDX[direction],
                "direction": direction,
                "vm_mpa":    vm_mpa,
                "y":         float(np.log1p(vm_mpa)),
                "fos":       YIELD_MPa / vm_mpa,
                "disp_mm":   float(bracket.get(DISP_COLS[direction], 0.0)),
                "name":      str(bracket["item_name"]),
                "mass_kg":   float(bracket["mass(kg)"]),
                "freq1_hz":  float(bracket["1st_mode_freq(Hz)"]),
                "volume":    float(bracket["volume(mm3)"]),
                "nodes":     int(bracket["num_nodes"]),
            })

    n = len(rows)
    print(f"  Loaded {n} samples from {len(df)} brackets x {len(load_directions)} directions")

    # Stratified split by load direction
    all_idx  = list(range(n))
    dir_labels = [r["direction"] for r in rows]
    test_frac  = 1.0 - train_frac - val_frac

    tv_idx, test_idx = train_test_split(
        all_idx, test_size=test_frac, stratify=dir_labels, random_state=seed
    )
    tv_dirs   = [dir_labels[i] for i in tv_idx]
    val_size  = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        tv_idx, test_size=val_size, stratify=tv_dirs, random_state=seed
    )

    # Build arrays
    def _pack(indices):
        geo   = np.stack([rows[i]["geo"]   for i in indices])     # (N, 19)
        lc_oh = np.stack([rows[i]["lc_oh"] for i in indices])     # (N, 4)
        X     = np.concatenate([geo, lc_oh], axis=1)              # (N, 23)
        y     = np.array([rows[i]["y"]     for i in indices], dtype=np.float32)
        vm    = np.array([rows[i]["vm_mpa"] for i in indices], dtype=np.float32)
        fos   = np.array([rows[i]["fos"]   for i in indices], dtype=np.float32)
        dirs  = [rows[i]["direction"]       for i in indices]
        names = [rows[i]["name"]            for i in indices]
        meta  = [{
            "vm_mpa":  rows[i]["vm_mpa"],
            "fos":     rows[i]["fos"],
            "dir":     rows[i]["direction"],
            "name":    rows[i]["name"],
            "mass_kg": rows[i]["mass_kg"],
            "freq1_hz":rows[i]["freq1_hz"],
            "volume":  rows[i]["volume"],
            "nodes":   rows[i]["nodes"],
            "disp_mm": rows[i]["disp_mm"],
        } for i in indices]
        return {"X": X, "y": y, "vm_mpa": vm, "fos": fos,
                "dirs": dirs, "names": names, "meta": meta}

    train_data = _pack(train_idx)
    val_data   = _pack(val_idx)
    test_data  = _pack(test_idx)

    # Fit StandardScaler on training geo features only (not the one-hot)
    scaler = StandardScaler()
    scaler.fit(train_data["X"][:, :19])   # fit on geo features

    def _scale(data):
        X = data["X"].copy()
        X[:, :19] = scaler.transform(X[:, :19]).astype(np.float32)
        data["X_scaled"] = X
        return data

    train_data = _scale(train_data)
    val_data   = _scale(val_data)
    test_data  = _scale(test_data)

    splits = {"train": train_data, "val": val_data, "test": test_data}

    feature_names = FEATURE_COLS + ["lc_ver", "lc_hor", "lc_dia", "lc_tor"]

    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    return splits, scaler, feature_names
