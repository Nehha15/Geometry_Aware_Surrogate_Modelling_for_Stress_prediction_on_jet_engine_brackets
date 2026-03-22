"""
dataset.py
==========
PyTorch Dataset — reads preprocessed samples from npz_streaming_loader v3.

Key changes from v2:
  - 13 physics features (was 4)
  - Feature keys match PHYS_FEAT_KEYS in npz_streaming_loader.py
  - Loads sample weight from {base}_weight.npy
  - Collate fn passes weight to trainer for weighted loss
  - Target is already normalised — no further transform needed
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from collections import Counter

LOAD_DIR_IDX    = {"ver": 0, "hor": 1, "dia": 2}
N_DIRECTIONS    = 3
N_PHYS_FEATURES = 13   # must match npz_streaming_loader.PHYS_FEAT_KEYS

PHYS_FEAT_KEYS = [
    "mass_kg", "volume_mm3", "surface_area_mm2",
    "I_1_kg_mm2", "I_2_kg_mm2", "I_3_kg_mm2",
    "freq_1st_hz", "freq_2nd_hz",
    "cg_x_mm", "cg_y_mm", "cg_z_mm",
    "num_nodes", "min_jac",
]


class DeepJEBDataset(Dataset):
    """
    Reads .npy files written by stream_process_npz.

    __getitem__ returns:
      pc      (2048, 3)  normalised point cloud
      dir_vec (3,)       one-hot direction
      feat    (13,)      StandardScaler-normalised physics features
      y       (1,)       normalised log-stress target
      weight  scalar     sample weight (1.0 normal, 0.3 outlier)
      meta    dict
    """

    def __init__(self, data_dir, augment=False, n_points=2048, seed=42):
        self.data_dir = data_dir
        self.augment  = augment
        self.n_points = n_points
        self.rng      = np.random.default_rng(seed)

        pc_files = sorted(f for f in os.listdir(data_dir) if f.endswith("_pc.npy"))
        self.indices = [f.replace("_pc.npy", "") for f in pc_files]
        if not self.indices:
            raise RuntimeError(f"No *_pc.npy found in '{data_dir}'. Run prepare_dataset.py first.")

        print(f"[DeepJEBDataset] {len(self.indices)} samples in '{data_dir}'")
        self._compute_feature_stats()

    def _compute_feature_stats(self):
        feats, missing = [], 0
        for base in self.indices:
            p = os.path.join(self.data_dir, f"{base}_meta.npy")
            if not os.path.exists(p):
                missing += 1; continue
            m = np.load(p, allow_pickle=True).item()
            feats.append([float(m.get(k, 0.0)) for k in PHYS_FEAT_KEYS])

        if missing:
            print(f"[DeepJEBDataset] WARNING: {missing} samples missing meta")

        if not feats:
            self.feat_mean = np.zeros(N_PHYS_FEATURES, dtype=np.float32)
            self.feat_std  = np.ones(N_PHYS_FEATURES,  dtype=np.float32)
            return

        arr = np.array(feats, dtype=np.float32)
        self.feat_mean = arr.mean(axis=0)
        self.feat_std  = arr.std(axis=0) + 1e-6

        for i, k in enumerate(PHYS_FEAT_KEYS):
            if self.feat_std[i] < 1e-4:
                print(f"[DeepJEBDataset] WARNING: '{k}' std~0 — may be missing from meta")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base = self.indices[idx]
        d    = self.data_dir

        pc  = np.load(os.path.join(d, f"{base}_pc.npy"))
        y   = float(np.load(os.path.join(d, f"{base}_y.npy")))
        dv  = np.load(os.path.join(d, f"{base}_dir.npy"))

        wt_path = os.path.join(d, f"{base}_weight.npy")
        wt = float(np.load(wt_path)) if os.path.exists(wt_path) else 1.0

        mp = os.path.join(d, f"{base}_meta.npy")
        meta = np.load(mp, allow_pickle=True).item() if os.path.exists(mp) else {}

        # Rebuild one-hot from direction string (safe for any dir vector size)
        direction = meta.get("direction", "ver")
        dir_vec   = np.zeros(N_DIRECTIONS, dtype=np.float32)
        dir_vec[LOAD_DIR_IDX.get(direction, 0)] = 1.0

        # Normalise physics features
        raw_feat = np.array([float(meta.get(k, 0.0)) for k in PHYS_FEAT_KEYS],
                            dtype=np.float32)
        feat = (raw_feat - self.feat_mean) / self.feat_std

        if self.augment:
            pc = _augment(pc, self.rng)

        return (
            torch.tensor(pc,      dtype=torch.float32),  # (2048,3)
            torch.tensor(dir_vec, dtype=torch.float32),  # (3,)
            torch.tensor(feat,    dtype=torch.float32),  # (13,)
            torch.tensor([y],     dtype=torch.float32),  # (1,)
            torch.tensor(wt,      dtype=torch.float32),  # scalar
            meta,
        )

    def get_load_directions(self):
        labels = []
        for base in self.indices:
            p = os.path.join(self.data_dir, f"{base}_meta.npy")
            if os.path.exists(p):
                m = np.load(p, allow_pickle=True).item()
                labels.append(LOAD_DIR_IDX.get(m.get("direction", "ver"), 0))
            else:
                labels.append(0)
        return labels


def _augment(pc, rng):
    pc  = pc.copy().astype(np.float32)
    pc += np.clip(rng.normal(0, 0.005, pc.shape).astype(np.float32), -0.02, 0.02)
    th  = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(th), np.sin(th)
    R   = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    pc  = (pc @ R.T) * rng.uniform(0.95, 1.05)
    return pc


def collate_fn(batch):
    pcs, dirs, feats, ys, wts, metas = zip(*batch)
    return (
        torch.stack(pcs),
        torch.stack(dirs),
        torch.stack(feats),
        torch.stack(ys),
        torch.stack(wts),
        list(metas),
    )


def make_splits(dataset, train_frac=0.70, val_frac=0.15, seed=42):
    labels    = dataset.get_load_directions()
    all_idx   = list(range(len(dataset)))
    test_frac = 1.0 - train_frac - val_frac

    tv_idx, test_idx = train_test_split(
        all_idx, test_size=test_frac, stratify=labels, random_state=seed)
    tv_labels = [labels[i] for i in tv_idx]
    val_size  = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        tv_idx, test_size=val_size, stratify=tv_labels, random_state=seed)

    aug_ds   = DeepJEBDataset(dataset.data_dir, augment=True,
                              n_points=dataset.n_points, seed=seed)
    train_ds = Subset(aug_ds,   train_idx)
    val_ds   = Subset(dataset,  val_idx)
    test_ds  = Subset(dataset,  test_idx)

    dir_names = {v: k for k, v in LOAD_DIR_IDX.items()}
    print(f"\n[make_splits] {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        c = Counter(labels[i] for i in idxs)
        print(f"  {name}: " + "  ".join(f"{dir_names.get(k,'?')}={v}"
                                         for k, v in sorted(c.items())))
    return train_ds, val_ds, test_ds