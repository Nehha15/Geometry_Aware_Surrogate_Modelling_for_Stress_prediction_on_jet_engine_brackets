"""
predict.py
==========
Step 4: Predict Von Mises stress for a new bracket geometry.

Accepts:
  --pc    path to a .npy point cloud (N,3)
  --npz   path to an .npz file with a named bracket (--key required)

The load direction must be specified — stress depends on how the
bracket is loaded, not just its shape.

Usage:
    python scripts/predict.py \
        --pc   data/processed/00001_pc.npy \
        --dir  ver

    python scripts/predict.py \
        --npz  data/raw/xyzdmlc.npz \
        --key  10_30 \
        --dir  hor \
        --checkpoint checkpoints/best_model.pth
"""

import os, sys, argparse, yaml, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.data.deepjeb_loader import (
    normalize_pointcloud, denormalize_stress,
    LOAD_DIRECTION_IDX, YIELD_STRESS_MPA, random_subsample
)
from src.models.pointnet_plus import PointNetPlusPlus


LOAD_DIRECTIONS = ["ver", "hor", "dia", "tor"]
DIR_FULL = {"ver": "Vertical", "hor": "Horizontal",
            "dia": "Diagonal", "tor": "Torsional"}


def predict_one(model, pc_norm, load_dir, device, cfg):
    """Run inference on one normalized (N,3) point cloud."""
    mc     = cfg["model"]
    n_pts  = cfg["data"]["n_points"]

    # Subsample if needed
    if pc_norm.shape[0] != n_pts:
        pc_norm = random_subsample(pc_norm, n_pts)
    pc_norm = normalize_pointcloud(pc_norm)

    pc_t  = torch.tensor(pc_norm, dtype=torch.float32).unsqueeze(0).to(device)
    lc_t  = torch.tensor([LOAD_DIRECTION_IDX[load_dir]],
                          dtype=torch.long).to(device)

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_log, _ = model(pc_t, lc_t)
    ms = (time.perf_counter() - t0) * 1000

    vm_mpa = denormalize_stress(pred_log.item())
    fos    = YIELD_STRESS_MPA / max(vm_mpa, 1e-6)

    return {
        "load_direction":   load_dir,
        "von_mises_mpa":    vm_mpa,
        "von_mises_pa":     vm_mpa * 1e6,
        "safety_factor":    fos,
        "yielded":          fos < 1.0,
        "inference_ms":     ms,
    }


def main():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pc",  help="Path to (N,3) .npy point cloud")
    grp.add_argument("--npz", help="Path to xyzdmlc.npz")

    p.add_argument("--key",        default=None,
                   help="Bracket key in NPZ (required with --npz)")
    p.add_argument("--dir",        required=True,
                   choices=LOAD_DIRECTIONS,
                   help="Load direction: ver | hor | dia | tor")
    p.add_argument("--all_dirs",   action="store_true",
                   help="Predict all 4 load directions")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--config",     default="configs/config.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc     = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────
    model = PointNetPlusPlus(
        use_tnet          = mc.get("use_tnet", True),
        dropout           = mc.get("dropout",  0.4),
        n_load_directions = mc.get("n_load_directions", 4),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Model loaded: {args.checkpoint}")

    # ── Load geometry ───────────────────────────────────────────
    if args.pc:
        raw_pc = np.load(args.pc).astype(np.float32)
        if raw_pc.ndim == 1:
            raw_pc = raw_pc.reshape(-1, 3)
        pc_norm = normalize_pointcloud(raw_pc)
        print(f"Point cloud: {args.pc}  shape={raw_pc.shape}")
    else:
        if not args.key:
            print("Error: --key required when using --npz")
            sys.exit(1)
        npz_data = np.load(args.npz, allow_pickle=True)
        raw_pc   = npz_data[args.key].astype(np.float32)
        if raw_pc.shape[1] > 3:
            raw_pc = raw_pc[:, :3]
        pc_norm = normalize_pointcloud(raw_pc)
        print(f"Bracket: {args.key}  shape={raw_pc.shape}")

    # ── Predict ─────────────────────────────────────────────────
    dirs = LOAD_DIRECTIONS if args.all_dirs else [args.dir]

    print(f"\n{'='*52}")
    print(f"  Structural Stress Prediction  —  DeepJEB Surrogate")
    print(f"  Material: Ti–6Al–4V  |  yield = {YIELD_STRESS_MPA} MPa")
    print(f"{'='*52}")
    print(f"  {'Direction':<14}  {'Von Mises':>12}  {'FoS':>7}  {'Status'}")
    print(f"  {'-'*48}")

    for d in dirs:
        res = predict_one(model, pc_norm.copy(), d, device, cfg)
        status = "YIELDED" if res["yielded"] else "Safe"
        print(f"  {DIR_FULL[d]:<14}  "
              f"{res['von_mises_mpa']:>10.2f} MPa  "
              f"{res['safety_factor']:>7.3f}  {status}")

    print(f"\n  Inference time: {res['inference_ms']:.1f} ms")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
