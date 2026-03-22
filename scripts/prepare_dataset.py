"""
prepare_dataset.py
==================
Step 1: Process raw DeepJEB data into training-ready .npy samples.

THREE MODES — choose based on what files you have:

  Mode A (RECOMMENDED): Real point clouds, local NPZ
  --------------------------------------------------
  You have xyzdmlc.npz on disk (any size — streaming reads it).
    python scripts/prepare_dataset.py --npz data/raw/xyzdmlc.npz

  Mode B: Real point clouds, download from HuggingFace
  -----------------------------------------------------
  You don't have the NPZ locally — stream it from HuggingFace.
    python scripts/prepare_dataset.py --huggingface

  Mode C: Fallback — geometry proxy (no point cloud file needed)
  --------------------------------------------------------------
  For pipeline testing only. Uses inertia tensor as geometry proxy.
  Results will be less accurate than A or B.
    python scripts/prepare_dataset.py --fallback

WHAT THIS PRODUCES (in data/processed/):
  00000_pc.npy    (2048, 3) float32  point cloud, unit sphere normalized
  00000_y.npy     scalar float32     log1p(Von Mises MPa)
  00000_meta.npy  dict               item_name, direction, stress, FoS, ...
  ... (one triplet per bracket per load direction = up to 8552 samples)

POINT CLOUD PROCESSING STEPS:
  1. Load raw (N, 3) xyz coords from NPZ  [N = 120k to 380k nodes]
  2. Subsample to 2048 points (FPS or random)
  3. Subtract centroid  -> zero mean
  4. Divide by max dist -> unit sphere (max dist = 1.0)
  5. Save as float32 .npy

STRESS LABEL STEPS:
  1. Read max_ver/hor/dia/tor_stress(MPa) from bracket_labels.csv
  2. Apply log1p transform: y = log1p(stress_MPa)
  3. Save as float32 .npy  (invert with expm1 at inference time)
"""

import os, sys, argparse, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    p = argparse.ArgumentParser(
        description="Prepare DeepJEB point cloud dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--csv",         default=None,
                   help="Path to bracket_labels.csv")
    p.add_argument("--npz",         default=None,
                   help="Path to xyzdmlc.npz (local, streaming)")
    p.add_argument("--out",         default=None,
                   help="Output directory for processed samples")
    p.add_argument("--n_pts",       type=int, default=None,
                   help="Points per bracket after subsampling (default 2048)")
    p.add_argument("--fps",         action="store_true",
                   help="Use Farthest Point Sampling (better, but slower)")
    p.add_argument("--huggingface", action="store_true",
                   help="Stream from HuggingFace (no local NPZ needed)")
    p.add_argument("--hf_repo",     default="Jonjey/DeepJEB",
                   help="HuggingFace dataset repo ID")
    p.add_argument("--fallback",    action="store_true",
                   help="Use geometry proxy fallback (no NPZ at all)")
    p.add_argument("--max",         type=int, default=None,
                   help="Process only first N brackets (for testing)")
    p.add_argument("--verify",      action="store_true",
                   help="Verify first sample after processing")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc   = cfg["data"]
    csv  = args.csv  or dc["labels_csv"]
    npz  = args.npz  or dc.get("pointcloud_npz", "data/raw/xyzdmlc.npz")
    out  = args.out  or dc["processed_dir"]
    npts = args.n_pts or dc["n_points"]
    dirs = tuple(dc["load_directions"])

    # ── Mode B: HuggingFace streaming ───────────────────────────
    if args.huggingface:
        from src.data.npz_streaming_loader import stream_from_huggingface
        stream_from_huggingface(
            labels_csv      = csv,
            out_dir         = out,
            hf_repo         = args.hf_repo,
            hf_filename     = "xyzdmlc.npz",
            n_points        = npts,
            load_directions = dirs,
            use_fps         = args.fps,
            seed            = args.seed,
        )

    # ── Mode C: Fallback proxy ───────────────────────────────────
    elif args.fallback or not os.path.exists(npz):
        if not args.fallback:
            print(f"\n[WARNING] {npz} not found.")
            print("  Use --npz to specify the path, or --huggingface to stream.")
            print("  Falling back to geometry proxy (less accurate).\n")
        from src.data.deepjeb_loader import load_deepjeb_fallback
        load_deepjeb_fallback(
            labels_csv      = csv,
            out_dir         = out,
            n_points        = npts,
            load_directions = dirs,
            seed            = args.seed,
            verbose         = True,
        )

    # ── Mode A: Local NPZ, streaming ────────────────────────────
    else:
        from src.data.npz_streaming_loader import stream_process_npz
    stream_process_npz(
    labels_csv = csv,
    npz_path   = npz,
    out_dir    = out,
    n_points   = npts,
    seed       = args.seed,
    max_samples= args.max,   # ✅ fixed
    verbose    = True,
)
    # ── Optional: verify first sample ───────────────────────────
    if args.verify:
        from src.data.npz_streaming_loader import verify_processed_sample
        verify_processed_sample(out, idx=0)


if __name__ == "__main__":
    main()
