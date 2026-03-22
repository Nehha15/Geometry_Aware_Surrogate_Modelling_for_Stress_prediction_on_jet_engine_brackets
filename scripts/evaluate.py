"""
evaluate.py — v3
Reads n_phys_features=13 from config, unpacks 6-element batch.
"""

import os, sys, json, argparse, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DeepJEBDataset, make_splits, collate_fn
from src.models.pointnet_plus import PointNetPlusPlus
from src.evaluation.validate import validate
from src.evaluation.visualize import VisualizationSuite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--split",      default="test", choices=["val","test"])
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]; mc = cfg["model"]; tc = cfg["training"]; ec = cfg["evaluation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointNetPlusPlus(
        use_tnet          = mc.get("use_tnet",          True),
        dropout           = mc.get("dropout",            0.4),
        n_load_directions = mc.get("n_load_directions",    3),
        n_phys_features   = mc.get("n_phys_features",     13),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded: {args.checkpoint}  "
          f"(epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss',0):.4f})")

    full_ds = DeepJEBDataset(dc["processed_dir"], augment=False,
                             n_points=dc["n_points"], seed=dc["seed"])
    _, val_ds, test_ds = make_splits(
        full_ds, dc["train_split"], dc["val_split"], dc["seed"])
    eval_ds = test_ds if args.split == "test" else val_ds
    print(f"Evaluating on {args.split} set: {len(eval_ds)} samples")

    loader = DataLoader(eval_ds, batch_size=tc["batch_size"]*2,
                        shuffle=False, num_workers=2, collate_fn=collate_fn)

    report = validate(model, loader, device, split_name=args.split)

    meta_list = []
    for idx in eval_ds.indices:
        base = full_ds.indices[idx]
        mp   = os.path.join(dc["processed_dir"], f"{base}_meta.npy")
        if os.path.exists(mp):
            meta_list.append(np.load(mp, allow_pickle=True).item())

    hist_path = os.path.join(tc.get("log_dir","logs"), "history.json")
    history   = {"train_loss":[], "val_loss":[], "lr":[]}
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

    plots_dir = ec.get("plots_dir", "outputs")
    viz       = VisualizationSuite(plots_dir=plots_dir)
    print("\nGenerating visualizations ...")
    viz.plot_training_curves(history)
    viz.plot_predicted_vs_actual(report)
    viz.plot_error_distribution(report)
    viz.plot_failure_analysis(report, ec.get("failure_threshold_pct", 10.0))
    viz.plot_safety_factor(report, meta_list)
    viz.plot_stress_vs_geometry(report, meta_list)

    all_meta = []
    for base in full_ds.indices:
        mp = os.path.join(dc["processed_dir"], f"{base}_meta.npy")
        if os.path.exists(mp):
            all_meta.append(np.load(mp, allow_pickle=True).item())
    if all_meta:
        viz.plot_dataset_overview(all_meta)

    os.makedirs(ec.get("report_dir","data/results"), exist_ok=True)
    rpt = os.path.join(ec["report_dir"], "eval_report.json")
    with open(rpt, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "split": args.split,
                   "n_samples": report["n"], "metrics": report["metrics"],
                   "per_dir": report["per_dir"]}, f, indent=2)
    print(f"\nReport: {rpt}")
    print(f"Plots:  {plots_dir}/")


if __name__ == "__main__":
    main()