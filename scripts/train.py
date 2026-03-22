"""
train.py — v3
n_phys_features=13, 6-element collate, early_stopping_patience=999 by default.
"""

import os, sys, argparse, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DeepJEBDataset, make_splits, collate_fn
from src.models.pointnet_plus import PointNetPlusPlus
from src.training.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]; mc = cfg["model"]; tc = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading dataset from {dc['processed_dir']} ...")
    full_ds = DeepJEBDataset(dc["processed_dir"], augment=False,
                             n_points=dc["n_points"], seed=dc["seed"])
    print(f"Total samples: {len(full_ds)}")

    train_ds, val_ds, test_ds = make_splits(
        full_ds, dc["train_split"], dc["val_split"], dc["seed"])
    print(f"Split: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"],
                              shuffle=True, num_workers=2,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=tc["batch_size"]*2,
                              shuffle=False, num_workers=2,
                              collate_fn=collate_fn)

    n_dir  = mc.get("n_load_directions", 3)
    n_phys = mc.get("n_phys_features",  13)

    model = PointNetPlusPlus(
        use_tnet          = mc.get("use_tnet", True),
        dropout           = mc.get("dropout",  0.4),
        n_load_directions = n_dir,
        n_phys_features   = n_phys,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: PointNet++  |  {n_params:,} parameters")
    print(f"       regressor input: 1024 + {n_dir} dir + {n_phys} phys = {1024+n_dir+n_phys}")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from: {args.resume}")

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.train(tc["epochs"])

    idx_path = os.path.join(tc["checkpoint_dir"], "test_indices.txt")
    with open(idx_path, "w") as f:
        for i in test_ds.indices:
            f.write(f"{i}\n")
    print(f"Test indices saved: {idx_path}")


if __name__ == "__main__":
    main()