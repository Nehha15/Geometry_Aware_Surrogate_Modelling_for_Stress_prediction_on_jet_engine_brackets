"""
validate.py — v3
Unpacks 6-element batch, moves all tensors to device, passes feats.
Also un-normalises predictions before computing MPa metrics.
"""

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

from src.evaluation.metrics import compute_all_metrics
from src.data.npz_streaming_loader import STRESS_LOG_MEAN, STRESS_LOG_STD


def _unnorm(y_arr):
    """Normalised log-stress → MPa."""
    return np.expm1(y_arr * STRESS_LOG_STD + STRESS_LOG_MEAN)


def validate(model, loader, device, split_name="val"):
    model.eval()
    preds, targets, dirs = [], [], []

    with torch.no_grad():
        for pcs, lcs, feats, ys, wts, metas in tqdm(
                loader, desc=f"[{split_name}]", leave=False):
            pcs   = pcs.to(device)
            lcs   = lcs.to(device)
            feats = feats.to(device)
            ys    = ys.to(device)

            pred, _ = model(pcs, lcs, feats)
            preds.append(pred.cpu().numpy())
            targets.append(ys.cpu().numpy())
            dirs.extend([m.get("direction", "unknown") for m in metas])

    pred_norm   = np.concatenate(preds).flatten()
    target_norm = np.concatenate(targets).flatten()
    dirs        = np.array(dirs)

    # Un-normalise for MPa metrics
    pred_log   = pred_norm   * STRESS_LOG_STD + STRESS_LOG_MEAN
    target_log = target_norm * STRESS_LOG_STD + STRESS_LOG_MEAN

    metrics = compute_all_metrics(pred_log, target_log)

    per_dir = {}
    for d in sorted(set(dirs)):
        mask = dirs == d
        if mask.sum() > 0:
            per_dir[d] = compute_all_metrics(pred_log[mask], target_log[mask])

    report = {
        "split":    split_name,
        "n":        len(pred_log),
        "metrics":  metrics,
        "per_dir":  per_dir,
        "pred_log": pred_log.tolist(),
        "true_log": target_log.tolist(),
        "dirs":     dirs.tolist(),
    }
    _print(report)
    return report


def _print(r):
    m  = r["metrics"]
    rd = m["rel_err"]
    print(f"\n{'='*60}")
    print(f"  {r['split'].upper()} set  ({r['n']} samples)")
    print(f"{'='*60}")
    print(f"  R²            : {m['r2']:.4f}")
    print(f"  RMSE (log)    : {m['rmse_log']:.4f}")
    print(f"  RMSE (MPa)    : {m['rmse_mpa']:.2f}")
    print(f"  MAE  (MPa)    : {m['mae_mpa']:.2f}")
    print(f"  MAPE          : {m['mape_pct']:.2f}%")
    print(f"  Max error     : {m['max_err_mpa']:.2f} MPa")
    print(f"\n  Accuracy:")
    print(f"    Within  5%  : {rd['within_5pct']*100:.1f}%")
    print(f"    Within 10%  : {rd['within_10pct']*100:.1f}%")
    if r["per_dir"]:
        print(f"\n  Per load direction:")
        for d, dm in r["per_dir"].items():
            print(f"    {d:<5} | MAPE: {dm['mape_pct']:.2f}% | R²: {dm['r2']:.3f}")
    print(f"{'='*60}\n")