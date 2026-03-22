"""
metrics.py
==========
Regression metrics for stress prediction in both log-space and MPa.
"""
import numpy as np


def denormalize_stress(log_v: float) -> float:
    """log1p(MPa) → MPa"""
    return float(np.expm1(log_v))


def rmse(p, t):      return float(np.sqrt(np.mean((p - t)**2)))
def mae(p, t):       return float(np.mean(np.abs(p - t)))
def max_err(p, t):   return float(np.max(np.abs(p - t)))

def r2_score(p, t):
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    return float(1 - ss_res / (ss_tot + 1e-10))

def mape(p, t, eps=1e-6):
    return float(100 * np.mean(np.abs((p - t) / (np.abs(t) + eps))))

def rel_err_dist(p, t, eps=1e-6):
    e = 100 * np.abs((p - t) / (np.abs(t) + eps))
    return {
        "p50": float(np.percentile(e, 50)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "within_5pct":  float(np.mean(e <  5)),
        "within_10pct": float(np.mean(e < 10)),
        "within_20pct": float(np.mean(e < 20)),
    }

def compute_all_metrics(pred_log: np.ndarray,
                        target_log: np.ndarray) -> dict:
    pred_mpa   = np.array([denormalize_stress(v) for v in pred_log])
    target_mpa = np.array([denormalize_stress(v) for v in target_log])

    return {
        "rmse_log":    rmse(pred_log, target_log),
        "mae_log":     mae(pred_log, target_log),
        "r2":          r2_score(pred_log, target_log),
        "rmse_mpa":    rmse(pred_mpa, target_mpa),
        "mae_mpa":     mae(pred_mpa, target_mpa),
        "mape_pct":    mape(pred_mpa, target_mpa),
        "max_err_mpa": max_err(pred_mpa, target_mpa),
        "rel_err":     rel_err_dist(pred_mpa, target_mpa),
    }
