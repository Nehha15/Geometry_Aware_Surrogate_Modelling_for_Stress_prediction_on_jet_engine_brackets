"""
losses.py
=========
Weighted physics-informed loss for DeepJEB stress regression.

FIX v2:
  - StressLoss.forward now accepts an optional `weights` tensor
    (B,) for per-sample weighting — outlier brackets get weight=0.3
  - Physics constraint still penalises negative predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressLoss(nn.Module):
    """
    Loss = weighted_mean(Huber(pred, target)) + physics_weight * penalty

    Physics constraint: stress >= 0 always (penalise negative predictions).

    Args:
        primary:        'huber' or 'mse'
        delta:          Huber delta parameter
        physics_weight: weight on the non-negativity penalty
    """

    def __init__(self, primary="huber", delta=1.0, physics_weight=0.05):
        super().__init__()
        self.physics_weight = physics_weight
        self.delta          = delta
        self.use_huber      = (primary == "huber")

    def forward(self, pred, target, weights=None):
        """
        Args:
            pred:    (B, 1)
            target:  (B, 1)
            weights: (B,) or None  — per-sample loss weights
        Returns:
            total_loss, info_dict
        """
        # Per-sample Huber or MSE
        if self.use_huber:
            per_sample = F.huber_loss(pred, target,
                                      delta=self.delta, reduction="none")
        else:
            per_sample = F.mse_loss(pred, target, reduction="none")

        per_sample = per_sample.squeeze(1)   # (B,)

        # Apply sample weights
        if weights is not None:
            per_sample = per_sample * weights

        loss_main = per_sample.mean()

        # Physics: no negative stress predictions
        loss_phys = torch.mean(F.relu(-pred) ** 2)

        total = loss_main + self.physics_weight * loss_phys

        return total, {
            "loss_primary": loss_main.item(),
            "loss_physics": loss_phys.item(),
            "loss_total":   total.item(),
        }