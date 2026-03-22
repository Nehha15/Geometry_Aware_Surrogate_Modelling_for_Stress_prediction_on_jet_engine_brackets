"""
stress_mlp.py
=============
Deep MLP for Von Mises stress prediction from tabular bracket features.

Input:  23-dimensional vector
          19 geometry/structural features (StandardScaler normalized)
           4 one-hot load direction flags
Output: scalar log1p(Von Mises MPa)

Architecture:
  Input(23) -> FC(256)+BN+SiLU -> FC(256)+BN+SiLU+Drop(0.3)
            -> FC(128)+BN+SiLU -> FC(128)+BN+SiLU+Drop(0.2)
            -> FC(64)+BN+SiLU  -> FC(1)

  Residual connections between same-width blocks stabilize training.
  SiLU (Swish) activation outperforms ReLU on tabular regression tasks.
  BatchNorm on all hidden layers for training stability.

Why SiLU instead of ReLU?
  SiLU(x) = x * sigmoid(x) is smooth everywhere, has non-zero gradient
  for negative inputs (unlike ReLU's dead neuron problem), and
  empirically converges faster on regression tasks with continuous targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressMLP(nn.Module):
    """
    Deep MLP surrogate for bracket Von Mises stress prediction.

    Args:
        n_features:  total input dimension (default 23: 19 geo + 4 one-hot)
        hidden_dims: list of hidden layer widths
        dropout:     dropout rate in deeper layers
    """

    def __init__(
        self,
        n_features:  int  = 23,
        hidden_dims       = (256, 256, 128, 128, 64),
        dropout:     float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = n_features

        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.SiLU())
            # Dropout on layers 2 and 4 only (not input, not output-adjacent)
            if i in (1, 3):
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for all linear layers — suited to SiLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 23) input features (geo StandardScaler-scaled + one-hot)
        Returns:
            (B, 1) predicted log1p(Von Mises MPa)
        """
        return self.net(x)


class ResidualStressMLP(nn.Module):
    """
    Stress MLP with residual (skip) connections.

    Residual connections allow gradients to flow directly through the
    network, enabling deeper architectures without vanishing gradients.
    Used when n_layers > 4.
    """

    def __init__(
        self,
        n_features: int   = 23,
        width:      int   = 256,
        n_blocks:   int   = 4,
        dropout:    float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, width)

        self.blocks = nn.ModuleList([
            _ResBlock(width, dropout if i > 0 else 0.0)
            for i in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.drop(x + self.net(x)))
