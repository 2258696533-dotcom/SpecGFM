"""Scale residual encoder: frozen-style base GCN + scaled Scale delta.

h = h_base + gamma * (h_scale - h_base)

At gamma=0 the module equals ``GcnLayers``; at gamma=1 it equals ``ScaleGcnLayers``
( separate weights ). ``gamma`` is fixed at construction (CLI ``--scale_residual_gamma``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.gcnlayers import GcnLayers
from models.scale_gcn_layers import ScaleGcnLayers


class ScaleResidualGcnLayers(torch.nn.Module):
    """Base GCN path plus learnable Scale branch mixed by constant gamma."""

    def __init__(
        self,
        n_in,
        n_h,
        num_layers_num,
        dropout,
        num_hops: int = 3,
        scale_residual_gamma: float = 0.15,
    ):
        super().__init__()
        self.base = GcnLayers(n_in, n_h, num_layers_num, dropout)
        self.scale = ScaleGcnLayers(n_in, n_h, num_layers_num, dropout, num_hops=num_hops)
        gamma = float(max(0.0, min(scale_residual_gamma, 1.0)))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, seq, adj, sparse, LP=False):
        h_base = self.base(seq, adj, sparse, LP)
        if self.gamma.item() <= 0.0:
            return h_base
        h_scale = self.scale(seq, adj, sparse, LP)
        if self.gamma.item() >= 1.0:
            return h_scale
        return h_base + self.gamma * (h_scale - h_base)
