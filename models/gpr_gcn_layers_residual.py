"""GPR residual encoder: base GCN + gamma * (GPR branch - base).

Preserves vanilla GCN path for heterophilic transfer while allowing signed
multi-hop correction for homophilic domains (UniProp pretrain).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.gcnlayers import GcnLayers
from models.gpr_gcn_layers import GprGcnLayers


class GprResidualGcnLayers(torch.nn.Module):
    """h = h_base + gamma * (h_gpr - h_base); gamma fixed at construction."""

    def __init__(
        self,
        n_in,
        n_h,
        num_layers_num,
        dropout,
        num_hops: int = 3,
        scale_residual_gamma: float = 0.2,
    ):
        super().__init__()
        self.base = GcnLayers(n_in, n_h, num_layers_num, dropout)
        self.gpr = GprGcnLayers(n_in, n_h, num_layers_num, dropout, num_hops=num_hops)
        gamma = float(max(0.0, min(scale_residual_gamma, 1.0)))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, seq, adj, sparse, LP=False):
        h_base = self.base(seq, adj, sparse, LP)
        if self.gamma.item() <= 0.0:
            return h_base
        h_gpr = self.gpr(seq, adj, sparse, LP)
        if self.gamma.item() >= 1.0:
            return h_gpr
        return h_base + self.gamma * (h_gpr - h_base)
