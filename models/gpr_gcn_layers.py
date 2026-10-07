"""GPR-style encoder: signed learnable hop coefficients (ICLR 2021 GPR-GNN spirit).

Unlike Scale1 (non-negative softmax hops), hop weights are free parameters so
high hops can receive negative weight on heterophilic graphs during pretrain.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers import GCN
from models.scale_gcn_layers import ScaleGcnLayers, _normalize_adj_dense, _sparse_to_dense


class SignedHopFusion(nn.Module):
    """Linear combination of normalized adjacency powers with signed coeffs."""

    def __init__(self, num_hops: int = 3):
        super().__init__()
        self.num_hops = max(1, int(num_hops))
        self.coeffs = nn.Parameter(torch.zeros(self.num_hops))
        with torch.no_grad():
            self.coeffs[0] = 1.0

    def fuse_dense(self, adj: torch.Tensor) -> torch.Tensor:
        s = _normalize_adj_dense(_sparse_to_dense(adj).float())
        n = s.size(0)
        fused = torch.zeros_like(s)
        power = torch.eye(n, device=s.device, dtype=s.dtype)
        for i in range(self.num_hops):
            if i > 0:
                power = power @ s
            fused = fused + self.coeffs[i] * power
        return fused


class GprGcnLayers(torch.nn.Module):
    """GCN stack on GPR signed multi-hop adjacency fusion."""

    def __init__(self, n_in, n_h, num_layers_num, dropout, num_hops: int = 3):
        super().__init__()
        self.num_layers_num = num_layers_num
        self.hop_fusion = SignedHopFusion(num_hops=num_hops)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=dropout)
        for i in range(num_layers_num):
            if i:
                self.convs.append(GCN(n_h, n_h))
            else:
                self.convs.append(GCN(n_in, n_h))
            self.bns.append(torch.nn.BatchNorm1d(n_h))

    def forward(self, seq, adj, sparse, LP=False):
        graph_output = torch.squeeze(seq, dim=0)
        adj_use = self.hop_fusion.fuse_dense(adj)
        for i in range(self.num_layers_num):
            if i:
                graph_output = ScaleGcnLayers._gcn_dense(self.convs[i], graph_output, adj_use) + graph_output
            else:
                graph_output = ScaleGcnLayers._gcn_dense(self.convs[i], graph_output, adj_use)
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
        return graph_output.unsqueeze(dim=0)
