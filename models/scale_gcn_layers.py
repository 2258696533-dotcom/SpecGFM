"""Scale1 encoder (legacy name: ScaleGNN base variant).

Single global learnable softmax over normalized k-hop adjacency powers; dense
fusion once before all GCN layers. Use ``--scale_encoder scale1`` or legacy
``--use_scale_gnn``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GCN


def _normalize_adj_dense(adj_dense: torch.Tensor) -> torch.Tensor:
    deg = adj_dense.sum(dim=1).clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt.unsqueeze(1) * adj_dense * inv_sqrt.unsqueeze(0)


def _sparse_to_dense(adj: torch.Tensor) -> torch.Tensor:
    if adj.is_sparse:
        return adj.to_dense()
    return adj.squeeze(0) if adj.dim() == 3 else adj


class LearnableHopFusion(nn.Module):
    """Softmax-weighted sum of normalized adjacency powers up to K hops."""

    def __init__(self, num_hops: int = 3):
        super().__init__()
        self.num_hops = max(1, int(num_hops))
        self.logits = nn.Parameter(torch.zeros(self.num_hops))

    def fuse_dense(self, adj: torch.Tensor) -> torch.Tensor:
        s = _normalize_adj_dense(_sparse_to_dense(adj).float())
        w = F.softmax(self.logits, dim=0)
        fused = torch.zeros_like(s)
        power = s
        for i in range(self.num_hops):
            if i > 0:
                power = power @ s
            fused = fused + w[i] * power
        return fused


class ScaleGcnLayers(torch.nn.Module):
    """Drop-in replacement for GcnLayers with fused multi-hop adjacency."""

    def __init__(self, n_in, n_h, num_layers_num, dropout, num_hops: int = 3):
        super().__init__()
        self.num_layers_num = num_layers_num
        self.hop_fusion = LearnableHopFusion(num_hops=num_hops)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=dropout)
        for i in range(num_layers_num):
            if i:
                self.convs.append(GCN(n_h, n_h))
            else:
                self.convs.append(GCN(n_in, n_h))
            self.bns.append(torch.nn.BatchNorm1d(n_h))

    @staticmethod
    def _gcn_dense(conv: GCN, x: torch.Tensor, adj_dense: torch.Tensor) -> torch.Tensor:
        seq_fts = conv.fc(x)
        out = torch.mm(adj_dense, seq_fts)
        if conv.bias is not None:
            out = out + conv.bias
        return conv.act(out)

    def forward(self, seq, adj, sparse, LP=False):
        graph_output = torch.squeeze(seq, dim=0)
        adj_use = self.hop_fusion.fuse_dense(adj)
        for i in range(self.num_layers_num):
            if i:
                graph_output = self._gcn_dense(self.convs[i], graph_output, adj_use) + graph_output
            else:
                graph_output = self._gcn_dense(self.convs[i], graph_output, adj_use)
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
        return graph_output.unsqueeze(dim=0)
