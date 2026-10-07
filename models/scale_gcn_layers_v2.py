"""Scale2 encoder: per-layer multi-hop adjacency fusion (ScaleGNN-style).

Differences from Scale1 (`scale_gcn_layers.py`):
- Separate learnable hop weights for each GCN layer (not one global fusion).
- Dense path: K-hop powers computed once; each layer applies its own weights on
  features (same math as fused-adj @ x, without materializing 3 fused N×N mats).
- Sparse path: same idea with cached sparse powers (memory-friendly on large graphs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GCN


def _sparse_to_dense(adj: torch.Tensor) -> torch.Tensor:
    if adj.is_sparse:
        return adj.to_dense()
    return adj.squeeze(0) if adj.dim() == 3 else adj


def _normalize_adj_dense(adj_dense: torch.Tensor) -> torch.Tensor:
    deg = adj_dense.sum(dim=1).clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt.unsqueeze(1) * adj_dense * inv_sqrt.unsqueeze(0)


def _as_sparse_adj(adj: torch.Tensor) -> torch.Tensor:
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    adj = adj.float()
    if adj.is_sparse:
        return adj.coalesce()
    return adj.to_sparse().coalesce()


def _normalize_adj_sparse(adj: torch.Tensor) -> torch.Tensor:
    adj = _as_sparse_adj(adj)
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    row, col = adj.indices()
    values = adj.values() * inv_sqrt[row] * inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), values, adj.size()).coalesce()


def _adj_powers_dense(norm_adj: torch.Tensor, num_hops: int):
    powers = [norm_adj]
    for _ in range(1, num_hops):
        powers.append(powers[-1] @ norm_adj)
    return powers


def _adj_powers_sparse(norm_adj: torch.Tensor, num_hops: int):
    powers = [norm_adj]
    for _ in range(1, num_hops):
        powers.append(torch.sparse.mm(powers[-1], norm_adj).coalesce())
    return powers


def _use_dense_path(adj: torch.Tensor) -> bool:
    """GSL refined adj is dense with grad; spmm backward needs strided adj."""
    if not adj.is_sparse:
        return True
    return bool(getattr(adj, "requires_grad", False) and adj.requires_grad)


class PerLayerHopFusion(nn.Module):
    """Layer-specific softmax weights over normalized k-hop adjacency powers."""

    def __init__(self, num_layers: int, num_hops: int):
        super().__init__()
        self.num_hops = max(1, int(num_hops))
        self.logits = nn.Parameter(torch.zeros(num_layers, self.num_hops))

    def layer_weights(self, layer_idx: int) -> torch.Tensor:
        return F.softmax(self.logits[layer_idx], dim=0)


class ScaleGcnLayersV2(torch.nn.Module):
    """Drop-in GcnLayers replacement with per-layer Scale2 fusion."""

    def __init__(self, n_in, n_h, num_layers_num, dropout, num_hops: int = 3):
        super().__init__()
        self.num_layers_num = num_layers_num
        self.num_hops = max(1, int(num_hops))
        self.hop_fusion = PerLayerHopFusion(num_layers_num, self.num_hops)
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
    def _aggregate_dense(powers, weights: torch.Tensor, seq_fts: torch.Tensor) -> torch.Tensor:
        out = torch.mm(powers[0], seq_fts) * weights[0]
        for k in range(1, len(powers)):
            out = out + weights[k] * torch.mm(powers[k], seq_fts)
        return out

    @staticmethod
    def _aggregate_sparse(powers, weights: torch.Tensor, seq_fts: torch.Tensor) -> torch.Tensor:
        out = torch.spmm(powers[0], seq_fts) * weights[0]
        for k in range(1, len(powers)):
            out = out + weights[k] * torch.spmm(powers[k], seq_fts)
        return out

    def _layer_conv(
        self,
        conv: GCN,
        x: torch.Tensor,
        powers,
        weights: torch.Tensor,
        dense: bool,
    ) -> torch.Tensor:
        seq_fts = conv.fc(x)
        if dense:
            out = self._aggregate_dense(powers, weights, seq_fts)
        else:
            out = self._aggregate_sparse(powers, weights, seq_fts)
        if conv.bias is not None:
            out = out + conv.bias
        return conv.act(out)

    def forward(self, seq, adj, sparse, LP=False):
        graph_output = torch.squeeze(seq, dim=0)
        dense_path = _use_dense_path(adj)
        if dense_path:
            norm = _normalize_adj_dense(_sparse_to_dense(adj).float())
            powers = _adj_powers_dense(norm, self.num_hops)
        else:
            norm = _normalize_adj_sparse(adj)
            powers = _adj_powers_sparse(norm, self.num_hops)

        for i in range(self.num_layers_num):
            w = self.hop_fusion.layer_weights(i)
            if i:
                graph_output = self._layer_conv(self.convs[i], graph_output, powers, w, dense_path) + graph_output
            else:
                graph_output = self._layer_conv(self.convs[i], graph_output, powers, w, dense_path)
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
        return graph_output.unsqueeze(dim=0)
