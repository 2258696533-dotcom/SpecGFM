"""Message Tuning (MTG) for frozen GNN encoders.

Per-layer learnable message prototypes fused before each GCN layer (Eq. 17):
  H_M = H + Softmax(H W_p) @ M
Reference: Message Tuning Outshines Graph Prompt Tuning (PS-Theory / MTG).
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scale_gcn_layers import ScaleGcnLayers


class MessageTuningFusion(nn.Module):
    """Single-layer message prototype fusion."""

    def __init__(self, in_dim: int, num_prototypes: int):
        super().__init__()
        self.num_prototypes = max(1, int(num_prototypes))
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, in_dim))
        self.proj = nn.Linear(in_dim, self.num_prototypes, bias=False)
        nn.init.xavier_uniform_(self.prototypes)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn = F.softmax(self.proj(h), dim=-1)
        return h + attn @ self.prototypes


def _layer_input_dims(base: nn.Module) -> List[int]:
    convs = getattr(base, "convs", None)
    if convs is None:
        raise TypeError(f"MTG requires encoder with .convs, got {type(base).__name__}")
    return [int(conv.fc.in_features) for conv in convs]


class MTGWrappedEncoder(nn.Module):
    """Wrap a frozen GcnLayers / ScaleGcnLayers / ScaleGcnLayersV2 with per-layer MTG."""

    _mtg_wrapped = True

    def __init__(self, base: nn.Module, num_prototypes: int = 4):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dims = _layer_input_dims(base)
        self.fusions = nn.ModuleList(
            MessageTuningFusion(d, num_prototypes) for d in dims
        )
        self.num_layers_num = int(getattr(base, "num_layers_num", len(dims)))

    def mtg_parameters(self) -> Iterable[nn.Parameter]:
        return self.fusions.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, seq, adj, sparse, LP=False):
        name = type(self.base).__name__
        if isinstance(self.base, ScaleGcnLayers):
            return self._forward_scale1(seq, adj, LP)
        if name == "ScaleGcnLayersV2":
            return self._forward_scale2(seq, adj, sparse, LP)
        return self._forward_gcn_layers(seq, adj, LP)

    def _forward_gcn_layers(self, seq, adj, LP=False):
        graph_output = torch.squeeze(seq, dim=0)
        device = graph_output.device
        for i in range(self.num_layers_num):
            graph_output = self.fusions[i](graph_output)
            self.fusions[i].to(device)
            inp = (graph_output, adj)
            if i:
                graph_output = self.base.convs[i](inp) + graph_output
            else:
                graph_output = self.base.convs[i](inp)
            if LP:
                graph_output = self.base.bns[i](graph_output)
                graph_output = self.base.dropout(graph_output)
        return graph_output.unsqueeze(dim=0)

    def _forward_scale1(self, seq, adj, LP=False):
        graph_output = torch.squeeze(seq, dim=0)
        adj_use = self.base.hop_fusion.fuse_dense(adj)
        for i in range(self.num_layers_num):
            graph_output = self.fusions[i](graph_output)
            if i:
                graph_output = (
                    ScaleGcnLayers._gcn_dense(self.base.convs[i], graph_output, adj_use)
                    + graph_output
                )
            else:
                graph_output = ScaleGcnLayers._gcn_dense(
                    self.base.convs[i], graph_output, adj_use
                )
            if LP:
                graph_output = self.base.bns[i](graph_output)
                graph_output = self.base.dropout(graph_output)
        return graph_output.unsqueeze(dim=0)

    def _forward_scale2(self, seq, adj, sparse, LP=False):
        from models.scale_gcn_layers_v2 import (
            _adj_powers_dense,
            _adj_powers_sparse,
            _normalize_adj_dense,
            _normalize_adj_sparse,
            _sparse_to_dense,
            _use_dense_path,
        )

        graph_output = torch.squeeze(seq, dim=0)
        dense_path = _use_dense_path(adj)
        if dense_path:
            norm = _normalize_adj_dense(_sparse_to_dense(adj).float())
            powers = _adj_powers_dense(norm, self.base.num_hops)
        else:
            norm = _normalize_adj_sparse(adj)
            powers = _adj_powers_sparse(norm, self.base.num_hops)

        for i in range(self.num_layers_num):
            graph_output = self.fusions[i](graph_output)
            w = self.base.hop_fusion.layer_weights(i)
            if i:
                graph_output = (
                    self.base._layer_conv(
                        self.base.convs[i], graph_output, powers, w, dense_path
                    )
                    + graph_output
                )
            else:
                graph_output = self.base._layer_conv(
                    self.base.convs[i], graph_output, powers, w, dense_path
                )
            if LP:
                graph_output = self.base.bns[i](graph_output)
                graph_output = self.base.dropout(graph_output)
        return graph_output.unsqueeze(dim=0)


def wrap_gcn_with_mtg(gcn: nn.Module, num_prototypes: int = 4) -> nn.Module:
    """Return MTG-wrapped encoder unless already wrapped."""
    if getattr(gcn, "_mtg_wrapped", False):
        return gcn
    return MTGWrappedEncoder(gcn, num_prototypes=num_prototypes)


def mtg_trainable_params(gcn: nn.Module) -> List[nn.Parameter]:
    if hasattr(gcn, "mtg_parameters"):
        return list(gcn.mtg_parameters())
    return []
