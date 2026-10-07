"""Structural coordinate alignment for the two BandGSL adjacencies.

Learnable geometric bases map each band adjacency onto one simplex.
The paper uses this loss together with BandGSL (--scgw_p4). It is not ablated alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _adj_to_dense(adj: torch.Tensor) -> torch.Tensor:
    if adj.is_sparse:
        return adj.to_dense()
    return adj


def _structural_metric(adj: torch.Tensor) -> torch.Tensor:
    """Degree-normalized 2-hop dissimilarity matrix (mm-space proxy)."""
    a = _adj_to_dense(adj).float()
    a = a / (a.max() + 1e-8)
    a2 = torch.mm(a, a)
    d = 1.0 - a2 / (a2.max() + 1e-8)
    d.fill_diagonal_(0.0)
    return d


def _pool_metric(d: torch.Tensor, m: int) -> torch.Tensor:
    n = d.shape[0]
    if n == m:
        return d
    if n < m:
        return F.pad(d, (0, m - n, 0, m - n))
    idx = torch.linspace(0, n - 1, m, device=d.device).long()
    return d.index_select(0, idx).index_select(1, idx)


class GeometricBasesSCGW(nn.Module):
    def __init__(
        self,
        num_bases: int = 8,
        base_size: int = 16,
        feat_dim: int = 50,
        tau: float = 1.0,
    ):
        super().__init__()
        self.num_bases = int(num_bases)
        self.base_size = int(base_size)
        self.feat_dim = int(feat_dim)
        self.tau = float(max(tau, 1e-6))
        init = torch.rand(self.num_bases, self.base_size, self.base_size) * 0.4
        init = 0.5 * (init + init.transpose(-1, -2))
        for k in range(self.num_bases):
            init[k].fill_diagonal_(0.0)
        self.bases = nn.Parameter(init)
        self.slot_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.domain_map = nn.Linear(self.num_bases, 5, bias=False)

    def discrepancies(self, adj: torch.Tensor) -> torch.Tensor:
        d = _structural_metric(adj)
        d_m = _pool_metric(d, self.base_size)
        deltas = []
        for k in range(self.num_bases):
            b = self.bases[k]
            b = 0.5 * (b + b.t())
            b = b - torch.diag(torch.diag(b))
            deltas.append(((d_m - b) ** 2).mean())
        return torch.stack(deltas)

    def structural_coords(self, adj: torch.Tensor) -> torch.Tensor:
        return F.softmax(-self.discrepancies(adj) / self.tau, dim=0)

    def reconstructed_metric(self, w: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.base_size, self.base_size, device=w.device, dtype=w.dtype)
        for k in range(self.num_bases):
            b = self.bases[k]
            b = 0.5 * (b + b.t())
            b = b - torch.diag(torch.diag(b))
            out = out + w[k] * b
        return out

    def _slot_features(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = features.float()
        if x.is_sparse:
            x = x.to_dense()
        a = _adj_to_dense(adj).float()
        n = x.shape[0]
        deg = a.sum(dim=1)
        rank = deg.argsort().argsort().float() / max(n - 1, 1)
        slots = (rank * (self.base_size - 1)).long().clamp(0, self.base_size - 1)
        h = torch.zeros(self.base_size, x.shape[1], device=x.device, dtype=x.dtype)
        cnt = torch.zeros(self.base_size, device=x.device, dtype=x.dtype)
        h.index_add_(0, slots, x)
        cnt.index_add_(0, slots, torch.ones(n, device=x.device, dtype=x.dtype))
        return h / cnt.clamp(min=1.0).unsqueeze(1)

    def project_node_features(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """OT-inspired slot pooling mapped back to per-node feat_dim."""
        w = self.structural_coords(adj)
        x = features.float()
        if x.is_sparse:
            x = x.to_dense()
        h = self._slot_features(x, adj)
        mixed = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        for k in range(self.num_bases):
            slot_w = F.softmax(self.bases[k].mean(dim=1), dim=0)
            mixed = mixed + w[k] * (slot_w.unsqueeze(1) * h).sum(dim=0)
        bias = self.slot_proj(mixed).unsqueeze(0).expand(x.shape[0], -1)
        return x + bias

    def domain_prompt_bias(self, adj: torch.Tensor) -> torch.Tensor:
        w = self.structural_coords(adj)
        return self.domain_map(w.unsqueeze(0)).squeeze(0)

    def pretrain_loss(self, adj_list) -> torch.Tensor:
        loss = adj_list[0].new_tensor(0.0)
        for adj in adj_list:
            w = self.structural_coords(adj)
            d_m = _pool_metric(_structural_metric(adj), self.base_size)
            e_b = self.reconstructed_metric(w)
            loss = loss + F.mse_loss(d_m, e_b)
        loss = loss / max(len(adj_list), 1)
        div = adj_list[0].new_tensor(0.0)
        pairs = 0
        margin = 0.05
        for i in range(self.num_bases):
            for j in range(i + 1, self.num_bases):
                dist = (self.bases[i] - self.bases[j]).pow(2).mean().sqrt()
                div = div + F.relu(margin - dist)
                pairs += 1
        if pairs:
            loss = loss + div / pairs
        return loss

    def band_coord_loss(self, adj_low: torch.Tensor, adj_high: torch.Tensor) -> torch.Tensor:
        w_l = self.structural_coords(adj_low)
        w_h = self.structural_coords(adj_high)
        return F.mse_loss(w_l, w_h)


def apply_scgw_p1_features(
    features: torch.Tensor,
    adj: torch.Tensor,
    scgw: GeometricBasesSCGW,
    blend: float = 0.3,
) -> torch.Tensor:
    """Blend PCA features with structure-projected features."""
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    dev = features.device
    if scgw.bases.device != dev:
        scgw = scgw.to(dev)
    proj = scgw.project_node_features(features, adj)
    b = float(max(0.0, min(1.0, blend)))
    return (1.0 - b) * features.float() + b * proj


def maybe_create_scgw(args, feat_dim: int) -> GeometricBasesSCGW | None:
    enabled = any(
        [
            getattr(args, "scgw_p1", False),
            getattr(args, "scgw_p2", False),
            getattr(args, "scgw_p3", False),
            getattr(args, "scgw_p4", False),
        ]
    )
    if not enabled:
        return None
    if getattr(args, "scgw_p4", False) and not getattr(args, "use_band_gsl", False):
        raise ValueError("--scgw_p4 requires --use_band_gsl")
    return GeometricBasesSCGW(
        num_bases=getattr(args, "scgw_num_bases", 8),
        base_size=getattr(args, "scgw_base_size", 16),
        feat_dim=feat_dim,
        tau=getattr(args, "scgw_tau", 1.0),
    )
