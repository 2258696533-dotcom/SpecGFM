"""GCIL-style auxiliary objectives (AAAI 2024, causal graph contrastive learning).

Lightweight port for MDGFM pretraining: invariance + independence terms on
two embedding views, optional spectral low-pass adjacency for a third view.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_adj_sparse(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization D^{-1/2} A D^{-1/2} for coalesced sparse adj."""
    adj = adj.coalesce()
    n = adj.size(0)
    rows, cols = adj.indices()
    vals = adj.values().float()
    deg = torch.zeros(n, device=adj.device, dtype=vals.dtype)
    deg.index_add_(0, rows, vals)
    deg = deg.clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    norm_vals = vals * inv_sqrt[rows] * inv_sqrt[cols]
    return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape).coalesce()


def spectral_lowpass_adj(adj: torch.Tensor, mix: float = 0.85) -> torch.Tensor:
    """Low-pass surrogate: convex mix of identity and normalized adjacency."""
    adj = adj.coalesce()
    n = adj.size(0)
    s = normalize_adj_sparse(adj)
    eye_idx = torch.arange(n, device=adj.device)
    eye = torch.sparse_coo_tensor(
        torch.stack([eye_idx, eye_idx]),
        torch.ones(n, device=adj.device, dtype=s.values().dtype),
        (n, n),
    ).coalesce()
    mix = float(max(0.0, min(1.0, mix)))
    # I * (1-mix) + S * mix  (sparse add via dense is too heavy; use two spmm at use site)
    return s, eye, mix


def apply_lowpass(adj: torch.Tensor, mix: float = 0.85) -> torch.Tensor:
    s, eye, mix = spectral_lowpass_adj(adj, mix)
    n = adj.size(0)
    # Build explicit sparse sum: scale values
    s = s.coalesce()
    si, sv = s.indices(), s.values() * mix
    ei = eye.indices()
    ev = eye.values() * (1.0 - mix)
    idx = torch.cat([si, ei], dim=1)
    val = torch.cat([sv, ev])
    out = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
    # merge duplicate indices
    return out.coalesce()


def gcil_invariance_loss(z_a: torch.Tensor, z_b: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Dimension-wise alignment: maximize normalized inner products + std anchoring."""
    za = F.normalize(z_a, dim=0, p=2)
    zb = F.normalize(z_b, dim=0, p=2)
    align = -(za * zb).sum(dim=0).mean()
    sa = z_a.std(dim=0, unbiased=False)
    sb = z_b.std(dim=0, unbiased=False)
    t = float(target_std)
    std_pen = torch.sqrt((sa - t).pow(2)).mean() + torch.sqrt((sb - t).pow(2)).mean()
    return align + std_pen


def gcil_independence_loss(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance penalty (linear-kernel HSIC surrogate)."""
    z = F.normalize(z, dim=0, p=2)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(z.shape[0] - 1, 1)
    d = cov.shape[0]
    off = cov.pow(2).sum() - cov.diag().pow(2).sum()
    denom = max(d * (d - 1), 1)
    return off / denom


def gcil_pair_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    inv_weight: float = 1.0,
    indep_weight: float = 0.1,
    target_std: float = 1.0,
) -> torch.Tensor:
    inv = gcil_invariance_loss(z_a, z_b, target_std=target_std)
    indep = gcil_independence_loss(z_a) + gcil_independence_loss(z_b)
    return float(inv_weight) * inv + float(indep_weight) * indep
