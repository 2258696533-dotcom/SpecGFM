"""Adaptive Propagation Prompt (AP) for downstream MDGFM.

GPR-inspired multi-hop adjacency fusion with learnable hop weights,
optionally conditioned on support-set homophily.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sym_normalize_adj_dense(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization D^{-1/2} (A + I) D^{-1/2} on dense adjacency."""
    adj = adj.float()
    n = adj.size(0)
    adj = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    rowsum = adj.sum(dim=1).clamp_min(1e-8)
    d_inv_sqrt = rowsum.pow(-0.5)
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def fuse_adj_powers(adj_dense: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Fuse normalized adjacency powers: sum_k w_k * norm(A)^k (k=0 is identity)."""
    s = sym_normalize_adj_dense(adj_dense)
    n = s.size(0)
    fused = torch.zeros_like(s)
    power = torch.eye(n, device=s.device, dtype=s.dtype)
    for i in range(weights.shape[0]):
        if i > 0:
            power = power @ s
        fused = fused + weights[i] * power
    return fused


class AdaptivePropagationPrompt(nn.Module):
    """Learn hop weights for adjacency propagation; homophily can shift emphasis across hops."""

    def __init__(self, num_hops: int = 3, homo_condition: bool = True):
        super().__init__()
        self.num_hops = int(num_hops)
        self.homo_condition = homo_condition
        self.logits = nn.Parameter(torch.zeros(self.num_hops))
        if homo_condition:
            self.homo_shift = nn.Parameter(torch.tensor(0.5))

    def hop_weights(self, homo_score: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.logits
        if self.homo_condition and homo_score is not None:
            h = homo_score.reshape(()).float().clamp(0.0, 1.0)
            shift = self.homo_shift * (0.5 - h)
            logits = logits.clone()
            denom = max(self.num_hops - 1, 1)
            for i in range(self.num_hops):
                logits[i] = logits[i] + shift * (float(i) / float(denom))
        return F.softmax(logits, dim=0)

    def forward(
        self,
        adj_dense: torch.Tensor,
        homo_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = self.hop_weights(homo_score)
        return fuse_adj_powers(adj_dense, w)
