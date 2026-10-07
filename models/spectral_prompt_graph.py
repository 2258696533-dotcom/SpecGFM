"""Per-band spectral-aligned prompt graph (HS-GPPT §4.3 style)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sym_normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric degree normalization D^{-1/2} A D^{-1/2}."""
    adj = adj.clamp_min(0.0)
    deg = adj.sum(dim=1).clamp_min(1e-6)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)


class SpectralPromptGraph(nn.Module):
    """Learnable virtual nodes inserted into a band-specific target graph."""

    def __init__(
        self,
        feat_dim: int,
        num_prompt: int = 10,
        tau_inner: float = 0.5,
        tau_cross: float = 0.3,
    ):
        super().__init__()
        self.num_prompt = max(1, int(num_prompt))
        self.tau_inner = float(tau_inner)
        self.tau_cross = float(tau_cross)
        self.prompt_feats = nn.Parameter(torch.empty(self.num_prompt, feat_dim))
        nn.init.xavier_uniform_(self.prompt_feats)

    @staticmethod
    def align_prompt_features(prompt_raw: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        """Match prompt feature moments to the target graph (HS-GPPT Eq.10)."""
        mu_p = prompt_raw.mean(dim=0, keepdim=True)
        std_p = prompt_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
        mu_o = target_feats.mean(dim=0, keepdim=True)
        std_o = target_feats.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (prompt_raw - mu_p) / std_p * std_o + mu_o

    def build_augmented_graph(
        self,
        node_feats: torch.Tensor,
        band_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [N+P, d] features and dense normalized adjacency for one spectral band."""
        device = node_feats.device
        dtype = node_feats.dtype
        n, _ = node_feats.shape
        p = self.align_prompt_features(self.prompt_feats, node_feats)
        np_ = p.shape[0]

        p_norm = F.normalize(p, dim=1, p=2)
        x_norm = F.normalize(node_feats, dim=1, p=2)
        sim_pp = torch.mm(p_norm, p_norm.t())
        sim_po = torch.mm(p_norm, x_norm.t())

        inner = (sim_pp > self.tau_inner).float()
        inner = inner * sim_pp.clamp_min(0.0)
        inner = 0.5 * (inner + inner.t())

        cross = (sim_po > self.tau_cross).float()
        cross = cross * sim_po.clamp_min(0.0)

        total = n + np_
        aug_adj = torch.zeros((total, total), device=device, dtype=dtype)
        aug_adj[:n, :n] = band_adj
        aug_adj[n:, n:] = inner
        aug_adj[n:, :n] = cross
        aug_adj[:n, n:] = cross.t()
        aug_adj = _sym_normalize_adj(aug_adj)

        aug_feats = torch.cat([node_feats, p], dim=0)
        return aug_feats, aug_adj
