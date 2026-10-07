"""SpecGFM branch core blocks (v3: homophily-gated routes).

Homophilic episodes (support homo > thresh): stay on original GNN+prototype path.
Heterophilic episodes: activate route-specific enhancements more strongly.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

HOMO_BYPASS_THRESH = 0.52


def heterophily_weight(
    homo_score: Optional[torch.Tensor],
    ref: torch.Tensor,
    default: float = 0.35,
) -> torch.Tensor:
    if homo_score is None:
        return ref.new_tensor(default)
    return (1.0 - homo_score.reshape(())).clamp(0.0, 1.0)


def homophilic_episode(homo_score: Optional[torch.Tensor], thresh: float = HOMO_BYPASS_THRESH) -> bool:
    if homo_score is None:
        return False
    return float(homo_score.reshape(())) > thresh


class ProGraphSubspacePrompt(nn.Module):
    """Homophily-scaled multi-subspace prompts (inactive on homophilic episodes)."""

    def __init__(self, dim: int, num_subspaces: int = 3, base_strength: float = 0.3):
        super().__init__()
        self.prompts = nn.Parameter(torch.empty(num_subspaces, dim))
        nn.init.xavier_uniform_(self.prompts, gain=0.15)
        self.mix_logits = nn.Parameter(torch.zeros(num_subspaces))
        self.base_strength = base_strength

    def forward(
        self,
        x: torch.Tensor,
        homo_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if homophilic_episode(homo_score):
            return x
        hetero = heterophily_weight(homo_score, x, default=0.5)
        strength = self.base_strength * (0.5 + 0.5 * hetero)
        weights = F.softmax(self.mix_logits, dim=0)
        delta = (weights.unsqueeze(1) * torch.tanh(self.prompts)).sum(dim=0)
        return x + strength * delta.unsqueeze(0) * x


class KnowledgeAwareViewMixer(nn.Module):
    """Low/high graph views for auxiliary contrast."""

    def __init__(self, init_homo_bias: float = 0.5):
        super().__init__()
        p = float(max(min(init_homo_bias, 1.0 - 1e-4), 1e-4))
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(p / (1.0 - p))))

    def forward(
        self,
        x: torch.Tensor,
        adj_dense: torch.Tensor,
        homo_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deg = adj_dense.sum(dim=1, keepdim=True).clamp_min(1e-6)
        a_norm = adj_dense / deg
        low_view = torch.mm(a_norm, x)
        high_view = x - low_view
        if homo_score is not None:
            alpha = homo_score.reshape(()).clamp(0.1, 0.9)
        else:
            alpha = torch.sigmoid(self.alpha_raw)
        mixed = alpha * low_view + (1.0 - alpha) * high_view
        return mixed, low_view, high_view


class GradientFingerprintEmbedder(nn.Module):
    """Support-gradient fingerprint -> domain embedding."""

    def __init__(self, stat_dim: int = 8, emb_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    @staticmethod
    def collect_stats(
        gcn: nn.Module,
        features: torch.Tensor,
        adjtot: torch.Tensor,
        sparse: bool,
        idx_train: torch.Tensor,
        train_lbls: torch.Tensor,
        nb_classes: int,
    ) -> torch.Tensor:
        was_training = gcn.training
        gcn.eval()
        try:
            feat = features.detach().clone().requires_grad_(True)
            adj = adjtot.detach()
            with torch.enable_grad():
                z = gcn(feat.unsqueeze(0), adj, sparse, None).squeeze(0)
                z_sup = z[idx_train]
                proto = torch_scatter.scatter(
                    src=z_sup, index=train_lbls, dim=0, reduce="mean", dim_size=nb_classes
                )
                sim = F.cosine_similarity(
                    z_sup.unsqueeze(1), proto.unsqueeze(0), dim=-1
                )
                logits = F.softmax(sim, dim=1)
                loss = F.cross_entropy(logits, train_lbls)
                grad_feat = torch.autograd.grad(
                    loss, feat, retain_graph=False, create_graph=False
                )[0]
            g_sup = grad_feat[idx_train].detach()
            z_det = z.detach()
            proto_det = proto.detach()
            proto_sep = (
                F.pairwise_distance(proto_det.unsqueeze(0), proto_det.unsqueeze(1))
                .mean()
                .clamp_min(1e-6)
            )
            conf = logits.max(dim=1).values.mean()
            stats = torch.stack(
                [
                    g_sup.norm(dim=1).mean(),
                    g_sup.norm(dim=1).std().clamp_min(1e-6),
                    g_sup.abs().mean(),
                    feat[idx_train].detach().std().clamp_min(1e-6),
                    z_sup.norm(dim=1).mean(),
                    z_sup.norm(dim=1).std().clamp_min(1e-6),
                    proto_sep,
                    conf,
                ],
                dim=0,
            )
            return stats.detach()
        finally:
            if was_training:
                gcn.train()

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.mlp(stats.unsqueeze(0)).squeeze(0)


class DomainConditionedAligner(nn.Module):
    """Homophily-scaled residual FiLM."""

    def __init__(self, feat_dim: int, domain_dim: int = 32, base_strength: float = 0.35):
        super().__init__()
        self.film = nn.Linear(domain_dim, 2 * feat_dim)
        self.base_strength = base_strength

    def forward(
        self,
        x: torch.Tensor,
        domain_emb: torch.Tensor,
        homo_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if homophilic_episode(homo_score):
            return x
        hetero = heterophily_weight(homo_score, x, default=0.5)
        strength = self.base_strength * (0.5 + 0.5 * hetero)
        gamma, beta = self.film(domain_emb).chunk(2, dim=-1)
        return x + strength * (x * torch.tanh(gamma) + beta)


class BiKTDualEncoder:
    @staticmethod
    def encode(
        gcn: nn.Module,
        features: torch.Tensor,
        adjtot: torch.Tensor,
        sparse: bool,
        lp: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_gnn = gcn(features.unsqueeze(0), adjtot, sparse, lp).squeeze(0)
        n = features.shape[0]
        eye = torch.eye(n, device=features.device, dtype=adjtot.dtype)
        z_mlp = gcn(features.unsqueeze(0), eye, False, lp).squeeze(0)
        return z_gnn, z_mlp


class BiKTMixer(nn.Module):
    """Homophily bypass + stronger MLP residual on heterophilic episodes."""

    def __init__(self, max_residual: float = 0.55):
        super().__init__()
        self.residual_logit = nn.Parameter(torch.tensor(0.0))
        self.max_residual = max_residual

    def forward(
        self,
        z_gnn: torch.Tensor,
        z_mlp: torch.Tensor,
        homo_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if homophilic_episode(homo_score):
            return z_gnn
        hetero = heterophily_weight(homo_score, z_gnn, default=0.5)
        scale = torch.sigmoid(self.residual_logit) * self.max_residual * hetero
        return z_gnn + scale * (z_mlp - z_gnn)


class HeteroSupportFiLM(nn.Module):
    """Support-guided FiLM on heterophilic BiKT episodes (analogous to Dual SGFM)."""

    def __init__(self, dim: int, bottleneck: int = 32, scale: float = 0.15):
        super().__init__()
        b = max(8, int(bottleneck))
        self.net = nn.Sequential(
            nn.Linear(dim, b),
            nn.ReLU(),
            nn.Linear(b, 2 * dim),
        )
        self.scale = float(scale)

    def forward(self, z: torch.Tensor, support_z: torch.Tensor) -> torch.Tensor:
        summary = support_z.mean(dim=0, keepdim=True)
        gb = self.net(summary)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(gamma)
        beta = self.scale * torch.tanh(beta)
        return z * gamma + beta


def effective_struct_alpha(
    base_alpha: torch.Tensor,
    homo_score: Optional[torch.Tensor],
    hetero_boost: float,
) -> torch.Tensor:
    """Lower struct_alpha on heterophilic episodes -> rely more on learned GSL adj."""
    if hetero_boost <= 0.0 or homophilic_episode(homo_score):
        return base_alpha
    hetero = heterophily_weight(homo_score, base_alpha, default=0.5)
    return base_alpha * (1.0 - float(hetero_boost) * hetero).clamp(0.05, 0.95)


def build_adjtot(downprompt, features1, adj, sparse, downk, struct_alpha_override=None):
    reseq1 = torch.sparse.mm(adj, features1) if sparse else torch.mm(adj, features1)
    reseq1 = torch.cat((features1, reseq1), dim=1)
    reseq111 = downprompt.balancedprompt(reseq1)
    adj1 = downprompt.learner.graph_process(downk, reseq111)
    if sparse:
        adj1 = adj1.to(features1.device)
    adj_dense = adj.to_dense() if sparse else adj
    sa = downprompt.struct_alpha if struct_alpha_override is None else struct_alpha_override
    adjtot = sa * adj_dense + (1.0 - sa) * adj1
    if downprompt.ap is not None:
        adjtot = downprompt.ap(
            adjtot, homo_score=getattr(downprompt, "_ap_homo_score", None)
        )
    return adjtot, adj_dense


def bikt_consistency_loss(
    z_gnn: torch.Tensor,
    z_mlp: torch.Tensor,
    idx: Optional[torch.Tensor] = None,
    weight: float = 0.05,
    homo_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if homophilic_episode(homo_score):
        return z_gnn.new_tensor(0.0)
    hetero = heterophily_weight(homo_score, z_gnn, default=0.5)
    eff_w = weight * hetero
    z_g = F.normalize(z_gnn if idx is None else z_gnn[idx], dim=1)
    z_m = F.normalize(z_mlp if idx is None else z_mlp[idx], dim=1)
    return z_g.new_tensor(float(eff_w)) * (1.0 - (z_g * z_m).sum(dim=1).mean())


def prograph_view_contrastive(
    z_low: torch.Tensor,
    z_high: torch.Tensor,
    idx: Optional[torch.Tensor] = None,
    weight: float = 0.05,
    homo_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if homophilic_episode(homo_score):
        return z_low.new_tensor(0.0)
    hetero = heterophily_weight(homo_score, z_low, default=0.5)
    eff_w = weight * hetero
    z_l = F.normalize(z_low if idx is None else z_low[idx], dim=1)
    z_h = F.normalize(z_high if idx is None else z_high[idx], dim=1)
    return z_l.new_tensor(float(eff_w)) * (1.0 - (z_l * z_h).sum(dim=1).mean())
