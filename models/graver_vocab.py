"""GRAVER-style generative graph vocabulary utilities (NeurIPS 2025)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sym_normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.clamp_min(0.0)
    deg = adj.sum(dim=1).clamp_min(1e-6)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)


def extract_ego_subgraph(
    center: int,
    features: torch.Tensor,
    adj_dense: torch.Tensor,
    hop: int = 1,
    max_nodes: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (local_features, local_adj, center_local_index) for an ego subgraph."""
    n = features.shape[0]
    visited = {int(center)}
    frontier = {int(center)}
    for _ in range(max(1, hop)):
        nxt = set()
        for u in frontier:
            nbrs = (adj_dense[u] > 0).nonzero(as_tuple=False).view(-1).tolist()
            for v in nbrs:
                if v not in visited:
                    visited.add(v)
                    nxt.add(v)
        frontier = nxt
        if len(visited) >= max_nodes:
            break
    nodes = sorted(visited)[:max_nodes]
    idx = torch.tensor(nodes, device=features.device, dtype=torch.long)
    local_x = features.index_select(0, idx)
    local_a = adj_dense.index_select(0, idx).index_select(1, idx)
    center_local = (idx == int(center)).nonzero(as_tuple=True)[0]
    if center_local.numel() == 0:
        center_local = 0
    else:
        center_local = int(center_local.squeeze().item())
    return local_x, local_a, center_local


class GraphVocabularyBank:
    """Class-conditional structure/feature templates per source domain.

    The last class index is a domain-level wildcard template (label-agnostic),
    useful when source/target class semantics do not align (e.g. Cornell).
    """

    def __init__(self, num_domains: int, num_target_classes: int, feat_dim: int, vocab_size: int = 8):
        self.num_domains = num_domains
        self.num_target_classes = num_target_classes
        self.num_classes = num_target_classes + 1  # +1 wildcard slot
        self.wildcard_class = num_target_classes
        self.feat_dim = feat_dim
        self.vocab_size = vocab_size
        self.struct = torch.zeros(num_domains, self.num_classes, vocab_size, vocab_size)
        self.feat = torch.zeros(num_domains, self.num_classes, vocab_size, feat_dim)
        self.counts = torch.zeros(num_domains, self.num_classes)

    def to(self, device: torch.device) -> GraphVocabularyBank:
        self.struct = self.struct.to(device)
        self.feat = self.feat.to(device)
        self.counts = self.counts.to(device)
        return self

    def update(self, domain_id: int, class_id: int, adj: torch.Tensor, x: torch.Tensor) -> None:
        n = min(self.vocab_size, adj.shape[0], x.shape[0])
        if n <= 0:
            return
        a = adj[:n, :n].detach().cpu()
        f = x[:n].detach().cpu()
        c = self.counts[domain_id, class_id].item()
        if c <= 0:
            self.struct[domain_id, class_id, :n, :n] = a
            self.feat[domain_id, class_id, :n] = f
        else:
            alpha = 1.0 / (c + 1.0)
            self.struct[domain_id, class_id, :n, :n] = (
                (1.0 - alpha) * self.struct[domain_id, class_id, :n, :n] + alpha * a
            )
            self.feat[domain_id, class_id, :n] = (
                (1.0 - alpha) * self.feat[domain_id, class_id, :n] + alpha * f
            )
        self.counts[domain_id, class_id] += 1.0

    def get_template(self, domain_id: int, class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.struct[domain_id, class_id], self.feat[domain_id, class_id]

    def domain_feat_centroid(self, domain_id: int) -> torch.Tensor:
        """Mean feature template for MoE soft-target routing."""
        wf = self.wildcard_class
        tpl = self.feat[domain_id, wf]
        if self.counts[domain_id, wf] > 0:
            return tpl.mean(dim=0)
        valid = self.counts[domain_id, : self.num_target_classes] > 0
        if valid.any():
            return self.feat[domain_id, : self.num_target_classes][valid].mean(dim=(0, 1))
        return self.feat[domain_id].mean(dim=(0, 1))


class MoECoERouter(nn.Module):
    """Lightweight domain (MoE) + class (CoE) routing for vocabulary assembly."""

    def __init__(self, feat_dim: int, num_domains: int, num_classes: int, hidden: int = 64):
        super().__init__()
        h = max(16, hidden)
        self.phi = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.ReLU(),
            nn.Linear(h, feat_dim),
        )
        self.domain_proj = nn.Linear(feat_dim, num_domains)
        self.class_proj = nn.Linear(feat_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        class_proto: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.phi(x)
        sm = F.softmax(self.domain_proj(h), dim=-1)
        if class_proto is None:
            class_proto = h
        if class_proto.dim() == 1:
            class_proto = class_proto.unsqueeze(0).expand_as(h)
        sc = F.softmax(self.class_proj(torch.cat([h, class_proto], dim=-1)), dim=-1)
        return sm, sc

    @staticmethod
    def routing_entropy_loss(sm: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        sm = sm.clamp_min(1e-8)
        sc = sc.clamp_min(1e-8)
        return -(sm * sm.log()).sum(dim=-1).mean() - (sc * sc.log()).sum(dim=-1).mean()


def _sample_adj_from_prob(prob: torch.Tensor, device: torch.device) -> torch.Tensor:
    n = prob.shape[0]
    gen_a = torch.zeros((n, n), device=device)
    for i in range(n):
        gen_a[i, i] = 1.0
        for j in range(i + 1, n):
            p = prob[i, j].clamp(0.0, 1.0)
            v = 1.0 if torch.rand((), device=device) < p else 0.0
            gen_a[i, j] = v
            gen_a[j, i] = v
    return gen_a


def compose_vocab_from_bank(
    bank: GraphVocabularyBank,
    sm: torch.Tensor,
    sc: torch.Tensor,
    device: torch.device,
    noise: float,
    wildcard_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mix of domain/class templates, with wildcard structural fallback."""
    n_dom = bank.num_domains
    n_cls = bank.num_target_classes
    wa = torch.zeros((bank.vocab_size, bank.vocab_size), device=device)
    wx = torch.zeros((bank.vocab_size, bank.feat_dim), device=device)
    total_w = 0.0
    for d in range(n_dom):
        cls_mass = 0.0
        for c in range(n_cls):
            w = float(sm[d].item() * sc[c].item())
            if w <= 1e-8:
                continue
            cls_mass += w
            t_a, t_f = bank.get_template(d, c)
            if bank.counts[d, c] <= 0:
                continue
            wa = wa + w * t_a.to(device)
            wx = wx + w * t_f.to(device)
            total_w += w
        w_wild = float(sm[d].item()) * max(wildcard_weight, 1.0 - cls_mass)
        if w_wild > 1e-8 and bank.counts[d, bank.wildcard_class] > 0:
            t_a, t_f = bank.get_template(d, bank.wildcard_class)
            wa = wa + w_wild * t_a.to(device)
            wx = wx + w_wild * t_f.to(device)
            total_w += w_wild

    if total_w <= 1e-8:
        d = int(sm.argmax().item())
        c = int(sc.argmax().item())
        if bank.counts[d, c] <= 0:
            c = bank.wildcard_class
        wa, wx = bank.get_template(d, c)
        wa, wx = wa.to(device), wx.to(device)
    else:
        wa = (wa / total_w).clamp(0.0, 1.0)
        wx = wx / total_w

    if wa.abs().sum() <= 1e-8:
        wa = torch.eye(bank.vocab_size, device=device)
    if wx.abs().sum() <= 1e-8:
        wx = torch.zeros((bank.vocab_size, bank.feat_dim), device=device)

    if noise > 0:
        wa = wa * (1.0 - noise) + noise * 0.5
    gen_a = _sample_adj_from_prob(wa, device)
    gen_x = wx + noise * torch.randn_like(wx)
    return gen_a, gen_x


def augment_ego_with_vocab(
    ego_x: torch.Tensor,
    ego_a: torch.Tensor,
    center_local: int,
    vocab_a: torch.Tensor,
    vocab_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach generated vocabulary to ego graph (GRAVER Eq.15 style overlap)."""
    n_e = ego_x.shape[0]
    n_v = vocab_x.shape[0]
    total = n_e + n_v
    aug_x = torch.cat([ego_x, vocab_x], dim=0)
    aug_a = torch.zeros((total, total), device=ego_x.device, dtype=ego_a.dtype)
    aug_a[:n_e, :n_e] = ego_a
    aug_a[n_e:, n_e:] = vocab_a
    c = int(center_local)
    hub = c if 0 <= c < n_e else 0
    cross_w = ego_a[hub].mean().clamp(0.1, 1.0)
    aug_a[hub, n_e:] = cross_w
    aug_a[n_e:, hub] = cross_w
    aug_a = sym_normalize_adj(aug_a)
    return aug_x, aug_a
