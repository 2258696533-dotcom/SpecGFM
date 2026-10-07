"""Dual-head downstream utilities (shared by MDGFM.py and graver_downstream.py)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_prototypes(embeds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    protos = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            protos.append(embeds[mask].mean(dim=0))
        else:
            protos.append(embeds.mean(dim=0))
    return torch.stack(protos, dim=0)


def cosine_proto_logits(query_embeds: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    q = F.normalize(query_embeds, dim=1)
    p = F.normalize(prototypes, dim=1)
    return torch.mm(q, p.t())


def prototype_loss(embeds: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    target_proto = prototypes[labels]
    return ((embeds - target_proto) ** 2).mean()


class SupportGuidedFiLM(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, scale: float = 0.2):
        super().__init__()
        b = max(8, int(bottleneck))
        self.net = nn.Sequential(
            nn.Linear(dim, b),
            nn.ReLU(),
            nn.Linear(b, 2 * dim),
        )
        self.scale = float(scale)

    def forward(self, emb: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        if summary.dim() == 1:
            summary = summary.unsqueeze(0)
        gb = self.net(summary)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(gamma)
        beta = self.scale * torch.tanh(beta)
        return emb * gamma + beta


def prototype_graph_regularization(prototypes: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    p = F.normalize(prototypes, dim=1)
    sim = torch.mm(p, p.t())
    c = sim.shape[0]
    if c <= 1:
        return torch.tensor(0.0, device=sim.device, dtype=sim.dtype)
    mask = ~torch.eye(c, dtype=torch.bool, device=sim.device)
    return F.relu(sim[mask] - margin).mean()


def margin_ce_loss(proto_logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    if margin <= 0:
        return F.cross_entropy(proto_logits, labels)
    adjusted = proto_logits.clone()
    adjusted[torch.arange(labels.shape[0], device=labels.device), labels] -= margin
    return F.cross_entropy(adjusted, labels)


def dual_mix_logits(logits_linear, logits_proto, alpha_raw, fixed_alpha: float, alpha_override=None):
    if alpha_override is not None:
        return alpha_override * logits_linear + (1 - alpha_override) * logits_proto
    if alpha_raw is not None:
        w = torch.sigmoid(alpha_raw)
        return w * logits_linear + (1 - w) * logits_proto
    return fixed_alpha * logits_linear + (1 - fixed_alpha) * logits_proto


def srm_episode_alpha(
    logits_linear: torch.Tensor,
    logits_proto: torch.Tensor,
    base_alpha: float,
    temp: float,
) -> torch.Tensor:
    with torch.no_grad():
        conf_linear = F.softmax(logits_linear, dim=1).max(dim=1).values.mean()
        conf_proto = F.softmax(logits_proto, dim=1).max(dim=1).values.mean()
        base = float(max(min(base_alpha, 1.0 - 1e-4), 1e-4))
        base_logit = math.log(base / (1.0 - base))
        alpha = torch.sigmoid(
            logits_linear.new_tensor(base_logit) + float(temp) * (conf_linear - conf_proto)
        )
        return alpha.clamp(0.05, 0.95)


def dual_cls_loss(logits, labels, smoothing: float):
    if smoothing and smoothing > 0:
        return F.cross_entropy(logits, labels, label_smoothing=float(smoothing))
    return F.cross_entropy(logits, labels)


def hat_mix_logits(
    logits_topo: torch.Tensor,
    logits_feat: torch.Tensor,
    prototypes: torch.Tensor,
    gate_mlp: nn.Module,
    homo_score: torch.Tensor,
):
    conf_topo = F.softmax(logits_topo, dim=1).max(dim=1).values.mean()
    if prototypes.shape[0] > 1:
        p = F.normalize(prototypes, dim=1)
        sim = torch.mm(p, p.t())
        c = sim.shape[0]
        off_diag = sim[~torch.eye(c, dtype=torch.bool, device=sim.device)]
        proto_sep = (1.0 - off_diag.mean()).clamp(0.0, 1.0)
    else:
        proto_sep = conf_topo.new_tensor(0.5)
    stats = torch.stack([homo_score, conf_topo, proto_sep], dim=0).unsqueeze(0)
    gate = torch.sigmoid(gate_mlp(stats))
    mixed = gate * logits_topo + (1.0 - gate) * logits_feat
    return mixed, gate.squeeze()
