"""GFMate + R-GFM plugins for original MDGFM downprompt path."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_node_matrix(h: torch.Tensor) -> torch.Tensor:
    """Normalize GCN outputs to (num_nodes, dim)."""
    if h.dim() == 3 and h.shape[0] == 1:
        h = h.squeeze(0)
    elif h.dim() == 1:
        h = h.unsqueeze(0)
    return h


def gcn_encode(
    gcn: nn.Module,
    seq: torch.Tensor,
    adj: torch.Tensor,
    sparse: bool,
    lp: bool = False,
    return_all_layers: bool = False,
):
    """Run GCN backbone; optionally return per-layer node embeddings."""
    if return_all_layers and hasattr(gcn, "forward"):
        try:
            out = gcn(seq, adj, sparse, lp, return_all_layers=True)
            if isinstance(out, tuple):
                final, layer_list = out
                final = as_node_matrix(final.squeeze())
                layers = [as_node_matrix(h) for h in layer_list]
                return final, layers
        except TypeError:
            pass
    final = as_node_matrix(gcn(seq, adj, sparse, lp).squeeze())
    return final, [final]


class TargetOnlyPrompt(nn.Module):
    """GFMate-style target-only prompt: skip multi-source meta token composition."""

    def __init__(self, dim: int, prompt_type: str = "mul"):
        super().__init__()
        self.open_prompt = nn.Parameter(torch.empty(1, dim))
        nn.init.xavier_uniform_(self.open_prompt)
        self.prompt_type = prompt_type

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        p = self.open_prompt
        if self.prompt_type == "add":
            return seq + p.expand_as(seq)
        return seq * p.expand_as(seq)


class LayerPromptEnsemble(nn.Module):
    """GFMate layer prompt η: softmax weights over per-layer prototype logits."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = max(1, int(num_layers))
        init = torch.zeros(self.num_layers)
        init[-1] = 3.0
        self.logits = nn.Parameter(init)

    def weights(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def ensemble_logits(self, layer_logits: list[torch.Tensor]) -> torch.Tensor:
        w = self.weights()
        out = layer_logits[0].new_zeros(layer_logits[0].shape)
        for i, lg in enumerate(layer_logits):
            out = out + w[i] * lg
        return out


def cosine_proto_matrix(query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    q = F.normalize(query, dim=1)
    p = F.normalize(prototypes, dim=1)
    return torch.mm(q, p.t())


def build_layer_prototypes(
    layer_embeds: list[torch.Tensor],
    labels: torch.Tensor,
    num_classes: int,
) -> list[torch.Tensor]:
    import torch_scatter

    protos = []
    for emb in layer_embeds:
        p = torch_scatter.scatter(src=emb, index=labels, dim=0, reduce="mean")
        if p.shape[0] < num_classes:
            pad = emb.mean(dim=0, keepdim=True).expand(num_classes - p.shape[0], -1)
            p = torch.cat([p, pad], dim=0)
        protos.append(p[:num_classes])
    return protos


def layer_proto_probs(
    layer_embeds: list[torch.Tensor],
    layer_protos: list[torch.Tensor],
    layer_prompt: LayerPromptEnsemble | None,
    centroid_offset: torch.Tensor | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """Per-layer cosine prototype logits -> softmax probs -> η-weighted ensemble."""
    layer_logits = []
    for emb, proto in zip(layer_embeds, layer_protos):
        p = proto
        if centroid_offset is not None:
            if centroid_offset.dim() == 2:
                p = proto + centroid_offset
            else:
                p = proto + centroid_offset[len(layer_logits)]
        layer_logits.append(cosine_proto_matrix(emb, p) / max(float(tau), 1e-6))
    if layer_prompt is not None:
        logits = layer_prompt.ensemble_logits(layer_logits)
    else:
        logits = layer_logits[-1]
    return F.softmax(logits, dim=1)


def pivot_layer_from_entropy(layer_embeds: list[torch.Tensor], layer_protos: list[torch.Tensor]) -> int:
    """GFMate: pick layer with lowest mean entropy on test nodes."""
    ent = []
    for emb, proto in zip(layer_embeds, layer_protos):
        logits = cosine_proto_matrix(emb, proto)
        probs = F.softmax(logits, dim=1)
        h = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
        ent.append(h)
    return int(torch.stack(ent).argmin().item())


def adaptive_hop_subgraph_embeds(
    node_embeds: torch.Tensor,
    adj_dense: torch.Tensor,
    node_idx: torch.Tensor,
    max_hop: int,
) -> torch.Tensor:
    """R-GFM adaptive-hop: k-hop mean-pooled embeddings as subgraph nodes (B, K, d)."""
    max_hop = max(1, int(max_hop))
    idx = node_idx.long()
    h = as_node_matrix(node_embeds)
    n = h.shape[0]
    device = h.device
    adj = adj_dense.float()
    if adj.shape[0] != n:
        adj = adj[:n, :n]
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
    a_norm = adj / deg
    power = torch.eye(n, device=device)
    scales = []
    for _ in range(max_hop):
        power = torch.mm(a_norm, power)
        pooled = torch.mm(power, h)
        scales.append(pooled[idx])
    return torch.stack(scales, dim=1)


class GoGEncoder(nn.Module):
    """R-GFM Graph-of-Graphs: similarity-sparsified message passing over hop scales."""

    def __init__(self, dim: int, edge_ratio: float = 0.6):
        super().__init__()
        self.edge_ratio = float(edge_ratio)
        self.msg = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.msg.weight)
        self.residual_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x_sub: torch.Tensor) -> torch.Tensor:
        # x_sub: (B, K, d)
        b, k, d = x_sub.shape
        if k <= 1:
            return x_sub.mean(dim=1)
        x_n = F.normalize(x_sub, dim=-1)
        sim = torch.bmm(x_n, x_n.transpose(1, 2))
        prob = F.softmax(sim, dim=-1)
        max_edges = max(1, int(self.edge_ratio * k * (k - 1) / 2))
        out_nodes = []
        msg = self.msg(x_sub)
        for bi in range(b):
            flat = prob[bi].reshape(-1)
            flat = flat * (1.0 - torch.eye(k, device=x_sub.device).reshape(-1))
            n_keep = min(max_edges, int((flat > 0).sum().item()))
            if n_keep <= 0:
                out_nodes.append(msg[bi].mean(dim=0))
                continue
            _, top_idx = torch.topk(flat, k=n_keep)
            adj = torch.zeros(k, k, device=x_sub.device)
            rows = torch.div(top_idx, k, rounding_mode="floor")
            cols = top_idx % k
            for r, c in zip(rows.tolist(), cols.tolist()):
                w = prob[bi, r, c]
                adj[r, c] = w
                adj[c, r] = w
            deg = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
            agg = torch.mm(adj / deg, msg[bi])
            out_nodes.append(agg.mean(dim=0))
        return torch.stack(out_nodes, dim=0)


class RiemannianExpert(nn.Module):
    """Single geometry expert (hyperbolic / euclidean / spherical surrogate)."""

    def __init__(self, dim: int, geometry: str):
        super().__init__()
        self.geometry = geometry
        self.lin = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.lin.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lin(x)
        if self.geometry == "hyperbolic":
            return torch.sign(z) * torch.log1p(z.abs())
        if self.geometry == "spherical":
            return F.normalize(z, dim=-1)
        return z


class RiemannianMoE(nn.Module):
    """R-GFM dynamic MoE over hyperbolic / euclidean / spherical experts."""

    def __init__(self, dim: int, router_hidden: int = 64, top_m: int = 2):
        super().__init__()
        self.top_m = max(1, int(top_m))
        self.experts = nn.ModuleList([
            RiemannianExpert(dim, g) for g in ("hyperbolic", "euclidean", "spherical")
        ])
        hid = max(8, int(router_hidden))
        self.router = nn.Sequential(
            nn.Linear(dim, hid),
            nn.ReLU(),
            nn.Linear(hid, len(self.experts)),
        )

    def forward(self, x: torch.Tensor, dynamic_top_m: int | None = None) -> torch.Tensor:
        m = self.top_m if dynamic_top_m is None else max(1, int(dynamic_top_m))
        m = min(m, len(self.experts))
        logits = self.router(x)
        w = F.softmax(logits, dim=-1)
        topw, topi = torch.topk(w, k=m, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = x.new_zeros(x.shape)
        for j in range(m):
            idx = topi[:, j]
            weight = topw[:, j].unsqueeze(-1)
            for e_id, expert in enumerate(self.experts):
                mask = idx == e_id
                if mask.any():
                    out[mask] = out[mask] + weight[mask] * expert(x[mask])
        return out


def refine_node_embeds(
    node_embeds: torch.Tensor,
    adj_dense: torch.Tensor,
    node_idx: torch.Tensor,
    gog_encoder: GoGEncoder | None,
    riemann_moe: RiemannianMoE | None,
    max_hop: int = 3,
    tgcl_step: int = 0,
    tgcl_total: int = 20,
    homo_score: torch.Tensor | None = None,
    homo_min: float = 0.35,
) -> torch.Tensor:
    """Apply adaptive-hop GoG + optional Riemannian MoE on selected nodes."""
    if gog_encoder is None and riemann_moe is None:
        return node_embeds
    idx = node_idx.long()
    h = as_node_matrix(node_embeds)
    base = h[idx]
    homo_gate = 1.0
    if homo_score is not None:
        homo_gate = float(homo_score.reshape(-1)[0].clamp(0.0, 1.0).item())
        if homo_gate < homo_min and gog_encoder is not None and riemann_moe is None:
            return node_embeds
    if gog_encoder is not None:
        sub = adaptive_hop_subgraph_embeds(h, adj_dense, idx, max_hop)
        delta = gog_encoder(sub)
        scale = torch.sigmoid(gog_encoder.residual_scale)
        base = base + homo_gate * scale * (delta - base)
    if riemann_moe is not None:
        # confidence-aware dynamic top-m: large m early, small m later
        if tgcl_total > 0:
            frac = float(tgcl_step) / float(max(tgcl_total, 1))
            m = max(1, len(riemann_moe.experts) - int(frac * (len(riemann_moe.experts) - 1)))
        else:
            m = riemann_moe.top_m
        riemann_delta = riemann_moe(base, dynamic_top_m=m)
        base = base + homo_gate * riemann_delta
    out = h.clone()
    out[idx] = base
    return out
