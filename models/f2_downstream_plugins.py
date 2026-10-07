"""F2 / F2+P4 downstream plugins (papers GCRD / GG / RGC).

Branches (hard homo router):
  dual  -> Dual head (homophilic episodes)
  bikt  -> BiKT / downprompt (heterophilic episodes); alias: hetero

E1–E3 (screened on plain F2):
  E1 GEE          -> dual
  E2 node_w       -> bikt
  E3 leaky        -> bikt

R1–R5 (intended on F2+P4):
  R1 weighted teacher prototypes -> bikt
  R2 multi-subclass / L_sub      -> dual
  R3 E2 salvage on F2+P4         -> bikt (reuse E2 flags)
  R4 GEE on Pubmed               -> dual (reuse E1 flags)
  R5 L_smo edge smoothness       -> bikt (optional dual)
  R6 winner transfer domains     -> no new code (reuse modes)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_BRANCH_CHOICES = ("off", "dual", "bikt", "hetero", "both")
_BRANCH_ALIASES = {"hetero": "bikt", "downprompt": "bikt", "dp": "bikt"}


def normalize_branch(branch: Optional[str]) -> str:
    b = (branch or "off").lower().strip()
    return _BRANCH_ALIASES.get(b, b)


def _branch_enabled(branch: str, target: str) -> bool:
    b = normalize_branch(branch)
    t = normalize_branch(target)
    if b == "off":
        return False
    if b == "both":
        return True
    return b == t


def gee_on_dual(args) -> bool:
    return _branch_enabled(getattr(args, "f2_gee_branch", "off"), "dual")


def gee_on_hetero(args) -> bool:
    return _branch_enabled(getattr(args, "f2_gee_branch", "off"), "bikt")


def gee_on_bikt(args) -> bool:
    return gee_on_hetero(args)


def node_w_on_hetero(args) -> bool:
    return _branch_enabled(getattr(args, "f2_node_w_branch", "off"), "bikt")


def node_w_on_bikt(args) -> bool:
    return node_w_on_hetero(args)


def node_w_on_dual(args) -> bool:
    return _branch_enabled(getattr(args, "f2_node_w_branch", "off"), "dual")


def leaky_on_hetero(args) -> bool:
    return _branch_enabled(getattr(args, "f2_leaky_branch", "off"), "bikt")


def leaky_on_bikt(args) -> bool:
    return leaky_on_hetero(args)


def leaky_on_dual(args) -> bool:
    return _branch_enabled(getattr(args, "f2_leaky_branch", "off"), "dual")


def teacher_on_bikt(args) -> bool:
    return _branch_enabled(getattr(args, "f2_teacher_branch", "off"), "bikt")


def teacher_on_dual(args) -> bool:
    return _branch_enabled(getattr(args, "f2_teacher_branch", "off"), "dual")


def subproto_on_dual(args) -> bool:
    return _branch_enabled(getattr(args, "f2_subproto_branch", "off"), "dual")


def subproto_on_bikt(args) -> bool:
    return _branch_enabled(getattr(args, "f2_subproto_branch", "off"), "bikt")


def lsmo_on_bikt(args) -> bool:
    return _branch_enabled(getattr(args, "f2_lsmo_branch", "off"), "bikt")


def lsmo_on_dual(args) -> bool:
    return _branch_enabled(getattr(args, "f2_lsmo_branch", "off"), "dual")


_PLUGIN_BRANCH_KEYS = (
    "f2_gee_branch",
    "f2_node_w_branch",
    "f2_leaky_branch",
    "f2_teacher_branch",
    "f2_subproto_branch",
    "f2_lsmo_branch",
)


def f2_plugins_need_adj(args) -> bool:
    return any(normalize_branch(getattr(args, k, "off")) != "off" for k in _PLUGIN_BRANCH_KEYS)


def row_normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    a = adj.float().clamp_min(0.0)
    deg = a.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return a / deg


def support_gee(
    adj_dense: torch.Tensor,
    labels: torch.Tensor,
    idx_support: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Supervised GEE Z=AW using support labels only (n x C)."""
    n = adj_dense.shape[0]
    device = adj_dense.device
    labels = labels.reshape(-1).long().to(device)
    idx = idx_support.reshape(-1).long().to(device)
    w = adj_dense.new_zeros(n, num_classes)
    if idx.numel() == 0:
        return w
    y = labels.index_select(0, idx)
    counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
    w[idx, y] = 1.0 / counts[y]
    return torch.mm(adj_dense.float(), w)


def leaky_propagate(
    h: torch.Tensor,
    adj_dense: torch.Tensor,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """h <- (1-a)*h + a * A_norm h  (retain self when a is small)."""
    a = float(max(0.0, min(1.0, alpha)))
    steps = max(0, int(steps))
    if steps == 0 or a <= 0.0:
        return h
    a_norm = row_normalize_adj(adj_dense)
    out = h
    for _ in range(steps):
        out = (1.0 - a) * out + a * torch.mm(a_norm, out)
    return out


def effective_leaky_alpha(args, homo_score: Optional[torch.Tensor]) -> float:
    base = float(getattr(args, "f2_leaky_alpha", 0.3))
    if not getattr(args, "f2_leaky_homo_cond", True) or homo_score is None:
        return base
    h = float(homo_score.reshape(()))
    # Hetero (low h) -> smaller alpha (less neighbor mixing).
    return base * (0.25 + 0.75 * max(0.0, min(1.0, h)))


def compute_node_weights(
    adj_dense: torch.Tensor,
    node_feats: torch.Tensor,
    labels: torch.Tensor,
    idx_labeled: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Higher weight for nodes whose neighbors agree in features / labels."""
    n = node_feats.shape[0]
    device = node_feats.device
    a_norm = row_normalize_adj(adj_dense)
    neigh = torch.mm(a_norm, node_feats)
    feat_diff = (node_feats - neigh).pow(2).mean(dim=1).clamp_min(0.0)
    feat_diff = feat_diff / feat_diff.mean().clamp_min(1e-6)

    label_diff = feat_diff.new_zeros(n)
    idx = idx_labeled.reshape(-1).long().to(device)
    if idx.numel() > 0:
        y = labels.reshape(-1).long().to(device)
        n_cls = int(y.max().item()) + 1
        y_oh = F.one_hot(y, num_classes=n_cls).float()
        neigh_y = torch.mm(a_norm, y_oh)
        ld = (y_oh - neigh_y).pow(2).mean(dim=1)
        label_diff = ld / ld[idx].mean().clamp_min(1e-6)
        mask = torch.zeros(n, device=device, dtype=torch.bool)
        mask[idx] = True
        label_diff = torch.where(mask, label_diff, torch.zeros_like(label_diff))

    score = feat_diff + label_diff
    w = torch.exp(-float(gamma) * score)
    return (w / w.mean().clamp_min(1e-6)).clamp(0.05, 3.0)


def _as_long_index(idx, device) -> torch.Tensor:
    if isinstance(idx, torch.Tensor):
        return idx.reshape(-1).long().to(device)
    return torch.as_tensor(list(idx), device=device, dtype=torch.long)


def weighted_class_prototypes(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Weighted mean prototype per class (teacher). Falls back to uniform mean."""
    device = embeds.device
    labels = labels.reshape(-1).long().to(device)
    w = weights.reshape(-1).float().to(device)
    if w.numel() != embeds.shape[0]:
        w = embeds.new_ones(embeds.shape[0])
    protos = []
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            protos.append(embeds.mean(dim=0))
            continue
        wc = w[mask].clamp_min(1e-6)
        wc = wc / wc.sum()
        protos.append((embeds[mask] * wc.unsqueeze(1)).sum(dim=0))
    return torch.stack(protos, dim=0)


def blend_teacher_prototypes(
    student_protos: torch.Tensor,
    teacher_protos: torch.Tensor,
    mix: float,
) -> torch.Tensor:
    m = float(max(0.0, min(1.0, mix)))
    if m <= 0.0:
        return student_protos
    if m >= 1.0:
        return teacher_protos
    return (1.0 - m) * student_protos + m * teacher_protos


def neighbor_augmented_supports(
    full_embeds: torch.Tensor,
    adj_dense: torch.Tensor,
    idx_support: torch.Tensor,
    labels_support: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand 1-shot supports with k-hop diffused self embeddings as virtual samples."""
    k = max(1, int(k))
    device = full_embeds.device
    idx = idx_support.reshape(-1).long().to(device)
    y = labels_support.reshape(-1).long().to(device)
    if k == 1 or adj_dense is None:
        return full_embeds.index_select(0, idx), y

    a_norm = row_normalize_adj(adj_dense.to(device))
    curated_h: List[torch.Tensor] = []
    curated_y: List[torch.Tensor] = []
    for i in range(idx.numel()):
        node = idx[i]
        h = full_embeds[node]
        curated_h.append(h)
        curated_y.append(y[i])
        prop = h
        for _ in range(k - 1):
            # Diffuse one step from a one-hot mass at `node`.
            mass = full_embeds.new_zeros(full_embeds.shape[0], 1)
            mass[node] = 1.0
            # Use feature propagation: A^t h_node via repeatedly applying A to embedding field
            # seeded with support embedding at node (others zero).
            field = full_embeds.new_zeros(full_embeds.shape)
            field[node] = prop
            prop = torch.mm(a_norm, field)[node]
            curated_h.append(prop)
            curated_y.append(y[i])
    return torch.stack(curated_h, dim=0), torch.stack(curated_y, dim=0)


def build_subclass_prototypes(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (class_mean_protos [C,d], flat_anchors [C*k,d] with padding)."""
    k = max(1, int(k))
    device = embeds.device
    labels = labels.reshape(-1).long().to(device)
    d = embeds.shape[1]
    means = []
    anchors = embeds.new_zeros(num_classes * k, d)
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            means.append(embeds.mean(dim=0))
            anchors[c * k : (c + 1) * k] = embeds.mean(dim=0)
            continue
        h = embeds[mask]
        means.append(h.mean(dim=0))
        if h.shape[0] >= k:
            # Farthest-point style diversity among class samples.
            chosen = [0]
            while len(chosen) < k:
                dist = torch.cdist(h, h[chosen]).min(dim=1).values
                dist[chosen] = -1.0
                chosen.append(int(dist.argmax().item()))
            sel = h[chosen]
        else:
            # Tile + small noise for shortfall (1-shot after neigh expand may still be short).
            reps = []
            for j in range(k):
                reps.append(h[j % h.shape[0]])
            sel = torch.stack(reps, dim=0)
            if k > h.shape[0]:
                noise = 0.01 * torch.randn_like(sel)
                noise[: h.shape[0]] = 0.0
                sel = sel + noise
        anchors[c * k : (c + 1) * k] = sel
    return torch.stack(means, dim=0), anchors


def subclass_lsub_loss(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    anchors: Optional[torch.Tensor] = None,
    k: int = 2,
) -> torch.Tensor:
    """GCRD-style L_sub: match standardized subclass–centroid geometry.

    For each class, d_j = (a_j - mu) / sigma; encourage ||d|| ~ O(1) and
    pull sample embeds toward nearest subclass anchor (shape, not contrastive push).
    """
    device = embeds.device
    labels = labels.reshape(-1).long().to(device)
    if anchors is None:
        _, anchors = build_subclass_prototypes(embeds, labels, num_classes, k)
    k = max(1, int(k))
    losses = []
    for c in range(num_classes):
        a = anchors[c * k : (c + 1) * k]
        mu = a.mean(dim=0)
        sigma = a.std(dim=0).clamp_min(1e-6)
        d = (a - mu) / sigma
        r = d.norm(dim=1)
        shape = (r.mean() - 1.0).pow(2) + r.var(unbiased=False)
        mask = labels == c
        if mask.any():
            h = embeds[mask]
            # Soft assign to nearest anchor; pull without pushing other classes.
            dist = torch.cdist(h, a)
            pull = dist.min(dim=1).values.mean()
            losses.append(shape + pull)
        else:
            losses.append(shape)
    if not losses:
        return embeds.new_zeros(())
    return torch.stack(losses).mean()


def lsmo_embed_loss(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    adj_ss: torch.Tensor,
) -> torch.Tensor:
    """Label-homophily edge smoothness on a support subgraph (GCRD L_smo spirit).

    Same-label edges: pull embeds; cross-label edges: soft hinge push.
    """
    if embeds.shape[0] < 2:
        return embeds.new_zeros(())
    a = adj_ss.float().clamp_min(0.0)
    # Keep undirected mass without self-loops dominating.
    a = a - torch.diag(torch.diag(a))
    if float(a.sum()) <= 0:
        # Fallback: fully-connected light graph among supports.
        n = embeds.shape[0]
        a = embeds.new_ones(n, n)
        a.fill_diagonal_(0.0)
    a = 0.5 * (a + a.t())
    y = labels.reshape(-1).long()
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    diff = 1.0 - same
    same.fill_diagonal_(0.0)
    diff.fill_diagonal_(0.0)
    dist2 = ((embeds.unsqueeze(0) - embeds.unsqueeze(1)) ** 2).sum(dim=-1)
    pull = (a * same * dist2).sum() / same.sum().clamp_min(1.0)
    # Soft push: only penalize if too close on cross edges with adjacency mass.
    margin = 1.0
    push = (a * diff * F.relu(margin - dist2.sqrt().clamp_min(0.0)).pow(2)).sum()
    push = push / diff.sum().clamp_min(1.0)
    return pull + 0.5 * push


def apply_f2_dual_plugins(
    args,
    all_embeds: torch.Tensor,
    adj_dense: torch.Tensor,
    labels: torch.Tensor,
    idx_train,
    idx_test,
    nb_classes: int,
    homo_score: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (train_embs, test_embs, in_dim) for Dual head."""
    h = all_embeds
    if h.dim() == 3:
        h = h.squeeze(0)
    device = h.device
    idx_tr = _as_long_index(idx_train, device)
    idx_te = _as_long_index(idx_test, device)
    if leaky_on_dual(args):
        alpha = effective_leaky_alpha(args, homo_score)
        h = leaky_propagate(h, adj_dense, alpha, int(getattr(args, "f2_leaky_steps", 2)))
    if node_w_on_dual(args):
        w = compute_node_weights(
            adj_dense, h.detach(), labels, idx_tr,
            gamma=float(getattr(args, "f2_node_w_gamma", 1.0)),
        )
        h = h * w.unsqueeze(1)
    pe = h.index_select(0, idx_tr)
    te = h.index_select(0, idx_te)
    if gee_on_dual(args):
        gee = support_gee(adj_dense, labels, idx_tr, nb_classes)
        pe = torch.cat([pe, gee.index_select(0, idx_tr)], dim=-1)
        te = torch.cat([te, gee.index_select(0, idx_te)], dim=-1)
    return pe, te, int(pe.shape[1])


def apply_hetero_embed_plugins(
    embeds: torch.Tensor,
    adj_dense: Optional[torch.Tensor],
    features: Optional[torch.Tensor],
    labels: torch.Tensor,
    idx_support: torch.Tensor,
    args,
    homo_score: Optional[torch.Tensor],
) -> torch.Tensor:
    """Post-process node embeds on BiKT/downprompt path (E2/E3)."""
    h = embeds
    if h.dim() == 3 and h.shape[0] == 1:
        h = h.squeeze(0)
    if adj_dense is None:
        return h
    if adj_dense.dim() == 3 and adj_dense.shape[0] == 1:
        adj_dense = adj_dense.squeeze(0)
    if leaky_on_hetero(args):
        alpha = effective_leaky_alpha(args, homo_score)
        h = leaky_propagate(h, adj_dense, alpha, int(getattr(args, "f2_leaky_steps", 2)))
    if node_w_on_hetero(args):
        feat = features if features is not None else h.detach()
        if feat.dim() == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        if feat.shape[0] != h.shape[0]:
            feat = h.detach()
        w = compute_node_weights(
            adj_dense, feat, labels, idx_support,
            gamma=float(getattr(args, "f2_node_w_gamma", 1.0)),
        )
        h = h * w.unsqueeze(1)
    return h


def apply_bikt_teacher_prototypes(
    student_ave: torch.Tensor,
    support_embeds: torch.Tensor,
    support_labels: torch.Tensor,
    adj_dense: Optional[torch.Tensor],
    features: Optional[torch.Tensor],
    idx_support: torch.Tensor,
    labels_all: torch.Tensor,
    args,
    num_classes: int,
) -> torch.Tensor:
    """R1: blend uniform BiKT prototypes with weighted teacher prototypes."""
    if not teacher_on_bikt(args):
        return student_ave
    if adj_dense is None:
        return student_ave
    feat = features if features is not None else support_embeds.detach()
    if feat.dim() == 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    # Weights on full graph; use only support slice for prototype mix.
    w_full = compute_node_weights(
        adj_dense, feat if feat.shape[0] == adj_dense.shape[0] else support_embeds.detach(),
        labels_all, idx_support,
        gamma=float(getattr(args, "f2_teacher_gamma", getattr(args, "f2_node_w_gamma", 1.0))),
    )
    w_sup = w_full.index_select(0, idx_support.reshape(-1).long().to(w_full.device))
    if w_sup.numel() != support_embeds.shape[0]:
        w_sup = support_embeds.new_ones(support_embeds.shape[0])
    teacher = weighted_class_prototypes(
        support_embeds, support_labels, w_sup, num_classes,
    )
    # class count mismatch guard (scattered averageemb can be short)
    c = min(student_ave.shape[0], teacher.shape[0])
    out = student_ave.clone()
    out[:c] = blend_teacher_prototypes(
        student_ave[:c], teacher[:c], float(getattr(args, "f2_teacher_mix", 1.0)),
    )
    return out


def dual_plugin_aux_loss(
    args,
    pe: torch.Tensor,
    train_lbls: torch.Tensor,
    nb_classes: int,
    adj_dense: Optional[torch.Tensor] = None,
    full_embeds: Optional[torch.Tensor] = None,
    idx_train=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """R2/R5 Dual aux: (proto_for_logits, extra_loss).

    Returns class prototypes (possibly from neighbor-augmented subclass means)
    and an auxiliary loss tensor (may be 0).
    """
    from dual_training import build_prototypes

    device = pe.device
    labels = train_lbls.reshape(-1).long().to(device)
    proto = build_prototypes(pe, labels, nb_classes)
    extra = pe.new_zeros(())

    use_sub = subproto_on_dual(args)
    use_lsmo = lsmo_on_dual(args)
    if not use_sub and not use_lsmo:
        return proto, extra

    k = int(getattr(args, "f2_subproto_k", 2))
    aug_pe, aug_y = pe, labels
    if use_sub and full_embeds is not None and adj_dense is not None and idx_train is not None:
        idx = _as_long_index(idx_train, device)
        aug_pe, aug_y = neighbor_augmented_supports(
            full_embeds if full_embeds.dim() == 2 else full_embeds.squeeze(0),
            adj_dense, idx, labels, k=k,
        )

    anchors = None
    if use_sub:
        proto, anchors = build_subclass_prototypes(aug_pe, aug_y, nb_classes, k)
        w_lsub = float(getattr(args, "f2_lsub_weight", 0.1))
        if w_lsub > 0:
            extra = extra + w_lsub * subclass_lsub_loss(
                aug_pe, aug_y, nb_classes, anchors=anchors, k=k,
            )

    if use_lsmo and adj_dense is not None and idx_train is not None:
        idx = _as_long_index(idx_train, device)
        adj_ss = adj_dense.index_select(0, idx).index_select(1, idx)
        w_lsmo = float(getattr(args, "f2_lsmo_weight", 0.05))
        if w_lsmo > 0:
            extra = extra + w_lsmo * lsmo_embed_loss(pe, labels, adj_ss)

    if teacher_on_dual(args) and adj_dense is not None and idx_train is not None and full_embeds is not None:
        idx = _as_long_index(idx_train, device)
        h_full = full_embeds if full_embeds.dim() == 2 else full_embeds.squeeze(0)
        labels_all = torch.zeros(h_full.shape[0], device=device, dtype=torch.long)
        labels_all[idx] = labels
        w_full = compute_node_weights(
            adj_dense, h_full.detach(), labels_all, idx,
            gamma=float(getattr(args, "f2_teacher_gamma", 1.0)),
        )
        teacher = weighted_class_prototypes(pe, labels, w_full.index_select(0, idx), nb_classes)
        proto = blend_teacher_prototypes(
            proto, teacher, float(getattr(args, "f2_teacher_mix", 1.0)),
        )

    return proto, extra


def bikt_plugin_aux_loss(
    args,
    train_embs: torch.Tensor,
    train_lbls: torch.Tensor,
    adj_dense: Optional[torch.Tensor],
    idx_support: torch.Tensor,
) -> torch.Tensor:
    """R5 aux on BiKT path (and optional subproto if enabled on bikt)."""
    extra = train_embs.new_zeros(())
    if lsmo_on_bikt(args) and adj_dense is not None:
        idx = idx_support.reshape(-1).long().to(adj_dense.device)
        if idx.numel() == train_embs.shape[0]:
            adj_ss = adj_dense.index_select(0, idx).index_select(1, idx)
        else:
            # train_embs already support-sized; use identity-ish fallback inside loss
            n = train_embs.shape[0]
            adj_ss = train_embs.new_ones(n, n)
        w = float(getattr(args, "f2_lsmo_weight", 0.05))
        if w > 0:
            extra = extra + w * lsmo_embed_loss(train_embs, train_lbls, adj_ss)
    if subproto_on_bikt(args):
        k = int(getattr(args, "f2_subproto_k", 2))
        w = float(getattr(args, "f2_lsub_weight", 0.1))
        if w > 0:
            extra = extra + w * subclass_lsub_loss(
                train_embs, train_lbls, int(train_lbls.max().item()) + 1 if train_lbls.numel() else 1, k=k,
            )
    return extra
