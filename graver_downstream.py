"""GRAVER downstream: vocabulary bank + support augmentation on frozen MDGFM backbone."""

from __future__ import annotations

import random
from typing import Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from downstream_encoder import build_downstream_encoder_from_model
from dual_training import (
    SupportGuidedFiLM,
    build_prototypes,
    cosine_proto_logits,
    dual_cls_loss,
    dual_mix_logits,
    hat_mix_logits,
    margin_ce_loss,
    prototype_graph_regularization,
    prototype_loss,
    srm_episode_alpha,
)
from models.graver_vocab import (
    GraphVocabularyBank,
    MoECoERouter,
    augment_ego_with_vocab,
    compose_vocab_from_bank,
    extract_ego_subgraph,
)


def build_vocabulary_bank(
    source_features: Sequence[torch.Tensor],
    source_adjs,
    source_labels: Sequence[torch.Tensor],
    target_num_classes: int,
    vocab_size: int = 8,
    ego_hop: int = 1,
    max_ego_nodes: int = 24,
    samples_per_class: int = 20,
    wildcard_per_domain: int = 30,
    seed: int = 0,
) -> GraphVocabularyBank:
    """Build class + wildcard graphon templates from multi-domain source graphs."""
    feat_dim = int(source_features[0].shape[1])
    bank = GraphVocabularyBank(
        len(source_features), target_num_classes, feat_dim, vocab_size
    )
    rng = random.Random(seed)
    for d, (feat, adj_sp, lbls) in enumerate(zip(source_features, source_adjs, source_labels)):
        adj = adj_sp.to_dense() if adj_sp.is_sparse else adj_sp
        feat = feat.float()
        lbls = lbls.view(-1).long()
        num_src_cls = int(lbls.max().item()) + 1 if lbls.numel() > 0 else 0

        for c in range(min(target_num_classes, num_src_cls)):
            nodes = (lbls == c).nonzero(as_tuple=True)[0].tolist()
            if not nodes:
                continue
            rng.shuffle(nodes)
            for u in nodes[:samples_per_class]:
                ex, ea, _ = extract_ego_subgraph(
                    int(u), feat, adj.float(), hop=ego_hop, max_nodes=max_ego_nodes
                )
                bank.update(d, c, ea, ex)

        all_nodes = list(range(feat.shape[0]))
        rng.shuffle(all_nodes)
        for u in all_nodes[:wildcard_per_domain]:
            ex, ea, _ = extract_ego_subgraph(
                int(u), feat, adj.float(), hop=ego_hop, max_nodes=max_ego_nodes
            )
            bank.update(d, bank.wildcard_class, ea, ex)

    return bank.to(source_features[0].device)


def _domain_soft_targets(
    x: torch.Tensor,
    bank: GraphVocabularyBank,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Cosine-similarity soft targets over source domains for MoE routing."""
    centroids = torch.stack(
        [bank.domain_feat_centroid(d) for d in range(bank.num_domains)],
        dim=0,
    ).to(x.device)
    sim = F.cosine_similarity(x.unsqueeze(0), centroids, dim=-1)
    return F.softmax(sim / max(temperature, 1e-4), dim=-1)


def encode_augmented_support(
    encoder: nn.Module,
    features: torch.Tensor,
    adj,
    gcn: nn.Module,
    downk: int,
    idx_train: torch.Tensor,
    train_lbls: torch.Tensor,
    bank: GraphVocabularyBank,
    router: MoECoERouter,
    global_support_embs: torch.Tensor | None,
    ego_hop: int,
    max_ego_nodes: int,
    vocab_noise: float,
    global_weight: float,
    wildcard_weight: float,
) -> torch.Tensor:
    """Per-support-node GRAVER augmentation + downstream encode; returns [S, hid]."""
    adj_dense = adj.to_dense() if adj.is_sparse else adj
    device = features.device
    class_protos = []
    for c in range(bank.num_target_classes):
        mask = train_lbls == c
        if mask.any():
            class_protos.append(features.index_select(0, idx_train[mask]).mean(dim=0))
        else:
            class_protos.append(features.index_select(0, idx_train).mean(dim=0))
    class_protos = torch.stack(class_protos, dim=0)

    support_embeds = []
    encoder.eval()
    gcn.eval()
    gw = float(max(0.0, min(global_weight, 1.0)))
    with torch.no_grad():
        for j, (u, y) in enumerate(zip(idx_train.tolist(), train_lbls.tolist())):
            ego_x, ego_a, center_local = extract_ego_subgraph(
                int(u), features, adj_dense.float(), hop=ego_hop, max_nodes=max_ego_nodes
            )
            x_u = features[int(u)]
            proto = class_protos[int(y)]
            sm, sc = router(x_u.unsqueeze(0), proto.unsqueeze(0))
            sm = sm.squeeze(0)
            sc = sc.squeeze(0)
            va, vx = compose_vocab_from_bank(
                bank, sm, sc, device, vocab_noise, wildcard_weight=wildcard_weight
            )
            aug_x, aug_a = augment_ego_with_vocab(ego_x, ego_a, center_local, va, vx)
            # Local subgraphs can be much smaller than full-graph downk (e.g. 30).
            local_k = max(1, min(downk, aug_x.shape[0] - 1))
            local_emb = encoder.encode_graph(aug_x, aug_a, False, gcn, local_k).squeeze()
            if local_emb.dim() > 1:
                local_emb = local_emb[center_local]
            if global_support_embs is not None and gw > 0:
                global_emb = global_support_embs[j]
                local_emb = (1.0 - gw) * local_emb + gw * global_emb
            support_embeds.append(local_emb)
    return torch.stack(support_embeds, dim=0)


def run_dual_graver_episode(
    args,
    model,
    features,
    sp_adj,
    sparse,
    downk,
    hid_units,
    nb_classes,
    unify_dim,
    idx_train,
    idx_test,
    train_lbls,
    homo_score,
    downstreamlr,
    seed,
    episode_i,
    shotnum,
    vocab_bank: GraphVocabularyBank,
    global_test_embs: torch.Tensor,
    global_support_embs: torch.Tensor | None = None,
):
    """GRAVER-augmented support + frozen global query embeddings + Dual head."""
    device = features.device
    encoder = build_downstream_encoder_from_model(
        model, args.combinetype, unify_dim, hid_units, use_ap=False,
        ap_num_hops=0, ap_homo_cond=False,
    ).cuda()
    router = MoECoERouter(
        unify_dim, vocab_bank.num_domains, vocab_bank.num_target_classes,
        hidden=max(16, int(args.graver_router_hidden)),
    ).cuda()

    class_protos = []
    for c in range(nb_classes):
        mask = train_lbls == c
        if mask.any():
            class_protos.append(features.index_select(0, idx_train[mask]).mean(dim=0))
        else:
            class_protos.append(features.index_select(0, idx_train).mean(dim=0))
    class_protos = torch.stack(class_protos, dim=0)

    router_opt = torch.optim.Adam(router.parameters(), lr=downstreamlr)
    moe_aux = float(getattr(args, "graver_moe_aux_weight", 0.1))
    for _ in range(max(1, int(args.graver_router_epochs))):
        router.train()
        router_opt.zero_grad()
        sm_list, sc_list = [], []
        loss = 0.0
        n_sup = max(1, len(idx_train))
        for u, y in zip(idx_train.tolist(), train_lbls.tolist()):
            x_u = features[int(u)]
            sm, sc = router(x_u.unsqueeze(0), class_protos[int(y)].unsqueeze(0))
            sm = sm.squeeze(0)
            sc = sc.squeeze(0)
            sm_list.append(sm.unsqueeze(0))
            sc_list.append(sc.unsqueeze(0))
            loss = loss - torch.log(sc[int(y)].clamp_min(1e-8))
            if moe_aux > 0:
                sm_tgt = _domain_soft_targets(x_u, vocab_bank, temperature=0.2)
                loss = loss + moe_aux * F.kl_div(
                    sm.log().unsqueeze(0), sm_tgt.unsqueeze(0), reduction="batchmean"
                )
        sm = torch.cat(sm_list, dim=0)
        sc = torch.cat(sc_list, dim=0)
        loss = loss / n_sup
        if args.graver_moe_weight > 0:
            loss = loss + args.graver_moe_weight * MoECoERouter.routing_entropy_loss(sm, sc)
        loss.backward()
        router_opt.step()

    pretrain_embs = encode_augmented_support(
        encoder, features, sp_adj, model.gcn, downk,
        idx_train, train_lbls, vocab_bank, router,
        global_support_embs,
        args.graver_ego_hop, args.graver_max_ego_nodes, args.graver_vocab_noise,
        getattr(args, "graver_global_weight", 0.4),
        getattr(args, "graver_wildcard_weight", 0.25),
    ).detach()
    test_embs = global_test_embs.detach()

    K = max(1, int(args.dual_ensemble))
    logits_accum = None
    ens_base = int(seed) + episode_i * 100003 + int(shotnum) * 9973
    for k_m in range(K):
        torch.manual_seed(ens_base + k_m * 1_000_003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ens_base + k_m * 1_000_003)
        dual_head = nn.Linear(hid_units, nb_classes).cuda()
        dual_params = list(dual_head.parameters())
        alpha_raw = None
        if args.learn_dual_alpha:
            p = float(max(min(args.dual_alpha, 1.0 - 1e-4), 1e-4))
            init_logit = math.log(p / (1.0 - p))
            alpha_raw = nn.Parameter(torch.tensor(init_logit, device=device, dtype=torch.float32))
            dual_params.append(alpha_raw)
        if args.dual_calib:
            calib_scale = nn.Parameter(torch.ones(1, nb_classes, device=device))
            calib_bias = nn.Parameter(torch.zeros(1, nb_classes, device=device))
            dual_params += [calib_scale, calib_bias]
        else:
            calib_scale = None
            calib_bias = None
        sgfm_mod = None
        if args.use_sgfm:
            sgfm_mod = SupportGuidedFiLM(
                hid_units, bottleneck=args.sgfm_bottleneck, scale=args.sgfm_scale,
            ).cuda()
            dual_params.extend(list(sgfm_mod.parameters()))
        gate_mlp = None
        if args.use_hat_adapter:
            gate_hidden = max(4, int(args.hat_gate_hidden))
            gate_mlp = nn.Sequential(
                nn.Linear(3, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1),
            ).cuda()
            dual_params.extend(list(gate_mlp.parameters()))
        opt = torch.optim.Adam(dual_params, lr=downstreamlr)
        for _ in range(args.dual_epochs):
            dual_head.train()
            pe = pretrain_embs
            if sgfm_mod is not None:
                sgfm_mod.train()
                pe = sgfm_mod(pe, pe.mean(dim=0, keepdim=True))
            opt.zero_grad()
            logits_linear = dual_head(pe)
            proto = build_prototypes(pe, train_lbls, nb_classes)
            logits_proto = cosine_proto_logits(pe, proto) / max(args.dual_tau, 1e-6)
            alpha_srm = None
            if args.use_srm:
                alpha_base = torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                alpha_srm = srm_episode_alpha(logits_linear, logits_proto, alpha_base, args.srm_temp)
            if gate_mlp is not None:
                logits_mix, gate_val = hat_mix_logits(
                    logits_linear, logits_proto, proto, gate_mlp, homo_score
                )
            else:
                logits_mix = dual_mix_logits(
                    logits_linear, logits_proto, alpha_raw, args.dual_alpha, alpha_override=alpha_srm
                )
            if args.dual_calib:
                logits_mix = logits_mix * calib_scale + calib_bias
            cls_loss = dual_cls_loss(logits_mix, train_lbls, args.dual_label_smoothing)
            if args.proto_margin_weight > 0:
                cls_loss = cls_loss + args.proto_margin_weight * margin_ce_loss(
                    logits_proto, train_lbls, args.proto_margin
                )
            if args.proto_weight > 0:
                loss = cls_loss + args.proto_weight * prototype_loss(pe, train_lbls, proto)
            else:
                loss = cls_loss
            if args.pgr_weight > 0:
                loss = loss + args.pgr_weight * prototype_graph_regularization(proto, margin=args.pgr_margin)
            loss.backward()
            opt.step()
        dual_head.eval()
        with torch.no_grad():
            pe = pretrain_embs
            te = test_embs
            if sgfm_mod is not None:
                ctx = pe.mean(dim=0, keepdim=True)
                pe = sgfm_mod(pe, ctx)
                te = sgfm_mod(te, ctx)
            logits_linear_test = dual_head(te)
            proto_test = build_prototypes(pe, train_lbls, nb_classes)
            logits_proto_test = cosine_proto_logits(te, proto_test) / max(args.dual_tau, 1e-6)
            if gate_mlp is not None:
                logits_k, _ = hat_mix_logits(
                    logits_linear_test, logits_proto_test, proto_test, gate_mlp, homo_score
                )
            else:
                logits_k = dual_mix_logits(
                    logits_linear_test, logits_proto_test, alpha_raw, args.dual_alpha
                )
            if args.dual_calib:
                logits_k = logits_k * calib_scale + calib_bias
        logits_accum = logits_k if logits_accum is None else logits_accum + logits_k
    return logits_accum / float(K)
