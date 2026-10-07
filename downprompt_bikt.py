"""Heterophilic branch: frozen GCN view mixed with an MLP view on the learned adjacency."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from downprompt import downprompt, averageemb
from models.branch_utils import (
    BiKTDualEncoder,
    BiKTMixer,
    bikt_consistency_loss,
    build_adjtot,
)


class downprompt_bikt(downprompt):
    def __init__(
        self,
        token1,
        token2,
        token3,
        token4,
        token5,
        sumtext,
        pretoken1,
        pretoken2,
        pretoken3,
        pretoken4,
        pretoken5,
        balancetoken1,
        balancetoken2,
        balancetoken3,
        balancetoken4,
        balancetoken5,
        ft_in,
        nb_classes,
        type,
        feature_dim,
        bikt_weight: float = 0.05,
    ):
        super().__init__(
            token1, token2, token3, token4, token5, sumtext,
            pretoken1, pretoken2, pretoken3, pretoken4, pretoken5,
            balancetoken1, balancetoken2, balancetoken3, balancetoken4, balancetoken5,
            ft_in, nb_classes, type, feature_dim,
        )
        self.bikt_mixer = BiKTMixer(max_residual=0.55)
        self._homo_score = None
        self._support_idx = None
        self._bikt_weight = bikt_weight
        self._f2_plugin_args = None
        self._f2_adj_dense = None
        self._f2_support_labels = None

    def set_episode_context(
        self,
        homo_score: Optional[torch.Tensor],
        support_idx: torch.Tensor,
        bikt_weight: Optional[float] = None,
    ) -> None:
        self._homo_score = homo_score
        self._support_idx = support_idx
        if bikt_weight is not None:
            self._bikt_weight = bikt_weight

    def set_f2_plugin_graph(self, args, adj_dense, support_labels=None) -> None:
        """Optional F2 E2/E3 plugins on the hetero BiKT path."""
        self._f2_plugin_args = args
        self._f2_adj_dense = adj_dense
        self._f2_support_labels = support_labels

    def forward(
        self, features, adj, sparse, gcn, idx, seq, downk,
        labels=None, train=0, return_aux=False,
    ):
        features1 = self.prefeature(features)
        adjtot, _ = build_adjtot(self, features1, adj, sparse, downk)

        z_gnn, z_mlp = BiKTDualEncoder.encode(gcn, features1, adjtot, sparse, lp=False)
        embeds1 = self.bikt_mixer(z_gnn, z_mlp, self._homo_score)
        if self._f2_plugin_args is not None and self._f2_adj_dense is not None:
            from models.f2_downstream_plugins import apply_hetero_embed_plugins
            adj_d = self._f2_adj_dense
            if isinstance(adjtot, torch.Tensor) and adjtot.is_sparse:
                adj_d = adjtot.to_dense()
            elif isinstance(adjtot, torch.Tensor) and adjtot.dim() == 2:
                adj_d = adjtot
            support_idx = self._support_idx if self._support_idx is not None else idx
            labels_all = torch.zeros(
                embeds1.shape[0], device=embeds1.device, dtype=torch.long
            )
            sup_lbl = self._f2_support_labels
            if sup_lbl is None and labels is not None and labels.numel() == support_idx.numel():
                sup_lbl = labels
            if sup_lbl is not None:
                labels_all[support_idx] = sup_lbl.reshape(-1).long().to(embeds1.device)
            embeds1 = apply_hetero_embed_plugins(
                embeds1, adj_d, features1, labels_all, support_idx,
                self._f2_plugin_args, self._homo_score,
            )
        pretrain_embs1 = embeds1[idx]
        rawret = pretrain_embs1.cuda()

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret, nb_class=self.nb_classes)
            # R1: weighted teacher prototypes on BiKT (GCRD-style).
            if self._f2_plugin_args is not None and self._f2_adj_dense is not None:
                from models.f2_downstream_plugins import apply_bikt_teacher_prototypes
                support_idx = self._support_idx if self._support_idx is not None else idx
                labels_all = torch.zeros(
                    embeds1.shape[0], device=embeds1.device, dtype=torch.long
                )
                if labels is not None:
                    labels_all[support_idx] = labels.reshape(-1).long().to(embeds1.device)
                self.ave = apply_bikt_teacher_prototypes(
                    self.ave, rawret, labels, self._f2_adj_dense, features1,
                    support_idx, labels_all, self._f2_plugin_args, self.nb_classes,
                )

        rawret_cat = torch.cat((rawret, self.ave), dim=0)
        sim = torch.cosine_similarity(rawret_cat.unsqueeze(1), rawret_cat.unsqueeze(0), dim=-1)
        sim_slice = sim[: seq.shape[0], seq.shape[0] :]
        ret = F.softmax(sim_slice, dim=1)

        if return_aux:
            support_idx = self._support_idx if self._support_idx is not None else idx
            aux = {
                "train_embs": pretrain_embs1,
                "proto": self.ave,
                "bikt_loss": bikt_consistency_loss(
                    z_gnn, z_mlp, idx=support_idx,
                    weight=self._bikt_weight, homo_score=self._homo_score,
                ),
            }
            if self._f2_plugin_args is not None and labels is not None:
                from models.f2_downstream_plugins import bikt_plugin_aux_loss
                aux["f2_plugin_loss"] = bikt_plugin_aux_loss(
                    self._f2_plugin_args, pretrain_embs1, labels,
                    self._f2_adj_dense, support_idx,
                )
            return ret, aux
        return ret
