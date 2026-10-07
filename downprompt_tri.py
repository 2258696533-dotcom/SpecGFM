"""MDGFM-Tri downstream head: original downprompt + BiKT/ProGraph/MF-GIA fusion."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from downprompt import downprompt, averageemb
from models.branch_utils import (
    BiKTDualEncoder,
    BiKTMixer,
    DomainConditionedAligner,
    GradientFingerprintEmbedder,
    KnowledgeAwareViewMixer,
    ProGraphSubspacePrompt,
    bikt_consistency_loss,
    build_adjtot,
    prograph_view_contrastive,
)


class downprompt_tri(downprompt):
    """Enhanced downstream prompt head built on original MDGFM downprompt."""

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
        tri_subspaces: int = 3,
        tri_domain_dim: int = 32,
    ):
        super().__init__(
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
        )
        self.subspace_prompt = ProGraphSubspacePrompt(feature_dim, tri_subspaces)
        self.view_mixer = KnowledgeAwareViewMixer()
        self.fingerprint = GradientFingerprintEmbedder(8, tri_domain_dim)
        self.domain_aligner = DomainConditionedAligner(ft_in, tri_domain_dim)
        self.bikt_mixer = BiKTMixer()
        self._domain_emb = None
        self._homo_score = None
        self._support_idx = None
        self._bikt_weight = 0.05
        self._view_weight = 0.05

    def set_episode_context(
        self,
        homo_score: Optional[torch.Tensor],
        bikt_weight: float,
        view_weight: float,
        support_idx: torch.Tensor,
    ) -> None:
        self._homo_score = homo_score
        self._bikt_weight = bikt_weight
        self._view_weight = view_weight
        self._support_idx = support_idx

    def refresh_domain_embedding(
        self,
        features,
        adj,
        sparse,
        gcn,
        downk,
        idx_train,
        train_lbls,
    ) -> None:
        with torch.no_grad():
            features1 = self.prefeature(features)
            features1 = self.subspace_prompt(features1, self._homo_score)
            adjtot, _ = build_adjtot(self, features1, adj, sparse, downk)
            stats = self.fingerprint.collect_stats(
                gcn, features1, adjtot, sparse, idx_train, train_lbls, self.nb_classes
            )
            self._domain_emb = self.fingerprint(stats).detach()

    def forward(
        self,
        features,
        adj,
        sparse,
        gcn,
        idx,
        seq,
        downk,
        labels=None,
        train=0,
        return_aux=False,
    ):
        features1 = self.prefeature(features)
        features1 = self.subspace_prompt(features1, self._homo_score)
        adjtot, adj_dense = build_adjtot(self, features1, adj, sparse, downk)

        z_gnn, z_mlp = BiKTDualEncoder.encode(gcn, features1, adjtot, sparse, lp=False)
        embeds1 = self.bikt_mixer(z_gnn, z_mlp, self._homo_score)
        if self._domain_emb is not None:
            embeds1 = self.domain_aligner(embeds1, self._domain_emb, self._homo_score)

        pretrain_embs1 = embeds1[idx]
        rawret = pretrain_embs1.cuda()

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret, nb_class=self.nb_classes)

        rawret_cat = torch.cat((rawret, self.ave), dim=0)
        sim = torch.cosine_similarity(rawret_cat.unsqueeze(1), rawret_cat.unsqueeze(0), dim=-1)
        ret = F.softmax(sim[: seq.shape[0], seq.shape[0] :], dim=1)

        if return_aux:
            support_idx = self._support_idx if self._support_idx is not None else idx
            _, low_view, high_view = self.view_mixer(
                features1, adj_dense, self._homo_score
            )
            z_low = gcn(low_view, adjtot, sparse, None).squeeze()
            z_high = gcn(high_view, adjtot, sparse, None).squeeze()
            aux = {
                "train_embs": pretrain_embs1,
                "proto": self.ave,
                "bikt_loss": bikt_consistency_loss(
                    z_gnn, z_mlp, idx=support_idx, weight=self._bikt_weight,
                    homo_score=self._homo_score,
                ),
                "view_loss": prograph_view_contrastive(
                    z_low, z_high, idx=support_idx, weight=self._view_weight,
                    homo_score=self._homo_score,
                ),
            }
            return ret, aux
        return ret
