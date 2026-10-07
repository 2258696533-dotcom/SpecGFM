"""MF-GIA downstream route: heterophily-gated domain FiLM on embeddings."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from downprompt import downprompt, averageemb
from models.mdgfm_tri import (
    DomainConditionedAligner,
    GradientFingerprintEmbedder,
    build_adjtot,
)


class downprompt_mfgia(downprompt):
    def __init__(
        self,
        token1, token2, token3, token4, token5, sumtext,
        pretoken1, pretoken2, pretoken3, pretoken4, pretoken5,
        balancetoken1, balancetoken2, balancetoken3, balancetoken4, balancetoken5,
        ft_in, nb_classes, type, feature_dim,
        mfgia_domain_dim: int = 32,
    ):
        super().__init__(
            token1, token2, token3, token4, token5, sumtext,
            pretoken1, pretoken2, pretoken3, pretoken4, pretoken5,
            balancetoken1, balancetoken2, balancetoken3, balancetoken4, balancetoken5,
            ft_in, nb_classes, type, feature_dim,
        )
        self.fingerprint = GradientFingerprintEmbedder(8, mfgia_domain_dim)
        self.domain_aligner = DomainConditionedAligner(ft_in, mfgia_domain_dim)
        self._domain_emb = None
        self._homo_score = None
        self._support_idx = None

    def set_episode_context(
        self,
        support_idx: torch.Tensor,
        homo_score: Optional[torch.Tensor] = None,
    ) -> None:
        self._support_idx = support_idx
        self._homo_score = homo_score

    def refresh_domain_embedding(
        self, features, adj, sparse, gcn, downk, idx_train, train_lbls,
    ) -> None:
        with torch.no_grad():
            features1 = self.prefeature(features)
            adjtot, _ = build_adjtot(self, features1, adj, sparse, downk)
            stats = self.fingerprint.collect_stats(
                gcn, features1, adjtot, sparse, idx_train, train_lbls, self.nb_classes
            )
            self._domain_emb = self.fingerprint(stats).detach()

    def forward(
        self, features, adj, sparse, gcn, idx, seq, downk,
        labels=None, train=0, return_aux=False,
    ):
        features1 = self.prefeature(features)
        adjtot, _ = build_adjtot(self, features1, adj, sparse, downk)
        embeds1 = gcn(features1, adjtot, sparse, None).squeeze()
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
            aux = {"train_embs": pretrain_embs1, "proto": self.ave}
            return ret, aux
        return ret
