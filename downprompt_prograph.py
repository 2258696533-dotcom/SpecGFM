"""ProGraph downstream route: heterophily-gated prompts + view contrast."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from downprompt import downprompt, averageemb
from models.mdgfm_tri import (
    KnowledgeAwareViewMixer,
    ProGraphSubspacePrompt,
    build_adjtot,
    homophilic_episode,
    prograph_view_contrastive,
)


class downprompt_prograph(downprompt):
    def __init__(
        self,
        token1, token2, token3, token4, token5, sumtext,
        pretoken1, pretoken2, pretoken3, pretoken4, pretoken5,
        balancetoken1, balancetoken2, balancetoken3, balancetoken4, balancetoken5,
        ft_in, nb_classes, type, feature_dim,
        prograph_subspaces: int = 3,
        view_weight: float = 0.05,
    ):
        super().__init__(
            token1, token2, token3, token4, token5, sumtext,
            pretoken1, pretoken2, pretoken3, pretoken4, pretoken5,
            balancetoken1, balancetoken2, balancetoken3, balancetoken4, balancetoken5,
            ft_in, nb_classes, type, feature_dim,
        )
        self.subspace_prompt = ProGraphSubspacePrompt(feature_dim, prograph_subspaces)
        self.view_mixer = KnowledgeAwareViewMixer()
        self._homo_score = None
        self._support_idx = None
        self._view_weight = view_weight

    def set_episode_context(
        self,
        homo_score: Optional[torch.Tensor],
        support_idx: torch.Tensor,
        view_weight: Optional[float] = None,
    ) -> None:
        self._homo_score = homo_score
        self._support_idx = support_idx
        if view_weight is not None:
            self._view_weight = view_weight

    def forward(
        self, features, adj, sparse, gcn, idx, seq, downk,
        labels=None, train=0, return_aux=False,
    ):
        features1 = self.prefeature(features)
        features1 = self.subspace_prompt(features1, self._homo_score)
        adjtot, adj_dense = build_adjtot(self, features1, adj, sparse, downk)

        if not homophilic_episode(self._homo_score):
            mixed, _, _ = self.view_mixer(features1, adj_dense, self._homo_score)
            hetero = 1.0 - float(self._homo_score.reshape(())) if self._homo_score is not None else 0.5
            features1 = features1 + 0.12 * hetero * (mixed - features1)
            adjtot, adj_dense = build_adjtot(self, features1, adj, sparse, downk)

        embeds_main = gcn(features1, adjtot, sparse, None).squeeze()
        pretrain_embs1 = embeds_main[idx]
        rawret = pretrain_embs1.cuda()

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret, nb_class=self.nb_classes)

        rawret_cat = torch.cat((rawret, self.ave), dim=0)
        sim = torch.cosine_similarity(rawret_cat.unsqueeze(1), rawret_cat.unsqueeze(0), dim=-1)
        ret = F.softmax(sim[: seq.shape[0], seq.shape[0] :], dim=1)

        if return_aux:
            support_idx = self._support_idx if self._support_idx is not None else idx
            _, low_view, high_view = self.view_mixer(features1, adj_dense, self._homo_score)
            z_low = gcn(low_view, adjtot, sparse, None).squeeze()
            z_high = gcn(high_view, adjtot, sparse, None).squeeze()
            aux = {
                "train_embs": pretrain_embs1,
                "proto": self.ave,
                "view_loss": prograph_view_contrastive(
                    z_low, z_high, idx=support_idx,
                    weight=self._view_weight, homo_score=self._homo_score,
                ),
            }
            return ret, aux
        return ret
