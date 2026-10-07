"""Downstream graph encoder: prompt + GSL + frozen GCN (SpecGFM)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from downprompt import (
    ATT_learner,
    composedtoken,
    combineprompt,
    downstreamprompt,
    prefeatureprompt,
    textprompt,
)
from tools import (
    apply_non_linearity,
    cal_similarity_graph,
    normalize,
    symmetrize,
    top_k,
)


def _adj_as_dense(adj: torch.Tensor) -> torch.Tensor:
    return adj.to_dense() if adj.is_sparse else adj


def _adj_mm(adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    if adj.is_sparse:
        return torch.sparse.mm(adj, features)
    return torch.mm(_adj_as_dense(adj), features)


class DownstreamSpecGFMEncoder(nn.Module):
    """Per-episode encoder matching SpecGFM downprompt topology (no prototype head)."""

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
        ft_in: int,
        combinetype: str,
        feature_dim: int,
        ap=None,
    ):
        super().__init__()
        self.downprompt = downstreamprompt(ft_in)
        self.composedprompt = composedtoken(token1, token2, token3, token4, token5, type=combinetype)
        self.prefeature = prefeatureprompt(
            pretoken1, pretoken2, pretoken3, pretoken4, pretoken5,
            sumtext, dim=feature_dim, type=combinetype, head_num=4,
        )
        self.combineprompt1 = combineprompt()
        self.combineprompt2 = combineprompt()
        self.balancedprompt = textprompt(2 * feature_dim, combinetype)
        self.learner = ATT_learner(2, 50, 6, 0.5, sparse=True, act="relu")
        self.struct_alpha = nn.Parameter(torch.tensor(0.5))
        self.ap = ap
        self._tiny_graph_threshold = 48

    def _refined_adj(self, k: int, embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
        n = embeddings.shape[0]
        k_eff = max(1, min(int(k), n - 1))
        if n <= self._tiny_graph_threshold:
            emb = F.normalize(embeddings, dim=1, p=2)
            sim = cal_similarity_graph(emb)
            sim = top_k(sim, k_eff + 1)
            sim = symmetrize(sim)
            sim = apply_non_linearity(sim, self.learner.non_linearity, self.learner.i)
            learned = normalize(sim, "sym")
            return F.dropout(learned, p=self.learner.dropedge_rate, training=self.learner.training)
        return self.learner.graph_process(k_eff, embeddings).to(device)

    def encode_graph(
        self,
        features: torch.Tensor,
        adj,
        sparse: bool,
        gcn: nn.Module,
        downk: int,
        homo_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features1 = self.prefeature(features)
        reseq1 = _adj_mm(adj, features1)
        reseq1 = torch.cat((features1, reseq1), dim=1)
        reseq111 = self.balancedprompt(reseq1)
        adj1 = self._refined_adj(downk, reseq111, features.device)
        adjtot = self.struct_alpha * _adj_as_dense(adj) + (1.0 - self.struct_alpha) * adj1
        if self.ap is not None:
            adjtot = self.ap(adjtot, homo_score=homo_score)
        use_sparse_gcn = sparse and (adj.is_sparse if isinstance(adj, torch.Tensor) else False)
        embeds = gcn(features1, adjtot, use_sparse_gcn, None).squeeze()
        if embeds.dim() == 1:
            embeds = embeds.unsqueeze(0)
        return embeds


def build_downstream_encoder_from_model(
    model,
    combinetype: str,
    unify_dim: int,
    hid_units: int,
    use_ap: bool,
    ap_num_hops: int,
    ap_homo_cond: bool,
) -> DownstreamSpecGFMEncoder:
    if use_ap:
        raise ValueError("Adaptive propagation is not included in this SpecGFM release")
    return DownstreamSpecGFMEncoder(
        model.texttoken1.weight.detach(),
        model.texttoken2.weight.detach(),
        model.texttoken3.weight.detach(),
        model.texttoken4.weight.detach(),
        model.texttoken5.weight.detach(),
        model.sumtext.weight.detach(),
        model.pretext1.weight.detach(),
        model.pretext2.weight.detach(),
        model.pretext3.weight.detach(),
        model.pretext4.weight.detach(),
        model.pretext5.weight.detach(),
        model.balancetoken1.weight.detach(),
        model.balancetoken2.weight.detach(),
        model.balancetoken3.weight.detach(),
        model.balancetoken4.weight.detach(),
        model.balancetoken5.weight.detach(),
        hid_units,
        combinetype,
        unify_dim,
        ap=None,
    )
