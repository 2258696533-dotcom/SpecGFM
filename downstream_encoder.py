"""Shared downstream graph encoder: prompt + GSL + optional AP/HS + frozen GCN."""

from __future__ import annotations

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
from models.adaptive_prop import AdaptivePropagationPrompt
from models.hybrid_spectral_encoder import HybridSpectralFusion, fuse_band_embeddings
from models.spectral_prompt_graph import SpectralPromptGraph
from preprompt import BandGSL
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


class DownstreamMDGFMEncoder(nn.Module):
    """Per-episode encoder matching MDGFM downprompt topology path (without prototype head)."""

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
        ap: AdaptivePropagationPrompt | None = None,
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
        homo_score: torch.Tensor | None = None,
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
) -> DownstreamMDGFMEncoder:
    ap = None
    if use_ap:
        ap = AdaptivePropagationPrompt(num_hops=ap_num_hops, homo_condition=ap_homo_cond)
    enc = DownstreamMDGFMEncoder(
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
        ap=ap,
    )
    return enc


class DownstreamMDGFMHSEncoder(nn.Module):
    """MDGFM-HS: prompt + BandGSL + per-band spectral prompt graph + frozen GCN."""

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
        num_prompt: int = 10,
        tau_inner: float = 0.5,
        tau_cross: float = 0.3,
        band_fusion: HybridSpectralFusion | None = None,
        freeze_band_fusion: bool = True,
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
        self.learner = BandGSL(
            2, 50, 6, 0.5, sparse=True, act="relu",
            init_alpha=0.5, adaptive_gate=True, trust_low=0.12, gate_clamp=True,
        )
        self.struct_alpha = nn.Parameter(torch.tensor(0.5))
        self.prompt_low = SpectralPromptGraph(feature_dim, num_prompt, tau_inner, tau_cross)
        self.prompt_high = SpectralPromptGraph(feature_dim, num_prompt, tau_inner, tau_cross)
        self.band_fusion = band_fusion
        self.freeze_band_fusion = freeze_band_fusion
        if self.band_fusion is None:
            self.register_buffer("_fixed_band_weights", torch.tensor([0.5, 0.5]), persistent=False)
        elif freeze_band_fusion:
            for p in self.band_fusion.parameters():
                p.requires_grad_(False)

    def _band_weights(self, device: torch.device) -> torch.Tensor:
        if self.band_fusion is not None:
            return self.band_fusion.weights().to(device)
        return self._fixed_band_weights.to(device)

    def encode_graph(
        self,
        features: torch.Tensor,
        adj,
        sparse: bool,
        gcn: nn.Module,
        downk: int,
        homo_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del homo_score  # reserved for future homophily-conditioned band weights
        features1 = self.prefeature(features)
        reseq1 = _adj_mm(adj, features1)
        reseq1 = torch.cat((features1, reseq1), dim=1)
        reseq111 = self.balancedprompt(reseq1)
        n = reseq111.shape[0]
        k_eff = max(1, min(int(downk), n - 1))
        _, low_adj, high_adj, _, _, _, _ = self.learner.graph_process_with_bands(k_eff, reseq111)
        orig_dense = _adj_as_dense(adj)
        low_mix = self.struct_alpha * orig_dense + (1.0 - self.struct_alpha) * low_adj
        high_mix = self.struct_alpha * orig_dense + (1.0 - self.struct_alpha) * high_adj

        band_embeds = []
        for band_adj, prompt_mod in ((low_mix, self.prompt_low), (high_mix, self.prompt_high)):
            aug_feats, aug_adj = prompt_mod.build_augmented_graph(features1, band_adj)
            emb = gcn(aug_feats, aug_adj, False, None).squeeze()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            band_embeds.append(emb[: features.shape[0]])

        fused = fuse_band_embeddings(
            band_embeds,
            fusion=self.band_fusion if not self.freeze_band_fusion else None,
            fixed_weights=self._band_weights(features.device),
        )
        return fused

    def spectral_trainable_parameters(self):
        """Episode-wise params for spectral alignment (exclude frozen meta tokens)."""
        return (
            list(self.prefeature.parameters())
            + [self.struct_alpha]
            + list(self.learner.parameters())
            + list(self.prompt_low.parameters())
            + list(self.prompt_high.parameters())
        )


def build_downstream_hs_encoder_from_model(
    model,
    combinetype: str,
    unify_dim: int,
    hid_units: int,
    num_prompt: int,
    tau_inner: float,
    tau_cross: float,
    freeze_band_fusion: bool = True,
) -> DownstreamMDGFMHSEncoder:
    band_fusion = None
    if getattr(model, "use_hybrid_spectral", False) and hasattr(model, "band_fusion"):
        band_fusion = model.band_fusion
    enc = DownstreamMDGFMHSEncoder(
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
        num_prompt=num_prompt,
        tau_inner=tau_inner,
        tau_cross=tau_cross,
        band_fusion=band_fusion,
        freeze_band_fusion=freeze_band_fusion,
    )
    if getattr(model, "use_band_gsl", False) and isinstance(getattr(model, "learner", None), BandGSL):
        enc.learner.load_state_dict(model.learner.state_dict(), strict=False)
    return enc
