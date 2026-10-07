"""Hybrid spectral fusion utilities for multi-band GCN encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSpectralFusion(nn.Module):
    """Learnable softmax weights over K spectral bands (frozen at downstream by default)."""

    def __init__(self, num_bands: int = 2, init_equal: bool = True):
        super().__init__()
        self.num_bands = max(1, int(num_bands))
        if init_equal:
            logits = torch.zeros(self.num_bands)
        else:
            logits = torch.randn(self.num_bands) * 0.01
        self.band_logits = nn.Parameter(logits)

    def weights(self) -> torch.Tensor:
        return F.softmax(self.band_logits, dim=0)

    def fuse(self, band_embeds: list[torch.Tensor]) -> torch.Tensor:
        if len(band_embeds) == 1:
            return band_embeds[0]
        w = self.weights()
        out = w[0] * band_embeds[0]
        for i in range(1, len(band_embeds)):
            out = out + w[i] * band_embeds[i]
        return out


def fuse_band_embeddings(
    band_embeds: list[torch.Tensor],
    fusion: HybridSpectralFusion | None = None,
    fixed_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse per-band node embeddings with learned or fixed band weights."""
    if len(band_embeds) == 1:
        return band_embeds[0]
    if fusion is not None:
        return fusion.fuse(band_embeds)
    if fixed_weights is not None:
        w = fixed_weights
    else:
        w = band_embeds[0].new_tensor([0.5, 0.5])
    out = w[0] * band_embeds[0]
    for i in range(1, len(band_embeds)):
        out = out + w[i] * band_embeds[i]
    return out
