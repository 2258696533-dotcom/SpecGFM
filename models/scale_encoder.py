"""GCN encoder factory (paper SpecGFM uses the default GcnLayers)."""

from __future__ import annotations

from typing import Optional

from models import GcnLayers


def resolve_scale_encoder(use_scale_gnn: bool, scale_encoder: Optional[str]) -> str:
    enc = (scale_encoder or "none").strip().lower()
    if enc not in ("none", ""):
        raise ValueError(
            f"This SpecGFM release only supports the default GCN encoder, got {enc!r}"
        )
    return "none"


def build_gcn_encoder(
    n_in,
    n_h,
    num_layers_num,
    dropout,
    scale_encoder: str,
    num_hops: int = 3,
    scale_residual_gamma: float = 0.15,
):
    return GcnLayers(n_in, n_h, num_layers_num, dropout)
