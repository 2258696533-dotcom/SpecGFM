"""Scale encoder selection helpers (none | scale1 | scale2)."""

from __future__ import annotations

from models import GcnLayers
from models.scale_gcn_layers import ScaleGcnLayers
from models.scale_gcn_layers_v2 import ScaleGcnLayersV2
from models.scale_gcn_layers_residual import ScaleResidualGcnLayers
from models.gpr_gcn_layers_residual import GprResidualGcnLayers

VALID_SCALE_ENCODERS = ("none", "scale1", "scale2", "scale_residual", "gpr_residual")


def resolve_scale_encoder(use_scale_gnn: bool, scale_encoder: str | None) -> str:
    enc = (scale_encoder or "none").strip().lower()
    if enc not in VALID_SCALE_ENCODERS:
        raise ValueError(f"scale_encoder must be one of {VALID_SCALE_ENCODERS}, got {enc!r}")
    if enc != "none":
        return enc
    if use_scale_gnn:
        return "scale1"
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
    enc = (scale_encoder or "none").strip().lower()
    if enc == "scale1":
        return ScaleGcnLayers(n_in, n_h, num_layers_num, dropout, num_hops=num_hops)
    if enc == "scale2":
        return ScaleGcnLayersV2(n_in, n_h, num_layers_num, dropout, num_hops=num_hops)
    if enc == "scale_residual":
        return ScaleResidualGcnLayers(
            n_in,
            n_h,
            num_layers_num,
            dropout,
            num_hops=num_hops,
            scale_residual_gamma=scale_residual_gamma,
        )
    if enc == "gpr_residual":
        return GprResidualGcnLayers(
            n_in,
            n_h,
            num_layers_num,
            dropout,
            num_hops=num_hops,
            scale_residual_gamma=scale_residual_gamma,
        )
    return GcnLayers(n_in, n_h, num_layers_num, dropout)
