"""Helpers to pick the GCN used in a downstream episode."""

from __future__ import annotations

import torch.nn as nn


def dual_uses_reencode(args) -> bool:
    return bool(getattr(args, "dual_adapted", False))


def gcn_for_downstream(model, use_dual_branch: bool) -> nn.Module:
    return model.gcn


def append_mtg_optimizer_group(optim_groups: list, model, args, lr: float) -> None:
    return


def apply_mtg_if_needed(model, args) -> None:
    return


def mtg_trainable_params(gcn) -> list:
    return []


def mtg_hetero_only_enabled(args) -> bool:
    return False
