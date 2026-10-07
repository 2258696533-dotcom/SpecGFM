"""MTG setup helpers: hetero-only gating aligned with homo router / BiKT."""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn

from models.message_tuning import mtg_trainable_params, wrap_gcn_with_mtg


def mtg_hetero_only_enabled(args) -> bool:
    if not getattr(args, "use_mtg", False):
        return False
    if hasattr(args, "mtg_hetero_only") and not args.mtg_hetero_only:
        return False
    return bool(getattr(args, "use_homo_router", False))


def dual_uses_reencode(args) -> bool:
    """Dual re-encode for dual_adapted, or global MTG (legacy)."""
    if getattr(args, "dual_adapted", False):
        return True
    if getattr(args, "use_mtg", False) and not mtg_hetero_only_enabled(args):
        return True
    return False


def apply_mtg_if_needed(model, args) -> None:
    """Attach MTG: hetero-only keeps base gcn for Dual, mtg gcn for BiKT/dp."""
    if not getattr(args, "use_mtg", False):
        return
    n_proto = int(getattr(args, "mtg_prototypes", 4))
    device = next(model.gcn.parameters()).device
    if mtg_hetero_only_enabled(args):
        model.gcn_mtg = wrap_gcn_with_mtg(model.gcn, num_prototypes=n_proto)
        model.gcn_mtg = model.gcn_mtg.to(device)
        print(
            f"[MTG] hetero-only: Dual->base GCN, BiKT/dp->MTG m={n_proto} "
            f"x {getattr(model.gcn_mtg, 'num_layers_num', '?')} layers",
            flush=True,
        )
        return
    model.gcn = wrap_gcn_with_mtg(model.gcn, num_prototypes=n_proto)
    model.gcn = model.gcn.to(device)
    n_layers = getattr(model.gcn, "num_layers_num", "?")
    print(
        f"[MTG] wrapped frozen GCN (global): m={n_proto} prototypes x {n_layers} layers",
        flush=True,
    )


def gcn_for_downstream(model, use_dual_branch: bool) -> nn.Module:
    """Pick GCN for current episode branch."""
    if use_dual_branch:
        return model.gcn
    if mtg_attached(model):
        return getattr(model, "gcn_mtg", model.gcn)
    return model.gcn


def mtg_attached(model) -> bool:
    return hasattr(model, "gcn_mtg") and model.gcn_mtg is not None


def append_mtg_optimizer_group(optim_groups: list, model, args, lr: float) -> None:
    if not getattr(args, "use_mtg", False):
        return
    gcn = getattr(model, "gcn_mtg", model.gcn) if mtg_hetero_only_enabled(args) else model.gcn
    params = mtg_trainable_params(gcn)
    if params:
        optim_groups.append({"params": params, "lr": lr})
