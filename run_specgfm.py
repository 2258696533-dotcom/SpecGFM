#!/usr/bin/env python3
"""Launch the SpecGFM main setting and the four RQ2 variants."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional

DATASETS = ("Cora", "Citeseer", "Pubmed", "Cornell", "Chameleon", "Squirrel")
# Per-target settings used for the paper tables. Episodes are 50 on every target.
DATASET_CONFIGS = {
    "Cora": {"lr": "0.0075", "downstreamlr": "0.001", "epochs": "60"},
    "Citeseer": {"lr": "0.001", "downstreamlr": "0.001", "epochs": "60"},
    "Pubmed": {"lr": "0.0001", "downstreamlr": "0.0014", "epochs": "60"},
    "Cornell": {"lr": "0.02", "downstreamlr": "0.0003", "epochs": "100"},
    "Chameleon": {"lr": "0.02", "downstreamlr": "0.02", "epochs": "100"},
    "Squirrel": {"lr": "0.01", "downstreamlr": "0.0003", "epochs": "100"},
}


def _dual_downstream() -> List[str]:
    return [
        "--dual_alpha", "0.6",
        "--dual_epochs", "300",
        "--proto_weight", "0.05",
        "--dual_ensemble", "3",
        "--dpc_weight", "0.01",
        "--dpc_temp", "0.2",
    ]


def _band_pretrain() -> List[str]:
    return [
        "--use_band_gsl",
        "--band_init_alpha", "0.5",
        "--band_v2_adaptive_gate",
        "--band_v2_cross_domain",
        "--band_trust_low", "0.12",
        "--band_gate_clamp",
    ]


def _tail_pretrain() -> List[str]:
    return [
        "--use_feat_dv_inv",
        "--feat_dv_weight", "0.15",
        "--feat_dv_dropout", "0.25",
        "--pretrain_ema",
        "--pretrain_ema_decay", "0.999",
    ]


def _homo_router(extra: Optional[List[str]] = None) -> List[str]:
    cmd = ["--use_homo_router", "--homo_bypass_thresh", "0.52"]
    if extra:
        cmd.extend(extra)
    return cmd


def _specgfm_stack() -> List[str]:
    """Full SpecGFM: BandGSL, coordinate alignment, dual-branch routing, support-GEE."""
    return (
        _dual_downstream()
        + _band_pretrain()
        + _tail_pretrain()
        + ["--scgw_p4"]
        + _homo_router(["--use_bikt", "--f2_gee_branch", "dual", "--result_tag", "specgfm"])
    )


def _rq2_wo_gee() -> List[str]:
    return (
        _dual_downstream()
        + _band_pretrain()
        + _tail_pretrain()
        + ["--scgw_p4"]
        + _homo_router(["--use_bikt", "--result_tag", "rq2_wo_gee"])
    )


def _rq2_wo_band() -> List[str]:
    return (
        _dual_downstream()
        + _tail_pretrain()
        + _homo_router(["--use_bikt", "--f2_gee_branch", "dual", "--result_tag", "rq2_wo_band"])
    )


def _rq2_wo_he() -> List[str]:
    """w/o-He: always the homophilic branch."""
    return (
        _dual_downstream()
        + _band_pretrain()
        + _tail_pretrain()
        + ["--scgw_p4", "--f2_gee_branch", "dual", "--result_tag", "rq2_wo_he"]
    )


def _rq2_wo_ho() -> List[str]:
    """w/o-Ho: always the heterophilic branch (tau = 1)."""
    return (
        _dual_downstream()
        + _band_pretrain()
        + _tail_pretrain()
        + [
            "--scgw_p4",
            "--use_homo_router",
            "--homo_bypass_thresh", "1.0",
            "--use_bikt",
            "--f2_gee_branch", "dual",
            "--result_tag", "rq2_wo_ho",
        ]
    )


MODES = {
    "specgfm": _specgfm_stack,
    "rq2_wo_band": _rq2_wo_band,
    "rq2_wo_gee": _rq2_wo_gee,
    "rq2_wo_he": _rq2_wo_he,
    "rq2_wo_ho": _rq2_wo_ho,
}


def build_cmd(dataset: str, seed: int, shot_num: int, mode: str, extra: List[str]) -> List[str]:
    py = os.environ.get("SPECGFM_PYTHON", sys.executable)
    cfg = DATASET_CONFIGS[dataset]
    cmd = [
        py, "-u", "SpecGFM.py",
        "--dataset", dataset,
        "--seed", str(seed),
        "--shot_num", str(shot_num),
        "--epochs", cfg["epochs"],
        "--eval_episodes", "50",
        "--lr", cfg["lr"],
        "--downstreamlr", cfg["downstreamlr"],
    ]
    cmd.extend(MODES[mode]())
    cmd.extend(extra)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpecGFM / RQ2 ablations")
    parser.add_argument("--dataset", default="Cora", choices=DATASETS)
    parser.add_argument("--mode", default="specgfm", choices=sorted(MODES.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1024])
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags after --")
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else list(args.extra)

    for seed in args.seeds:
        cmd = build_cmd(args.dataset, seed, args.shot_num, args.mode, extra)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
