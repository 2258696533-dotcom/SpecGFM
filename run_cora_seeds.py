"""Sequential runner for Cora experiments across multiple seeds.

Usage examples:
  # Original MDGFM (downprompt), default 5 seeds
  python run_cora_seeds.py --mode original

  # 5-shot single seed (CSV gets *_5shot_* tag when shot_num!=1)
  python run_cora_seeds.py --mode original --dataset Cora --seeds 1024 --shot_num 5
  python run_cora_seeds.py --mode f2p4_r4_gee --dataset Cora --seeds 1024 --shot_num 5
  # Full 6-domain suite: bash run_5shot_mdgfm_specgfm_6dom.sh

  # Original MDGFM pretrain + Dual head (GRAVER ablation baseline)
  python run_cora_seeds.py --mode original_dual --dataset Cora --seeds 512 1024 2048

  # Original MDGFM pretrain + GRAVER + Dual head
  python run_cora_seeds.py --mode original_graver --dataset Cornell --seeds 512 1024 2048

  # Paper routes on original MDGFM pretrain + downprompt (isolated ablations)
  python run_cora_seeds.py --mode original_bikt --dataset Cora --seeds 512
  python run_cora_seeds.py --mode original_prograph --dataset Cornell --seeds 512
  python run_cora_seeds.py --mode original_mfgia --dataset Cora --seeds 512
  python run_cora_seeds.py --mode original_tri --dataset Cora --seeds 512  # fusion after ablations

  # SCGFM ablations (original MDGFM pretrain + downprompt)
  python run_cora_seeds.py --mode original_scgw_p1 --seeds 1024
  python run_cora_seeds.py --mode original_scgw_p2 --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode original_scgw_p3 --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode original_scgw_p4 --seeds 512 1024 2048 4096 8192

  # SCGW P2 five-seed screen (4 domains; skips seeds already in logs)
  python run_cora_seeds.py --mode original_scgw_p2 --dataset Cora --seeds 512 1024 2048 4096 8192

  # Full baseline (dual+dpc+K=3), custom seeds
  python run_cora_seeds.py --mode full --seeds 1024 2048 4096

  # Full + BandGSL V2
  python run_cora_seeds.py --mode full_bandv2 --seeds 512 1024 2048 4096 8192

  # Full + BandGSL V2 + V3 stability (trust-region + gate clamp)
  python run_cora_seeds.py --mode full_bandv3 --seeds 512 1024 2048 4096 8192

  # V3「更强拉回低频」：trust_low=0.2，跨域项减弱 cd_weight=0.05（专治尾部分数/8192 类种子）
  python run_cora_seeds.py --mode full_bandv3_strong --seeds 512 1024 2048 4096 8192

  # V4：在 v3_strong 上增加预训练度轮廓锚定 + gate 熵（不改 50 episodes）
  python run_cora_seeds.py --mode full_bandv4 --seeds 512 1024 2048 4096 8192

  # V5：在「最佳均值」v3 上加*弱*锚定/熵（deg=0.03, ent=0.01），折中 v3 与过强 v4
  python run_cora_seeds.py --mode full_bandv5 --seeds 512 1024 2048 4096 8192

  # BandGSL v3 + 预训练 Graph-MAE（特征掩码重建，大改收益主路径）
  python run_cora_seeds.py --mode full_bandv3_mae --seeds 512 1024 2048 4096 8192

  # 尾部友好：v3 + 特征双视图不变性 + 预训练 EMA 权重（阶段末覆盖 ckpt）
  python run_cora_seeds.py --mode full_bandv3_tail --seeds 512 1024 2048 4096 8192

  # 式 (2) 大改：v3 + FiLM（节点级 gamma/beta 替代共享 sumtext）
  python run_cora_seeds.py --mode full_bandv3_film --seeds 512 1024 2048 4096 8192

  # 推荐：v3 + sumtext + 轻量 FiLM 残差（scale=0.15）
  python run_cora_seeds.py --mode full_bandv3_film_res --seeds 512 1024 2048 4096 8192

  # FiLM 残差 + tail（dv + EMA）
  python run_cora_seeds.py --mode full_bandv3_film_res_tail --seeds 512 1024 2048 4096 8192

  # ★ 当前最佳：tail 预训练 + ScaleGNN + Dual 下游（消融确认优于 combo）
  python run_cora_seeds.py --mode full_bandv3_tail_scale_gnn --seeds 512 1024 2048 4096 8192
  # Citeseer：加 --dataset Citeseer

  # 历史 combo / GCIL 消融（复现用，非主方法）
  python run_cora_seeds.py --mode full_bandv3_tail_gcil_combo --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode full_bandv3_tail_gcil --seeds 512 1024 2048 4096 8192

  # FiLM 残差 scale 消融
  python run_cora_seeds.py --mode full_bandv3_film_res_s010 --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode full_bandv3_film_res_s020 --seeds 512 1024 2048 4096 8192

  # GCIL / ScaleGNN 预训练对照（PCA 特征 + 默认 downprompt，60 epoch）
  python run_cora_seeds.py --mode pca_gcil --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode pca_scale_gnn --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode pca_gcil_scale_combo --seeds 512 1024 2048 4096 8192

  # Band / tail 预训练分解（原始栈 + downprompt）：Cora 或 Cornell
  python run_cora_seeds.py --dataset Cora --mode band_downprompt --seeds 512 1024 2048
  python run_cora_seeds.py --dataset Cora --mode original_tail --seeds 512 1024 2048
  python run_cora_seeds.py --dataset Cora --mode band_tail_downprompt --seeds 512 1024 2048
  python run_cora_seeds.py --dataset Cornell --mode band_downprompt --seeds 512 1024 2048
  python run_cora_seeds.py --dataset Cornell --mode original_tail --seeds 512 1024 2048
  python run_cora_seeds.py --dataset Cornell --mode band_tail_downprompt --seeds 512 1024 2048

  # Unified Band+tail pretrain + per-episode homophily router (Dual vs downprompt)
  python run_cora_seeds.py --mode full_bandv3_tail_homo_router --dataset Cora --seeds 512 1024 2048
  python run_cora_seeds.py --mode full_bandv3_tail_homo_router_bikt --dataset Cornell --seeds 512 1024 2048
  # F2 + E1/E2/E3 single-seed screen (see docs/F2_E123_BASELINE.md)
  python run_cora_seeds.py --mode f2_e1_gee --dataset Citeseer --seeds 1024
  python run_cora_seeds.py --mode f2_e2_node_w --dataset Chameleon --seeds 1024
  python run_cora_seeds.py --mode f2_e3_leaky --dataset Chameleon --seeds 1024

  # Band融合2 + SCGW P4 only (unified pretrain; same downstream as fusion2)
  python run_cora_seeds.py --mode band_fusion2_scgw_p4 --dataset Citeseer --seeds 512 1024 2048

  # Paper RQ2 ablation (1-shot; suite: bash run_rq2_ablation_6dom.sh)
  #   rq2_full / rq2_wo_gee / rq2_wo_band / rq2_wo_He / rq2_wo_Ho
  #   Naming: w/o-X = remove component X
  #     rq2_wo_he -> w/o-He (drop heterophilic branch; always homophilic)
  #     rq2_wo_ho -> w/o-Ho (drop homophilic branch; always heterophilic)
  #   (rq2_wo_router is a deprecated alias of rq2_wo_he)
  python run_cora_seeds.py --mode rq2_wo_band --dataset Cora --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode rq2_wo_he --dataset Cornell --seeds 512 1024 2048 4096 8192
  python run_cora_seeds.py --mode rq2_wo_ho --dataset Cornell --seeds 512 1024 2048 4096 8192

  # Band融合2 + P4 pretrain + P3 structural prompt on BiKT/downprompt (Chameleon 优先)
  python run_cora_seeds.py --mode band_fusion2_scgw_p34 --dataset Chameleon --seeds 512 1024 2048

  # F2 + dataset-scope homo router (mean homo over 50 ep -> all Dual or all BiKT)
  python run_cora_seeds.py --mode full_bandv3_tail_homo_router_bikt_dataset --dataset Citeseer --seeds 512 1024 2048
  python run_cora_seeds.py --mode full_bandv3_tail_homo_router_bikt_dataset --dataset Chameleon --seeds 512 1024 2048

  # P34 + H1 + dataset-scope router (Chameleon: prescan mean_homo -> all BiKT + struct_boost)
  python run_cora_seeds.py --mode band_fusion2_scgw_p34_hetero_struct_dataset --dataset Chameleon --seeds 512 1024 2048
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from typing import List, Optional


def _python_for_mdgfm() -> str:
    """Prefer explicit env, then interpreter if already in `mdgfm`, else common conda path."""
    override = os.environ.get("MDGFM_PYTHON", "").strip()
    if override and os.path.isfile(override):
        return override
    exe = os.path.realpath(sys.executable)
    if f"{os.sep}envs{os.sep}mdgfm{os.sep}" in exe:
        return sys.executable
    fallback = "/root/miniconda3/envs/mdgfm/bin/python"
    if os.path.isfile(fallback):
        return fallback
    return sys.executable


DEFAULT_SEEDS = [512, 1024, 2048, 4096, 8192]

# README Table 1/2 hyperparameters per target domain (1-shot uses same lr/epochs as listed).
DATASET_CONFIGS = {
    "Cora": {
        "lr": "0.0075",
        "downstreamlr": "0.001",
        "epochs": "60",
        "drop_percent": "0.5",
    },
    "Citeseer": {
        "lr": "0.001",
        "downstreamlr": "0.001",
        "epochs": "60",
        "drop_percent": "0.5",
    },
    "Pubmed": {
        "lr": "0.0001",
        "downstreamlr": "0.0014",
        "epochs": "60",
        "drop_percent": "0.5",
    },
    "Cornell": {
        "lr": "0.02",
        "downstreamlr": "0.0003",
        "epochs": "100",
        "drop_percent": "0.5",
    },
    "Chameleon": {
        "lr": "0.02",
        "downstreamlr": "0.02",
        "epochs": "100",
        "drop_percent": "0.5",
    },
    "Squirrel": {
        "lr": "0.01",
        "downstreamlr": "0.0003",
        "epochs": "100",
        "drop_percent": "0.5",
    },
}


def build_base_cmd(
    seed: int, dataset: str = "Cora", shot_num: int = 1
) -> List[str]:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[dataset]
    return [
        _python_for_mdgfm(),
        "runexp.py",
        "--dataset", dataset,
        "--drop_percent", cfg["drop_percent"],
        "--lr", cfg["lr"],
        "--downstreamlr", cfg["downstreamlr"],
        "--epochs", cfg["epochs"],
        "--shot_num", str(int(shot_num)),
        "--seed", str(seed),
        "--feature_adapter", "pca",
    ]


def apply_shot_result_tag(cmd: List[str], shot_num: int) -> List[str]:
    """Keep 5-shot (etc.) CSV rows out of 1-shot files via result_tag suffix."""
    if int(shot_num) == 1:
        return cmd
    suffix = f"{int(shot_num)}shot"
    out = list(cmd)
    if "--result_tag" in out:
        i = out.index("--result_tag")
        out[i + 1] = f"{out[i + 1]}_{suffix}"
    else:
        out.extend(["--result_tag", suffix])
    return out


def _set_flag_value(cmd: List[str], flag: str, value: str) -> List[str]:
    out = list(cmd)
    if flag in out:
        i = out.index(flag)
        if i + 1 < len(out):
            out[i + 1] = value
        else:
            out.append(value)
    else:
        out.extend([flag, value])
    return out


def apply_homo_bypass_thresh(cmd: List[str], thresh: float) -> List[str]:
    """Override router threshold and suffix result_tag for RQ3 tau sweeps."""
    t = float(thresh)
    out = _set_flag_value(cmd, "--homo_bypass_thresh", str(t))
    tag_suffix = f"tau{int(round(t * 100)):03d}"
    if "--result_tag" in out:
        i = out.index("--result_tag")
        base = out[i + 1]
        if f"_{tag_suffix}" not in base:
            out[i + 1] = f"{base}_{tag_suffix}"
    else:
        out.extend(["--result_tag", tag_suffix])
    return out


def _band_pretrain_only() -> List[str]:
    """BandGSL v3 pretrain flags without Dual downstream (keeps downprompt)."""
    return [
        "--use_band_gsl",
        "--band_init_alpha", "0.5",
        "--band_v2_adaptive_gate",
        "--band_v2_cross_domain",
        "--band_trust_low", "0.12",
        "--band_gate_clamp",
    ]


def _band_strong_pretrain_only() -> List[str]:
    """Weaker cross-domain / stronger low-freq trust (hetero screen)."""
    return [
        "--use_band_gsl",
        "--band_init_alpha", "0.5",
        "--band_v2_adaptive_gate",
        "--band_v2_cross_domain",
        "--band_trust_low", "0.2",
        "--band_gate_clamp",
        "--band_cd_weight", "0.05",
    ]


def _band_pretrain_no_trust() -> List[str]:
    """Band v3 without low-freq anchor (trust_low=0)."""
    return [
        "--use_band_gsl",
        "--band_init_alpha", "0.5",
        "--band_v2_adaptive_gate",
        "--band_v2_cross_domain",
        "--band_trust_low", "0.0",
        "--band_gate_clamp",
    ]


def _band_pretrain_no_cd() -> List[str]:
    """Band v3 without cross-domain low/high prototype alignment."""
    return _band_pretrain_only() + ["--band_cd_weight", "0.0"]


def _scgw_p1_args() -> List[str]:
    return ["--scgw_p1", "--result_tag", "scgw_p1"]


def _scgw_p2_args() -> List[str]:
    return ["--scgw_p2", "--result_tag", "scgw_p2"]


def _scgw_p2_bikt_args() -> List[str]:
    return ["--scgw_p2", "--use_bikt", "--result_tag", "scgw_p2_bikt"]


def _scgw_p2_fusion2_args() -> List[str]:
    """P2 pretrain + Route A fusion2 downstream (homo router + BiKT)."""
    return ["--scgw_p2"] + _dual_downstream_only() + _homo_router_args(
        ["--use_bikt", "--result_tag", "scgw_p2_homo_router_bikt"]
    )


def _scgw_p3_args() -> List[str]:
    # P2 trains geometric bases during pretrain; P3 applies downstream prompt blend.
    return ["--scgw_p2", "--scgw_p3", "--result_tag", "scgw_p3"]


def _scgw_p4_args() -> List[str]:
    return _band_pretrain_only() + ["--scgw_p4", "--result_tag", "scgw_p4"]


def _band_fusion2_scgw_p4_stack() -> List[str]:
    """Band+tail + P4 pretrain; fusion2 downstream (hard router + BiKT)."""
    return _full_bandv3_tail_stack() + ["--scgw_p4"] + _homo_router_args(
        ["--use_bikt", "--result_tag", "band_fusion2_scgw_p4"]
    )


def _f2p4_plugin_stack(extra: List[str]) -> List[str]:
    """F2+P4 base + one downstream plugin (extra must include --result_tag)."""
    return _full_bandv3_tail_stack() + ["--scgw_p4"] + _homo_router_args(
        ["--use_bikt"] + list(extra)
    )


def _band_fusion2_scgw_p34_stack() -> List[str]:
    """Band+tail + P4 pretrain + P3 downstream structural prompt (BiKT path wired)."""
    return _full_bandv3_tail_stack() + ["--scgw_p4", "--scgw_p3"] + _homo_router_args(
        ["--use_bikt", "--result_tag", "band_fusion2_scgw_p34"]
    )


def _band_fusion2_scgw_p34_hetero_struct_stack() -> List[str]:
    """P34 + hetero_struct_boost (lower struct_alpha on hetero episodes)."""
    return _full_bandv3_tail_stack() + ["--scgw_p4", "--scgw_p3"] + _homo_router_args(
        [
            "--use_bikt",
            "--hetero_struct_boost", "0.3",
            "--result_tag", "band_fusion2_scgw_p34_hetero_struct",
        ]
    )


def _band_fusion2_scgw_p34_hetero_struct_mtg_stack() -> List[str]:
    """P34 + H1 episode router + MTG hetero-only on BiKT/downprompt branch."""
    return _full_bandv3_tail_stack() + ["--scgw_p4", "--scgw_p3"] + _homo_router_args(
        [
            "--use_bikt",
            "--hetero_struct_boost", "0.3",
            "--use_mtg",
            "--mtg_prototypes", "4",
            "--mtg_hetero_only",
            "--result_tag", "band_fusion2_scgw_p34_hetero_struct_mtg",
        ]
    )


def _band_fusion2_scgw_p34_hetero_struct_dataset_stack() -> List[str]:
    """P34 + H1 + dataset-scope router (prescan mean homo -> uniform branch)."""
    return _full_bandv3_tail_stack() + ["--scgw_p4", "--scgw_p3"] + _homo_router_dataset_args(
        [
            "--use_bikt",
            "--hetero_struct_boost", "0.3",
            "--result_tag", "band_fusion2_scgw_p34_hetero_struct_dataset",
        ]
    )


def _band_fusion2_scgw_p34_blend_stack(blend: str, tag: str) -> List[str]:
    """P34 with custom scgw_prompt_blend for meta-prompt structural bias."""
    return _full_bandv3_tail_stack() + [
        "--scgw_p4", "--scgw_p3", "--scgw_prompt_blend", blend,
    ] + _homo_router_args(["--use_bikt", "--result_tag", tag])


def _dual_head_only() -> List[str]:
    """Dual classifier head without extra pretrain losses (matches original MDGFM pretrain)."""
    return [
        "--downstream_head", "dual",
        "--dual_alpha", "0.6",
        "--dual_epochs", "300",
        "--proto_weight", "0.05",
        "--dual_ensemble", "3",
    ]


def _dual_downstream_only() -> List[str]:
    return _dual_head_only() + [
        "--dpc_weight", "0.01",
        "--dpc_temp", "0.2",
    ]


def _tail_ema_only_args() -> List[str]:
    """Tail pretrain without feature dual-view invariance."""
    return [
        "--pretrain_ema",
        "--pretrain_ema_decay", "0.999",
    ]


def _full_bandv3_dual_band() -> List[str]:
    return _dual_downstream_only() + _band_pretrain_only()


def _full_bandv3_tail_stack() -> List[str]:
    """Band v3 + tail pretrain with Dual downstream hyperparams (no Scale)."""
    return _full_bandv3_dual_band() + _tail_pretrain_args()


def _band_fusion2_plus_downstream() -> List[str]:
    """DEPRECATED: reuses SGFM/SRM/LP (see EXPERIMENT_LOG §11.2, 淘汰清单). Do not use."""
    return [
        "--use_sgfm", "--sgfm_scale", "0.15", "--sgfm_bottleneck", "64",
        "--use_srm", "--srm_temp", "4.0",
        "--lp_refine", "--lp_beta", "0.3", "--lp_steps", "2",
        "--bikt_weight", "0.08",
        "--hetero_bikt_scale", "2.0",
        "--hetero_sim_temp", "0.7",
        "--use_hetero_film",
        "--hetero_struct_boost", "0.3",
    ]


# Modes that recycle failed downstream tricks; see docs/BAND_FUSION2_DOWNSTREAM_ROADMAP.md
DEPRECATED_MODES = frozenset({
    "band_fusion2_plus",
})

def _homo_router_args(extra: Optional[List[str]] = None) -> List[str]:
    cmd = ["--use_homo_router", "--homo_bypass_thresh", "0.52"]
    if extra:
        cmd.extend(extra)
    return cmd


def _mtg_downstream_args(tag: str) -> List[str]:
    """Band融合2 + MTG layer-wise message tuning (dual auto-reencodes)."""
    return ["--use_mtg", "--mtg_prototypes", "4", "--result_tag", tag]


def _band_fusion2_mtg_stack(tag: str) -> List[str]:
    return _full_bandv3_tail_stack() + _homo_router_args(
        ["--use_bikt"] + _mtg_downstream_args(tag)
    )


def _f2_base_stack(extra: Optional[List[str]] = None) -> List[str]:
    """Band融合2 = Band+tail + hard homo router + BiKT (no P4)."""
    bikt = ["--use_bikt", "--result_tag", "homo_router_bikt"]
    if extra:
        bikt = ["--use_bikt"] + list(extra)
    return _full_bandv3_tail_stack() + _homo_router_args(bikt)


def _f2_e1_gee_args() -> List[str]:
    return ["--f2_gee_branch", "dual", "--result_tag", "f2_e1_gee"]


def _f2_e2_node_w_args() -> List[str]:
    return [
        "--f2_node_w_branch", "bikt",
        "--f2_node_w_gamma", "1.0",
        "--result_tag", "f2_e2_node_w",
    ]


def _f2_e3_leaky_args() -> List[str]:
    return [
        "--f2_leaky_branch", "bikt",
        "--f2_leaky_alpha", "0.3",
        "--f2_leaky_steps", "2",
        "--f2_leaky_homo_cond",
        "--result_tag", "f2_e3_leaky",
    ]


def _f2p4_r1_teacher_args() -> List[str]:
    return [
        "--f2_teacher_branch", "bikt",
        "--f2_teacher_mix", "1.0",
        "--f2_teacher_gamma", "1.0",
        "--result_tag", "f2p4_r1_teacher",
    ]


def _f2p4_r2_subproto_args() -> List[str]:
    return [
        "--f2_subproto_branch", "dual",
        "--f2_subproto_k", "2",
        "--f2_lsub_weight", "0.1",
        "--result_tag", "f2p4_r2_subproto",
    ]


def _f2p4_r3_e2_args() -> List[str]:
    return [
        "--f2_node_w_branch", "bikt",
        "--f2_node_w_gamma", "1.0",
        "--result_tag", "f2p4_r3_e2",
    ]


def _f2p4_r4_gee_args() -> List[str]:
    return [
        "--f2_gee_branch", "dual",
        "--result_tag", "f2p4_r4_gee",
    ]


def _rq2_wo_band_stack() -> List[str]:
    """RQ2 pretrain ablation: drop BandGSL (+P4 align); keep tail + Router + BiKT + Dual-GEE."""
    return _dual_downstream_only() + _tail_pretrain_args() + _homo_router_args(
        [
            "--use_bikt",
            "--f2_gee_branch", "dual",
            "--result_tag", "rq2_wo_band",
        ]
    )


def _rq2_wo_he_stack() -> List[str]:
    """RQ2 w/o-He: remove heterophilic branch → always homophilic (Band+P4+GEE).

    Former mode/tag: rq2_wo_router.
    """
    return _full_bandv3_tail_stack() + [
        "--scgw_p4",
        "--f2_gee_branch", "dual",
        "--result_tag", "rq2_wo_he",
    ]


def _rq2_wo_ho_stack() -> List[str]:
    """RQ2 w/o-Ho: remove homophilic branch → always heterophilic (tau=1 + BiKT).

    Equiv. to full SpecGFM stack with h_e>tau never true.
    support-GEE stays Dual-only, so it is inactive under this lock.
    """
    return _full_bandv3_tail_stack() + [
        "--scgw_p4",
        "--use_homo_router",
        "--homo_bypass_thresh", "1.0",
        "--use_bikt",
        "--f2_gee_branch", "dual",
        "--result_tag", "rq2_wo_ho",
    ]


def _rq2_wo_router_stack() -> List[str]:
    """Deprecated alias of _rq2_wo_he_stack (keeps old CSV tag for resume)."""
    return _full_bandv3_tail_stack() + [
        "--scgw_p4",
        "--f2_gee_branch", "dual",
        "--result_tag", "rq2_wo_router",
    ]


def _f2p4_r5_lsmo_args() -> List[str]:
    return [
        "--f2_lsmo_branch", "bikt",
        "--f2_lsmo_weight", "0.05",
        "--result_tag", "f2p4_r5_lsmo",
    ]


def _f2p4_r1r4_combo_args() -> List[str]:
    """R1 teacher (BiKT) + R4 GEE (Dual) on F2+P4."""
    return [
        "--f2_teacher_branch", "bikt",
        "--f2_teacher_mix", "1.0",
        "--f2_teacher_gamma", "1.0",
        "--f2_gee_branch", "dual",
        "--result_tag", "f2p4_r1r4_combo",
    ]


def _f2p4_r3r4_combo_args() -> List[str]:
    """R3 node_w (BiKT) + R4 GEE (Dual) on F2+P4."""
    return [
        "--f2_node_w_branch", "bikt",
        "--f2_node_w_gamma", "1.0",
        "--f2_gee_branch", "dual",
        "--result_tag", "f2p4_r3r4_combo",
    ]


def _homo_router_dataset_args(extra: Optional[List[str]] = None) -> List[str]:
    """F2 stack + dataset-level router: mean episode homo -> all Dual or all BiKT."""
    return _homo_router_args(extra) + ["--homo_router_scope", "dataset"]


def _homo_router_soft_args(extra: Optional[List[str]] = None) -> List[str]:
    cmd = [
        "--use_homo_router",
        "--homo_bypass_thresh", "0.52",
        "--homo_router_soft",
        "--homo_soft_temp", "0.08",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def _uniprop_stack(scale_residual_gamma: float = 0.15) -> List[str]:
    """UniProp: unified Scale-residual pretrain; hard homo router + BiKT + dp-AP."""
    return (
        _full_bandv3_tail_stack()
        + _scale_residual_args(scale_residual_gamma)
        + _homo_router_args([
            "--use_bikt",
            "--use_uniprop",
            "--ap_num_hops", "3",
            "--dual_ensemble", "1",
            "--result_tag", "uniprop",
        ])
    )


def _homo_router_v2_args(extra: Optional[List[str]] = None) -> List[str]:
    """Deprecated: Dual only on Cora ( hurts Citeseer). Use v3 / beat_original instead."""
    cmd = [
        "--use_homo_router",
        "--homo_bypass_thresh", "0.52",
        "--homo_dual_domains", "Cora",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def _homo_router_v3_args(extra: Optional[List[str]] = None) -> List[str]:
    """Planetoid homophilic targets may use Dual; hetero targets use downprompt (+ BiKT)."""
    cmd = [
        "--use_homo_router",
        "--homo_bypass_thresh", "0.52",
        "--homo_dual_domains", "Cora,Citeseer,Pubmed",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


HOMOPHILIC_PLANETOID = frozenset({"Cora", "Citeseer", "Pubmed"})
HETERO_WIKI = frozenset({"Chameleon", "Squirrel"})


def _beat_original_pretrain(dataset: str) -> List[str]:
    """Per-target pretrain stack tuned to beat original MDGFM (see docs/FULL_DATA_COMPARISON)."""
    if dataset in HOMOPHILIC_PLANETOID:
        # Citeseer/Cora need Scale+Dual; historical +2~3 pt vs original
        return _full_bandv3_dual_band() + _tail_pretrain_args() + _scale_gnn_args()
    if dataset in HETERO_WIKI:
        # Wikipedia hetero: vanilla pretrain + downprompt/BiKT (Band+tail hurts)
        return []
    # WebKB hetero (Cornell): band+tail + downprompt, no Scale (Scale hurts Cornell)
    return _full_bandv3_dual_band() + _tail_pretrain_args()


def _film_residual_args(scale: str) -> List[str]:
    return ["--use_film_residual", "--film_residual_scale", scale]


def _tail_pretrain_args() -> List[str]:
    return [
        "--use_feat_dv_inv",
        "--feat_dv_weight", "0.15",
        "--feat_dv_dropout", "0.25",
        "--pretrain_ema",
        "--pretrain_ema_decay", "0.999",
    ]


def _gcil_spectral_args() -> List[str]:
    return ["--use_gcil", "--use_gcil_spectral"]


def _scale_gnn_args() -> List[str]:
    return ["--scale_encoder", "scale1"]


def _scale1_args() -> List[str]:
    return ["--scale_encoder", "scale1"]


def _scale2_args() -> List[str]:
    return ["--scale_encoder", "scale2"]


def _scale_residual_args(gamma: float) -> List[str]:
    return ["--scale_encoder", "scale_residual", "--scale_residual_gamma", str(gamma)]


def _mdgfm_ap_args() -> List[str]:
    return ["--use_mdgfm_ap", "--ap_homo_cond", "--ap_num_hops", "3"]


def _mdgfm_hs_args() -> List[str]:
    return ["--use_mdgfm_hs"]


def _mdgfm_graver_args() -> List[str]:
    return ["--use_mdgfm_graver"]


def _hybrid_spectral_pretrain_args() -> List[str]:
    return ["--use_hybrid_spectral_pretrain"]


def _gcil_scale_combo_args() -> List[str]:
    return _gcil_spectral_args() + _scale_gnn_args()


def _original_fusion1_args(extra: Optional[List[str]] = None) -> List[str]:
    """Route A: vanilla MDGFM pretrain + homo router (Dual vs downprompt)."""
    return _dual_downstream_only() + _homo_router_args(extra)


def _original_fusion2_args(extra: Optional[List[str]] = None) -> List[str]:
    """Route A: vanilla pretrain + homo router + BiKT on heterophilic episodes."""
    bikt = ["--use_bikt", "--result_tag", "original_fusion2"]
    if extra:
        bikt.extend(extra)
    return _dual_downstream_only() + _homo_router_args(bikt)


def append_mode_args(
    cmd: List[str],
    mode: str,
    scale_residual_gamma: float = 0.15,
    dataset: str = "Cora",
) -> List[str]:
    if mode == "original":
        return cmd
    if mode == "original_mtg":
        # Ablation only: MDGFM joint pretrain + MTG. Paper main-table MTG is
        # baselines/run_paper_baselines.py --method mtg (GraphCL-isolated SSL).
        return cmd + [
            "--use_mtg",
            "--no_mtg_hetero_only",
            "--result_tag",
            "mtg_on_mdgfm",
        ]
    if mode == "original_scgw_p1":
        return cmd + _scgw_p1_args()
    if mode == "original_scgw_p2":
        return cmd + _scgw_p2_args()
    if mode == "original_scgw_p2_bikt":
        return cmd + _scgw_p2_bikt_args()
    if mode == "original_scgw_p2_homo_router_bikt":
        return cmd + _scgw_p2_fusion2_args()
    if mode == "original_scgw_p3":
        return cmd + _scgw_p3_args()
    if mode == "original_scgw_p4":
        return cmd + _scgw_p4_args()
    if mode == "original_bikt":
        return cmd + ["--use_bikt", "--result_tag", "bikt"]
    if mode == "original_prograph":
        return cmd + ["--use_prograph", "--result_tag", "prograph"]
    if mode == "original_mfgia":
        return cmd + ["--use_mfgia", "--result_tag", "mfgia"]
    if mode == "original_tri":
        return cmd + ["--use_mdgfm_tri", "--result_tag", "tri"]
    if mode == "original_dual":
        # Original MDGFM pretrain (PCA + vanilla GCN + downprompt stack) + Dual head
        return cmd + _dual_head_only()
    if mode == "original_homo_router":
        # Route A fusion1: same pretrain as original, episode homo router
        return cmd + _original_fusion1_args(["--result_tag", "original_fusion1"])
    if mode == "original_homo_router_bikt":
        # Route A fusion2: original pretrain + router + BiKT hetero branch
        return cmd + _original_fusion2_args()
    if mode == "original_homo_router_bikt_adapted":
        # Route A fusion2+: homophilic Dual on MDGFM 4.3 prompt+GSL path (HS-GPPT-style)
        return cmd + _original_fusion2_args(
            ["--dual_adapted", "--result_tag", "original_fusion2_adapted"]
        )
    if mode == "original_homo_router_bikt_adapted_fast":
        # Fast screen: 10 episodes × 50 dual_epochs × ensemble=1 (~3-4 min/run)
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "50",
            "--proto_weight", "0.05",
            "--dual_ensemble", "1",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_homo_router", "--homo_bypass_thresh", "0.52",
            "--use_bikt", "--dual_adapted",
            "--eval_episodes", "10",
            "--result_tag", "original_fusion2_adapted_fast",
        ]
    if mode == "original_graver":
        # Original MDGFM pretrain + GRAVER support augmentation + Dual head
        return cmd + _dual_head_only() + _mdgfm_graver_args()
    if mode == "band_downprompt":
        # Original pretrain stack + BandGSL v3 only; default downprompt downstream
        return cmd + _band_pretrain_only()
    if mode == "original_tail":
        # Original pretrain stack + tail (feat_dv + EMA) only; default downprompt
        return cmd + _tail_pretrain_args()
    if mode == "band_tail_downprompt":
        # Original pretrain stack + BandGSL v3 + tail; default downprompt (no Scale / Dual)
        return cmd + _band_pretrain_only() + _tail_pretrain_args()
    if mode == "full":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
        ]
    if mode == "full_bandv2":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
        ]
    if mode == "full_bandv3":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
            "--band_trust_low", "0.12",
            "--band_gate_clamp",
        ]
    if mode == "full_bandv3_strong":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
            "--band_trust_low", "0.2",
            "--band_gate_clamp",
            "--band_cd_weight", "0.05",
        ]
    if mode == "full_bandv4":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
            "--band_trust_low", "0.2",
            "--band_gate_clamp",
            "--band_cd_weight", "0.05",
            "--band_deg_anchor", "0.05",
            "--band_gate_entropy", "0.02",
        ]
    if mode == "full_bandv5":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
            "--band_trust_low", "0.12",
            "--band_gate_clamp",
            "--band_cd_weight", "0.1",
            "--band_deg_anchor", "0.03",
            "--band_gate_entropy", "0.01",
        ]
    if mode == "full_bandv3_mae":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
            "--band_trust_low", "0.12",
            "--band_gate_clamp",
            "--use_graph_mae",
            "--mae_weight", "0.25",
            "--mae_mask_ratio", "0.2",
        ]
    if mode == "full_bandv3_tail":
        return cmd + _full_bandv3_tail_stack()
    if mode == "full_bandv3_tail_homo_router":
        return cmd + _full_bandv3_tail_stack() + _homo_router_args()
    if mode == "full_bandv3_tail_homo_router_bikt":
        return cmd + _f2_base_stack()
    # F2 + single downstream plugins (E1/E2/E3); compare vs F2 baseline
    if mode == "f2_e1_gee":
        return cmd + _f2_base_stack(_f2_e1_gee_args())
    if mode == "f2_e2_node_w":
        return cmd + _f2_base_stack(_f2_e2_node_w_args())
    if mode == "f2_e3_leaky":
        return cmd + _f2_base_stack(_f2_e3_leaky_args())
    # F2+P4 + single downstream plugin (R1–R5); R6 = reuse winner modes on other --dataset
    if mode == "f2p4_r1_teacher":
        return cmd + _f2p4_plugin_stack(_f2p4_r1_teacher_args())
    if mode == "f2p4_r2_subproto":
        return cmd + _f2p4_plugin_stack(_f2p4_r2_subproto_args())
    if mode == "f2p4_r3_e2":
        return cmd + _f2p4_plugin_stack(_f2p4_r3_e2_args())
    if mode == "f2p4_r4_gee":
        return cmd + _f2p4_plugin_stack(_f2p4_r4_gee_args())
    if mode == "f2p4_r5_lsmo":
        return cmd + _f2p4_plugin_stack(_f2p4_r5_lsmo_args())
    if mode == "f2p4_r1r4_combo":
        return cmd + _f2p4_plugin_stack(_f2p4_r1r4_combo_args())
    if mode == "f2p4_r3r4_combo":
        return cmd + _f2p4_plugin_stack(_f2p4_r3r4_combo_args())
    # Paper RQ2 aliases (1-shot ablation): full / w/o-GEE reuse existing stacks & CSV tags
    if mode == "rq2_full":
        return cmd + _f2p4_plugin_stack(_f2p4_r4_gee_args())
    if mode == "rq2_wo_gee":
        return cmd + _band_fusion2_scgw_p4_stack()
    if mode == "rq2_wo_band":
        return cmd + _rq2_wo_band_stack()
    if mode == "rq2_wo_he":
        return cmd + _rq2_wo_he_stack()
    if mode == "rq2_wo_ho":
        return cmd + _rq2_wo_ho_stack()
    if mode == "rq2_wo_router":
        # Deprecated alias of rq2_wo_he / w/o-He (old CSV tag rq2_wo_router).
        return cmd + _rq2_wo_router_stack()
    if mode == "band_fusion2_mtg":
        return cmd + _band_fusion2_mtg_stack("band_fusion2_mtg")
    if mode == "band_fusion2_mtg_bikt":
        return cmd + _band_fusion2_mtg_stack("band_fusion2_mtg_bikt")
    if mode == "full_bandv3_tail_homo_router_bikt_dataset":
        return cmd + _full_bandv3_tail_stack() + _homo_router_dataset_args(
            ["--use_bikt", "--result_tag", "homo_router_bikt_dataset"]
        )
    if mode == "band_fusion2_scgw_p4":
        return cmd + _band_fusion2_scgw_p4_stack()
    if mode == "band_fusion2_scgw_p34":
        return cmd + _band_fusion2_scgw_p34_stack()
    if mode == "band_fusion2_scgw_p34_hetero_struct":
        return cmd + _band_fusion2_scgw_p34_hetero_struct_stack()
    if mode == "band_fusion2_scgw_p34_hetero_struct_mtg":
        return cmd + _band_fusion2_scgw_p34_hetero_struct_mtg_stack()
    if mode == "band_fusion2_scgw_p34_hetero_struct_dataset":
        return cmd + _band_fusion2_scgw_p34_hetero_struct_dataset_stack()
    if mode == "band_fusion2_scgw_p34_blend02":
        return cmd + _band_fusion2_scgw_p34_blend_stack("0.2", "band_fusion2_scgw_p34_blend02")
    if mode == "band_fusion2_scgw_p34_blend04":
        return cmd + _band_fusion2_scgw_p34_blend_stack("0.4", "band_fusion2_scgw_p34_blend04")
    if mode == "band_fusion2_plus":
        # DEPRECATED — see docs/BAND_FUSION2_DOWNSTREAM_ROADMAP.md
        return cmd + _full_bandv3_tail_stack() + _homo_router_args(
            ["--use_bikt", "--result_tag", "band_fusion2_plus"]
        ) + _band_fusion2_plus_downstream()
    if mode == "full_bandv3_tail_homo_router_soft":
        return cmd + _full_bandv3_tail_stack() + _homo_router_soft_args()
    if mode == "full_bandv3_tail_homo_router_soft_bikt":
        return cmd + _full_bandv3_tail_stack() + _homo_router_soft_args(
            ["--use_bikt", "--result_tag", "homo_router_soft_bikt"]
        )
    if mode == "full_bandv3_tail_uniprop":
        return cmd + _uniprop_stack(scale_residual_gamma)
    if mode == "full_bandv3_tail_homo_router_v2":
        return cmd + _full_bandv3_tail_stack() + _homo_router_v2_args()
    if mode == "full_bandv3_tail_homo_router_bikt_v2":
        return cmd + _full_bandv3_tail_stack() + _homo_router_v2_args(
            [
                "--use_bikt",
                "--result_tag", "homo_router_bikt_v2",
                "--bikt_weight", "0.08",
                "--hetero_bikt_scale", "2.0",
            ]
        )
    if mode == "beat_original_homo_router":
        return (
            cmd
            + _dual_downstream_only()
            + _beat_original_pretrain(dataset)
            + _homo_router_v3_args()
        )
    if mode == "beat_original_homo_router_bikt":
        return (
            cmd
            + _dual_downstream_only()
            + _beat_original_pretrain(dataset)
            + _homo_router_v3_args(
                [
                    "--use_bikt",
                    "--result_tag", "beat_original_bikt",
                    "--bikt_weight", "0.08",
                    "--hetero_bikt_scale", "2.5",
                ]
            )
        )
    if mode == "full_bandv3_film":
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
            "--band_trust_low", "0.12",
            "--band_gate_clamp",
            "--use_film_prompt",
        ]
    if mode == "full_bandv3_film_res":
        return cmd + _full_bandv3_dual_band() + _film_residual_args("0.15")
    if mode == "full_bandv3_film_res_tail":
        return cmd + _full_bandv3_dual_band() + _film_residual_args("0.15") + _tail_pretrain_args()
    if mode == "full_bandv3_tail_gcil_combo":
        return cmd + _full_bandv3_dual_band() + _tail_pretrain_args() + _gcil_scale_combo_args()
    if mode == "full_bandv3_tail_scale_gnn":
        return cmd + _full_bandv3_dual_band() + _tail_pretrain_args() + _scale_gnn_args()
    if mode == "full_bandv3_tail_scale2":
        return cmd + _full_bandv3_dual_band() + _tail_pretrain_args() + _scale2_args()
    if mode == "full_bandv3_tail_scale_residual":
        return cmd + _full_bandv3_dual_band() + _tail_pretrain_args() + _scale_residual_args(scale_residual_gamma)
    if mode == "full_bandv3_tail_scale_downprompt":
        # Ours pretrain stack + MDGFM default downprompt (hetero/homo fast screen)
        return cmd + _band_pretrain_only() + _tail_pretrain_args() + _scale_gnn_args()
    if mode == "full_bandv3_scale_gnn":
        # Band + Scale + Dual, no tail (hetero pretrain ablation)
        return cmd + _full_bandv3_dual_band() + _scale_gnn_args()
    if mode == "full_bandv3_tail_scale_gnn_band_strong":
        return cmd + _dual_downstream_only() + _band_strong_pretrain_only() + _tail_pretrain_args() + _scale_gnn_args()
    if mode == "full_bandv3_tail_scale_gnn_no_dv":
        return cmd + _full_bandv3_dual_band() + _scale_gnn_args() + _tail_ema_only_args()
    if mode == "full_bandv3_tail_scale_gnn_no_trust":
        return cmd + _dual_downstream_only() + _band_pretrain_no_trust() + _tail_pretrain_args() + _scale_gnn_args()
    if mode == "full_bandv3_tail_scale_gnn_no_cd":
        return cmd + _dual_downstream_only() + _band_pretrain_no_cd() + _tail_pretrain_args() + _scale_gnn_args()
    if mode == "full_bandv3_tail_gcil":
        return cmd + _full_bandv3_dual_band() + _tail_pretrain_args() + _gcil_spectral_args()
    if mode == "full_bandv3_film_res_s010":
        return cmd + _full_bandv3_dual_band() + _film_residual_args("0.10")
    if mode == "full_bandv3_film_res_s020":
        return cmd + _full_bandv3_dual_band() + _film_residual_args("0.20")
    if mode == "pca_gcil":
        return cmd + ["--use_gcil"]
    if mode == "pca_scale_gnn" or mode == "pca_scale1":
        return cmd + _scale1_args()
    if mode == "pca_scale2":
        return cmd + _scale2_args()
    if mode == "pca_gcil_scale_combo":
        return cmd + ["--use_gcil", "--use_gcil_spectral"] + _scale1_args()
    if mode == "mdgfm_ap":
        # Original MDGFM pretrain + Dual/DPC + adaptive propagation (no Scale)
        return cmd + _dual_downstream_only() + _mdgfm_ap_args()
    if mode == "mdgfm_hs_p0":
        # Deprecated alias → full MDGFM-HS (hybrid pretrain + two-phase downstream)
        return (
            cmd
            + _full_bandv3_dual_band()
            + _tail_pretrain_args()
            + _mdgfm_hs_args()
            + _hybrid_spectral_pretrain_args()
        )
    if mode == "mdgfm_hs":
        # P1: tail+band hybrid-spectral pretrain + spectral prompt downstream (no Scale)
        return (
            cmd
            + _full_bandv3_dual_band()
            + _tail_pretrain_args()
            + _mdgfm_hs_args()
            + _hybrid_spectral_pretrain_args()
        )
    if mode == "mdgfm_graver":
        # Ours pretrain (tail+band+Scale) + GRAVER support augmentation downstream
        return (
            cmd
            + _full_bandv3_dual_band()
            + _tail_pretrain_args()
            + _scale_gnn_args()
            + _mdgfm_graver_args()
        )
    if mode == "full_bandv3_tail_ap":
        # Tail+band pretrain + Dual/DPC + AP (no Scale)
        return cmd + [
            "--downstream_head", "dual",
            "--dual_alpha", "0.6",
            "--dual_epochs", "300",
            "--proto_weight", "0.05",
            "--dual_ensemble", "3",
            "--dpc_weight", "0.01",
            "--dpc_temp", "0.2",
            "--use_band_gsl",
            "--band_init_alpha", "0.5",
            "--band_v2_adaptive_gate",
            "--band_v2_cross_domain",
            "--band_trust_low", "0.12",
            "--band_gate_clamp",
            "--use_feat_dv_inv",
            "--feat_dv_weight", "0.15",
            "--feat_dv_dropout", "0.25",
            "--pretrain_ema",
            "--pretrain_ema_decay", "0.999",
        ] + _mdgfm_ap_args()
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Run target-domain seeds sequentially")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=list(DATASET_CONFIGS.keys()),
        help="Target domain (hyperparameters from README presets)",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "original",
            "original_mtg",
            "original_scgw_p1",
            "original_scgw_p2",
            "original_scgw_p2_bikt",
            "original_scgw_p2_homo_router_bikt",
            "original_scgw_p3",
            "original_scgw_p4",
            "original_bikt",
            "original_prograph",
            "original_mfgia",
            "original_tri",
            "original_dual",
            "original_homo_router",
            "original_homo_router_bikt",
            "original_homo_router_bikt_adapted",
            "original_homo_router_bikt_adapted_fast",
            "original_graver",
            "band_downprompt",
            "original_tail",
            "band_tail_downprompt",
            "full",
            "full_bandv2",
            "full_bandv3",
            "full_bandv3_strong",
            "full_bandv4",
            "full_bandv5",
            "full_bandv3_mae",
            "full_bandv3_tail",
            "full_bandv3_tail_homo_router",
            "full_bandv3_tail_homo_router_bikt",
            "f2_e1_gee",
            "f2_e2_node_w",
            "f2_e3_leaky",
            "f2p4_r1_teacher",
            "f2p4_r2_subproto",
            "f2p4_r3_e2",
            "f2p4_r4_gee",
            "f2p4_r5_lsmo",
            "f2p4_r1r4_combo",
            "f2p4_r3r4_combo",
            "rq2_full",
            "rq2_wo_gee",
            "rq2_wo_band",
            "rq2_wo_ho",
            "rq2_wo_he",
            "rq2_wo_router",
            "band_fusion2_mtg",
            "band_fusion2_mtg_bikt",
            "full_bandv3_tail_homo_router_bikt_dataset",
            "band_fusion2_scgw_p4",
            "band_fusion2_scgw_p34",
            "band_fusion2_scgw_p34_hetero_struct",
            "band_fusion2_scgw_p34_hetero_struct_mtg",
            "band_fusion2_scgw_p34_hetero_struct_dataset",
            "band_fusion2_scgw_p34_blend02",
            "band_fusion2_scgw_p34_blend04",
            "band_fusion2_plus",
            "full_bandv3_tail_homo_router_soft",
            "full_bandv3_tail_homo_router_soft_bikt",
            "full_bandv3_tail_uniprop",
            "full_bandv3_tail_homo_router_v2",
            "full_bandv3_tail_homo_router_bikt_v2",
            "beat_original_homo_router",
            "beat_original_homo_router_bikt",
            "full_bandv3_film",
            "full_bandv3_film_res",
            "full_bandv3_film_res_tail",
            "full_bandv3_film_res_s010",
            "full_bandv3_film_res_s020",
            "full_bandv3_tail_gcil_combo",
            "full_bandv3_tail_scale_gnn",
            "full_bandv3_tail_scale2",
            "full_bandv3_tail_scale_residual",
            "full_bandv3_tail_scale_downprompt",
            "full_bandv3_scale_gnn",
            "full_bandv3_tail_scale_gnn_band_strong",
            "full_bandv3_tail_scale_gnn_no_dv",
            "full_bandv3_tail_scale_gnn_no_trust",
            "full_bandv3_tail_scale_gnn_no_cd",
            "full_bandv3_tail_gcil",
            "pca_gcil",
            "pca_scale_gnn",
            "pca_scale1",
            "pca_scale2",
            "pca_gcil_scale_combo",
            "mdgfm_ap",
            "mdgfm_hs_p0",
            "mdgfm_hs",
            "mdgfm_graver",
            "full_bandv3_tail_ap",
        ],
        required=True,
        help="Experiment mode to run sequentially",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seed list, run in provided order",
    )
    parser.add_argument(
        "--scale_residual_gamma",
        type=float,
        default=0.15,
        help="[scale_residual] gamma in [0,1] when mode=full_bandv3_tail_scale_residual",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one run fails",
    )
    parser.add_argument(
        "--shot_num",
        type=int,
        default=1,
        help="Few-shot K (default 1; use 5 for README Table-2 style)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands only; do not launch training",
    )
    parser.add_argument(
        "--homo_bypass_thresh",
        type=float,
        default=None,
        help="Override homo router threshold (RQ3 tau sweep); also suffixes result_tag",
    )
    parser.add_argument(
        "--fixed_ckpt",
        type=str,
        default="",
        help="Fixed pretrain ckpt path (no timestamp); for shared-pretrain RQ3",
    )
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default="",
        help="Skip pretrain and load this ckpt (downstream-only)",
    )
    parser.add_argument(
        "--pretrain_only",
        action="store_true",
        help="Run pretrain, save ckpt, exit before downstream",
    )
    args = parser.parse_args()

    if args.mode in DEPRECATED_MODES:
        print(
            f"WARNING: mode={args.mode!r} is DEPRECATED "
            "(recycles SGFM/SRM/LP etc.; see docs/BAND_FUSION2_DOWNSTREAM_ROADMAP.md)",
            flush=True,
        )

    print(
        f"[{datetime.now()}] dataset={args.dataset}, mode={args.mode}, "
        f"seeds={args.seeds}, shot_num={args.shot_num}, "
        f"scale_residual_gamma={args.scale_residual_gamma}, "
        f"homo_bypass_thresh={args.homo_bypass_thresh}"
    )
    print("Runs are sequential (non-parallel).")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    failures = []
    for idx, seed in enumerate(args.seeds, start=1):
        cmd = append_mode_args(
            build_base_cmd(seed, args.dataset, shot_num=args.shot_num),
            args.mode,
            scale_residual_gamma=args.scale_residual_gamma,
            dataset=args.dataset,
        )
        cmd = apply_shot_result_tag(cmd, args.shot_num)
        if args.homo_bypass_thresh is not None:
            cmd = apply_homo_bypass_thresh(cmd, args.homo_bypass_thresh)
        if args.fixed_ckpt:
            cmd.extend(["--fixed_ckpt", args.fixed_ckpt])
        if args.load_pretrained:
            cmd.extend(["--load_pretrained", args.load_pretrained])
        if args.pretrain_only:
            cmd.append("--pretrain_only")
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        print("\n" + "=" * 100)
        print(f"[{datetime.now()}] ({idx}/{len(args.seeds)}) START seed={seed}")
        print(cmd_str)
        print("=" * 100)
        if args.dry_run:
            print(f"[{datetime.now()}] DRY_RUN skip seed={seed}")
            continue
        ret = subprocess.run(cmd, cwd=repo_root).returncode
        if ret == 0:
            print(f"[{datetime.now()}] DONE seed={seed} exit_code=0")
        else:
            print(f"[{datetime.now()}] FAIL seed={seed} exit_code={ret}")
            failures.append((seed, ret))
            if args.stop_on_error:
                break

    print("\n" + "-" * 100)
    if failures:
        print("Finished with failures:")
        for seed, code in failures:
            print(f"  - seed={seed}, exit_code={code}")
        sys.exit(1)
    if args.dry_run:
        print("Dry run finished (no training launched).")
    else:
        print("All runs finished successfully.")


if __name__ == "__main__":
    main()

