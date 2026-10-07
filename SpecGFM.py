"""SpecGFM pre-training (BandGSL) and homophily-guided dual-branch few-shot evaluation."""

from __future__ import annotations

import math
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
import time
from utils import Calbound
from models import LogReg
from preprompt import PrePrompt,pca_compression
import preprompt as preprompt
from scgw_utils import apply_scgw_p1_features, maybe_create_scgw
from utils import process
import pdb
import aug
import os
CKPT_DIR = 'checkpoints'


def _resolve_ckpt(name: str) -> str:
    if os.path.isabs(name):
        return name
    return os.path.join(CKPT_DIR, os.path.basename(name))


import tqdm
import argparse
from downprompt import downprompt,prefeatureprompt
from downprompt_bikt import downprompt_bikt
from downstream_encoder import build_downstream_encoder_from_model
import csv
from tqdm import tqdm


def gcn_for_downstream(model, use_dual_branch=False):
    del use_dual_branch
    return model.gcn


def append_mtg_optimizer_group(*_args, **_kwargs):
    return None


def apply_mtg_if_needed(*_args, **_kwargs):
    return None


def dual_uses_reencode(_args) -> bool:
    return False


def mtg_hetero_only_enabled(_args) -> bool:
    return False


def mtg_trainable_params(_gcn):
    return []


parser = argparse.ArgumentParser(
    "SpecGFM",
    description="Spectral Graph Foundation Model: BandGSL pre-training and homophily-guided dual-branch adaptation.",
)
import torch.nn.functional as F

parser.add_argument("--dataset", type=str, default="Cora",
                    choices=["Cora", "Citeseer", "Pubmed", "Cornell", "Chameleon", "Squirrel"],
                    help="Unseen target domain. The other five graphs are pre-training sources.")
parser.add_argument("--shot_num", type=int, default=1, help="Support labels per class (1 or 5).")
parser.add_argument("--seed", type=int, default=1024, help="Random seed.")
parser.add_argument("--epochs", type=int, default=60, help="Pre-training epochs.")
parser.add_argument("--eval_episodes", type=int, default=50, help="Few-shot episodes on the target.")
parser.add_argument("--lr", type=float, default=0.02, help="Pre-training learning rate.")
parser.add_argument("--downstreamlr", type=float, default=0.003, help="Downstream learning rate.")
parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
parser.add_argument("--save_name", type=str, default="checkpoints/specgfm.pkl", help="Pre-trained checkpoint path.")
parser.add_argument("--fixed_ckpt", type=str, default="", help="Reuse this checkpoint and skip pre-training.")
parser.add_argument("--load_pretrained", type=str, default="", help="Initialize from this checkpoint, then continue.")
parser.add_argument("--pretrain_only", action="store_true", help="Stop after pre-training.")
parser.add_argument("--combinetype", type=str, default="mul", help="Prompt combination: mul or add.")
parser.add_argument("--result_tag", type=str, default="", help="Suffix for the result CSV.")

parser.add_argument("--use_band_gsl", action="store_true",
                    help="BandGSL: low-pass and high-pass structures during pre-training.")
parser.add_argument("--band_init_alpha", type=float, default=0.5, help="Initial low/high band mix.")
parser.add_argument("--band_v2_adaptive_gate", action="store_true", help="Learned band gate.")
parser.add_argument("--band_v2_cross_domain", action="store_true", help="Cross-domain band consistency.")
parser.add_argument("--band_trust_low", type=float, default=0.0, help="Lower trust on the low-pass band.")
parser.add_argument("--band_gate_clamp", action="store_true", help="Clamp the band gate.")
parser.add_argument("--band_cd_weight", type=float, default=0.1, help="Weight of the cross-domain band loss.")

parser.add_argument("--use_feat_dv_inv", action="store_true",
                    help="Dual-view feature invariance on the fused adjacency.")
parser.add_argument("--feat_dv_weight", type=float, default=0.15)
parser.add_argument("--feat_dv_dropout", type=float, default=0.25)
parser.add_argument("--pretrain_ema", action="store_true", help="EMA of pre-training parameters.")
parser.add_argument("--pretrain_ema_decay", type=float, default=0.999)

parser.add_argument("--scgw_p4", action="store_true",
                    help="Structural coordinate alignment of the two BandGSL adjacencies.")
parser.add_argument("--scgw_num_bases", type=int, default=8, help="Number of geometric bases K.")
parser.add_argument("--scgw_base_size", type=int, default=16, help="Pooled coordinate size M.")
parser.add_argument("--scgw_tau", type=float, default=1.0, help="Coordinate softmax temperature.")
parser.add_argument("--scgw_weight", type=float, default=0.1, help="Weight of the coordinate-alignment loss.")

parser.add_argument("--downstream_head", type=str, default="dual", choices=["downprompt", "dual"],
                    help="Homophilic branch head. Paper runs use dual.")
parser.add_argument("--dual_alpha", type=float, default=0.5, help="Mix of linear and prototype logits.")
parser.add_argument("--dual_epochs", type=int, default=400, help="Homophilic-branch steps.")
parser.add_argument("--dual_ensemble", type=int, default=1,
                    help="Number of linear heads averaged on the homophilic branch.")
parser.add_argument("--proto_weight", type=float, default=0.0, help="Prototype loss weight.")
parser.add_argument("--dpc_weight", type=float, default=0.0, help="Domain prototype contrast weight.")
parser.add_argument("--dpc_temp", type=float, default=0.2)

parser.add_argument("--use_homo_router", action="store_true",
                    help="Route each episode by neighborhood homophily h_e.")
parser.add_argument("--homo_bypass_thresh", type=float, default=0.5,
                    help="Routing threshold tau. Homophilic branch if h_e > tau.")
parser.add_argument("--use_bikt", action="store_true",
                    help="Heterophilic branch (token/GSL re-encoding with an MLP view).")
parser.add_argument("--bikt_weight", type=float, default=0.05,
                    help="Consistency weight inside the heterophilic branch.")
parser.add_argument("--f2_gee_branch", type=str, default="off", choices=["off", "dual", "bikt", "hetero", "both"],
                    help="support-GEE. Paper full model uses dual (homophilic branch only).")

args = parser.parse_args()
_HIDDEN_DEFAULTS = {
    'feature_adapter': 'pca',
    'rp_dim': 512,
    'allin_post_norm': 'none',
    'coral_weight': 0.0,
    'finetune_gcn': False,
    'finetune_gcn_lr_scale': 0.1,
    'dual_tau': 1.0,
    'dual_calib': False,
    'pgr_weight': 0.0,
    'pgr_margin': 0.2,
    'lp_refine': False,
    'lp_beta': 0.3,
    'lp_steps': 2,
    'self_train_steps': 0,
    'self_train_thresh': 0.9,
    'self_train_weight': 0.3,
    'use_gfmate': False,
    'gfmate_tgcl_steps': -1,
    'gfmate_tgcl_weight': 1.0,
    'gfmate_tgcl_tau': 1.0,
    'gfmate_tgcl_tune': 'centroid',
    'use_gfmate_layer': False,
    'use_gfmate_decouple': False,
    'gfmate_decouple_blend': -1.0,
    'use_rgfm_gog': False,
    'use_rgfm_riemann': False,
    'rgfm_max_hop': 3,
    'rgfm_gog_edge_ratio': 0.6,
    'rgfm_gog_homo_min': 0.35,
    'rgfm_moe_top_m': 2,
    'rgfm_router_hidden': 64,
    'paper_plugins_all': False,
    'proto_margin': 0.0,
    'proto_margin_weight': 0.0,
    'learn_dual_alpha': False,
    'dual_label_smoothing': 0.0,
    'use_srm': False,
    'srm_temp': 4.0,
    'srm_reg_weight': 0.0,
    'domain_gate': False,
    'domain_gate_gamma': 0.3,
    'use_sgfm': False,
    'sgfm_bottleneck': 64,
    'sgfm_scale': 0.2,
    'use_hat_adapter': False,
    'hat_gate_hidden': 16,
    'hat_reg_weight': 0.0,
    'band_deg_anchor': 0.0,
    'band_gate_entropy': 0.0,
    'use_graph_mae': False,
    'mae_weight': 0.25,
    'mae_mask_ratio': 0.2,
    'use_film_prompt': False,
    'use_film_residual': False,
    'film_residual_scale': 0.15,
    'use_gcil': False,
    'gcil_weight': 0.1,
    'use_gcil_spectral': False,
    'gcil_inv_weight': 1.0,
    'gcil_indep_weight': 0.1,
    'use_scale_gnn': False,
    'scale_encoder': 'none',
    'scale_gnn_hops': 3,
    'scale_residual_gamma': 0.15,
    'use_ap': False,
    'ap_num_hops': 3,
    'ap_homo_cond': False,
    'ap_reg_weight': 0.0,
    'use_hs': False,
    'use_hybrid_spectral_pretrain': False,
    'hs_num_prompt': 10,
    'hs_tau_inner': 0.35,
    'hs_tau_cross': 0.25,
    'hs_prompt_epochs': 200,
    'hs_reg_weight': 0.01,
    'use_graver': False,
    'graver_vocab_size': 8,
    'graver_ego_hop': 1,
    'graver_max_ego_nodes': 24,
    'graver_vocab_noise': 0.05,
    'graver_router_hidden': 64,
    'graver_router_epochs': 30,
    'graver_moe_weight': 0.01,
    'graver_samples_per_class': 20,
    'graver_wildcard_per_domain': 30,
    'graver_global_weight': 0.4,
    'graver_wildcard_weight': 0.25,
    'graver_moe_aux_weight': 0.1,
    'use_prograph': False,
    'prograph_subspaces': 3,
    'prograph_view_weight': 0.05,
    'use_mfgia': False,
    'mfgia_domain_dim': 32,
    'mfgia_refresh_every': 0,
    'use_tri': False,
    'tri_subspaces': 3,
    'tri_domain_dim': 32,
    'tri_bikt_weight': 0.02,
    'tri_view_weight': 0.02,
    'tri_refresh_every': 0,
    'homo_dual_domains': '',
    'homo_router_scope': 'episode',
    'hetero_bikt_scale': 1.0,
    'hetero_sim_temp': 1.0,
    'use_hetero_film': False,
    'hetero_film_scale': 0.15,
    'hetero_struct_boost': 0.0,
    'f2_node_w_branch': 'off',
    'f2_node_w_gamma': 1.0,
    'f2_leaky_branch': 'off',
    'f2_leaky_alpha': 0.3,
    'f2_leaky_steps': 2,
    'f2_leaky_homo_cond': True,
    'f2_teacher_branch': 'off',
    'f2_teacher_mix': 1.0,
    'f2_teacher_gamma': 1.0,
    'f2_subproto_branch': 'off',
    'f2_subproto_k': 2,
    'f2_lsub_weight': 0.1,
    'f2_lsmo_branch': 'off',
    'f2_lsmo_weight': 0.05,
    'scgw_p1': False,
    'scgw_p2': False,
    'scgw_p3': False,
    'scgw_feat_blend': 0.3,
    'scgw_prompt_blend': 0.3,
    'homo_router_soft': False,
    'homo_soft_temp': 0.08,
    'use_uniprop': False,
    'dual_adapted': False,
    'use_mtg': False,
    'mtg_prototypes': 4,
    'mtg_hetero_only': True,
}
for _k, _v in _HIDDEN_DEFAULTS.items():
    if not hasattr(args, _k):
        setattr(args, _k, _v)

_route_flags = [args.use_bikt, args.use_prograph, args.use_mfgia, args.use_tri]
if args.paper_plugins_all:
    args.use_gfmate = True
    args.use_gfmate_layer = True
    args.use_gfmate_decouple = True
    args.use_rgfm_gog = True
    args.use_rgfm_riemann = True
if args.gfmate_tgcl_steps < 0:
    args.gfmate_tgcl_steps = 10 if args.use_gfmate else 0
if args.use_gfmate_decouple and args.gfmate_decouple_blend < 0:
    args.gfmate_decouple_blend = 0.5 if args.dataset.lower() == 'chameleon' else 0.0
elif args.gfmate_decouple_blend < 0:
    args.gfmate_decouple_blend = 0.0
if sum(int(x) for x in _route_flags) > 1:
    raise ValueError("only one of --use_bikt / --use_prograph / --use_mfgia / --use_tri")
if args.scgw_p4 and not args.use_band_gsl:
    raise ValueError("--scgw_p4 requires --use_band_gsl")
if args.scgw_p3 and not (args.scgw_p1 or args.scgw_p2 or args.scgw_p4):
    print(
        "[scgw_p3] WARNING: no P1/P2/P4 during pretrain — geometric bases stay at init; "
        "for meaningful P3, also enable --scgw_p2 (or use original_scgw_p3 mode).",
        flush=True,
    )
if args.use_homo_router:
    if args.use_ap or args.use_hs or args.use_graver:
        if not args.use_uniprop:
            raise ValueError("use_homo_router is mutually exclusive with use_ap/hs/graver")
    if args.downstream_head != 'dual':
        raise ValueError("use_homo_router requires --downstream_head dual (homophilic episodes use Dual)")
    if args.homo_router_soft and args.homo_soft_temp <= 0:
        raise ValueError("homo_soft_temp must be positive when homo_router_soft is enabled")
    if args.homo_router_scope == 'dataset':
        if args.homo_router_soft:
            raise ValueError("homo_router_scope=dataset is incompatible with --homo_router_soft")
        if args.use_uniprop:
            raise ValueError("homo_router_scope=dataset is incompatible with --use_uniprop")
if args.dual_adapted and not args.use_homo_router:
    raise ValueError("dual_adapted requires --use_homo_router")
if args.dual_adapted and args.use_ap:
    raise ValueError("dual_adapted is mutually exclusive with --use_ap")
if args.use_mtg and args.use_ap:
    raise ValueError("use_mtg is mutually exclusive with --use_ap")
if args.use_mtg and args.finetune_gcn:
    raise ValueError("use_mtg keeps GCN frozen; disable --finetune_gcn")
if args.use_uniprop:
    if not args.use_homo_router:
        raise ValueError("use_uniprop requires --use_homo_router")
    if args.homo_router_soft:
        raise ValueError("use_uniprop uses hard homo router; do not pass --homo_router_soft")
    if not args.use_bikt:
        raise ValueError("use_uniprop requires --use_bikt")
    if (args.scale_encoder or "none").strip().lower() != "scale_residual":
        raise ValueError(
            "use_uniprop requires --scale_encoder scale_residual "
            "(same pretrain method for all target domains)"
        )
else:
    if not args.use_homo_router:
        for _rf in _route_flags:
            if _rf and args.downstream_head != 'downprompt':
                raise ValueError("paper routes require --downstream_head downprompt (original MDGFM)")
        if any(_route_flags) and (args.use_ap or args.use_hs or args.use_graver):
            raise ValueError("paper routes are mutually exclusive with use_ap/hs/graver")

if args.use_tri and not args.use_homo_router:
    if args.downstream_head != 'downprompt':
        raise ValueError("use_tri requires --downstream_head downprompt (original MDGFM path)")
    if args.use_ap or args.use_hs or args.use_graver:
        raise ValueError("use_tri is mutually exclusive with use_ap/hs/graver")

if args.use_ap and args.use_hs:
    raise ValueError("use_ap and use_hs are mutually exclusive")
if args.use_graver and (args.use_ap or args.use_hs):
    raise ValueError("use_graver is mutually exclusive with use_ap and use_hs")
if args.use_hybrid_spectral_pretrain and not args.use_band_gsl:
    raise ValueError("use_hybrid_spectral_pretrain requires --use_band_gsl")

_paper_plugin_flags = [
    args.use_gfmate,
    args.use_gfmate_layer,
    args.use_gfmate_decouple,
    args.use_rgfm_gog,
    args.use_rgfm_riemann,
]
if any(_paper_plugin_flags):
    if args.downstream_head != 'downprompt':
        raise ValueError("paper plugins require --downstream_head downprompt (original MDGFM path)")
    if any(_route_flags):
        raise ValueError("paper plugins are for original downprompt only; disable paper routes")
    if args.use_ap or args.use_hs or args.use_graver:
        raise ValueError("paper plugins are mutually exclusive with use_ap/hs/graver")
if args.gfmate_tgcl_steps > 0 and not args.use_gfmate:
    raise ValueError("gfmate_tgcl_steps requires --use_gfmate (centroid prompt)")
if args.use_rgfm_riemann and not args.use_rgfm_gog:
    raise ValueError("use_rgfm_riemann requires --use_rgfm_gog")

print(
    "SpecGFM target={} shot={} seed={} epochs={} episodes={} "
    "BandGSL={} coordinate_align={} router={} tau={} "
    "heterophilic_branch={} support_GEE={}".format(
        args.dataset, args.shot_num, args.seed, args.epochs, args.eval_episodes,
        args.use_band_gsl, args.scgw_p4, args.use_homo_router, args.homo_bypass_thresh,
        args.use_bikt, args.f2_gee_branch,
    )
)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
# 固定随机种子，尽量保证实验可复现
seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit,Actor,WikipediaNetwork,WebKB,Flickr
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
print('-' * 100)
# 训练超参数（主流程里会分别用于预训练和下游阶段）
batch_size = 128
nb_epochs = args.epochs
patience = 500
lr_list=args.lr
l2_coef = 0.0001
drop_prob = 0.5
hid_units = 256
sparse = True
useMLP =False
LP = False
shot_num=args.shot_num
# 默认测试节点数；Pubmed 在后面会特殊改为 100
testnum = 10000
downstreamlrlist = args.downstreamlr
nonlinearity = 'prelu' 
dataset = args.dataset
device = torch.device("cuda")
best = 1e9
firstbest = 0

# =========================
# 预训练数据组织（对齐论文 6.1）
# - 默认 5 个源域：Cora/Pubmed/Citeseer/Chameleon/Squirrel
# - 当下游目标恰好是上述之一时，用 Cornell 作为替换源域
#   => 保证目标域为 unseen domain（更符合跨域迁移评估）
# =========================
dataset1 = Planetoid(root='data', name='Cora')                                                                                               
loader1 = DataLoader(dataset1)
dataset2 = Planetoid(root='data', name='Pubmed')                                                                                         
loader2 = DataLoader(dataset2)
dataset3 = Planetoid(root='data', name='Citeseer')
loader3 = DataLoader(dataset3)
dataset4 = WikipediaNetwork(root='data',name='Chameleon')
loader4 = DataLoader(dataset4)
dataset5 =WikipediaNetwork(root='data', name='Squirrel')
loader5 = DataLoader(dataset5)
dataset6 = WebKB(root='data', name='Cornell') 
loader6 = DataLoader(dataset6)
cnt_wait = 0
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
unify_dim = 50


def adapt_features_for_mdgfm(raw_features, unify_dim, args, seed_offset=0):
    """Feature adapter entry for Eq.(1) projection stage.

    - pca: original MDGFM path (default, unchanged).
    - allin: ALL-IN-inspired random projection + compression.
    """
    if args.feature_adapter == 'allin':
        # [ALL-IN] All domains must share the same random projection seed.
        # Keep seed_offset arg for backward-compatible call sites, but do not use
        # it in this branch to avoid per-domain projection mismatch.
        return allin_project_and_compress(
            raw_features,
            out_dim=unify_dim,
            rp_dim=args.rp_dim,
            seed=args.seed,
            post_norm=args.allin_post_norm,
        )
    # Original baseline branch (kept for backward compatibility).
    return pca_compression(raw_features, k=unify_dim)


def create_scgw_module(args, feat_dim: int):
    mod = maybe_create_scgw(args, feat_dim)
    if mod is not None and torch.cuda.is_available():
        mod = mod.cuda()
    return mod


def apply_scgw_p1_if_needed(features, adj_sp, scgw_module, args):
    if scgw_module is None or not args.scgw_p1:
        return features
    dev = next(scgw_module.parameters()).device
    x = torch.as_tensor(features, dtype=torch.float32, device=dev)
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj_sp).to(dev)
    out = apply_scgw_p1_features(x, sp_adj, scgw_module, blend=args.scgw_feat_blend)
    return out.detach().cpu().numpy()


def preprompt_ctor_kwargs(args, scgw_module):
    return dict(
        use_domain_gate=args.domain_gate,
        domain_gate_gamma=args.domain_gate_gamma,
        dpc_temp=args.dpc_temp,
        use_band_gsl=args.use_band_gsl,
        band_init_alpha=args.band_init_alpha,
        band_v2_adaptive_gate=args.band_v2_adaptive_gate,
        band_v2_cross_domain=args.band_v2_cross_domain,
        band_trust_low=args.band_trust_low,
        band_gate_clamp=args.band_gate_clamp,
        band_cd_weight=args.band_cd_weight,
        band_deg_anchor_weight=args.band_deg_anchor,
        band_gate_entropy_weight=args.band_gate_entropy,
        use_graph_mae=args.use_graph_mae,
        mae_weight=args.mae_weight,
        mae_mask_ratio=args.mae_mask_ratio,
        use_feat_dv_inv=args.use_feat_dv_inv,
        feat_dv_weight=args.feat_dv_weight,
        feat_dv_dropout=args.feat_dv_dropout,
        use_film_prompt=args.use_film_prompt,
        use_film_residual=args.use_film_residual,
        film_residual_scale=args.film_residual_scale,
        use_gcil=args.use_gcil,
        gcil_weight=args.gcil_weight,
        use_gcil_spectral=args.use_gcil_spectral,
        gcil_inv_weight=args.gcil_inv_weight,
        gcil_indep_weight=args.gcil_indep_weight,
        use_scale_gnn=args.use_scale_gnn,
        scale_gnn_hops=args.scale_gnn_hops,
        scale_encoder=args.scale_encoder,
        scale_residual_gamma=args.scale_residual_gamma,
        use_hybrid_spectral=args.use_hybrid_spectral_pretrain,
        use_scgw_p2=args.scgw_p2,
        use_scgw_p4=args.scgw_p4,
        scgw_module=scgw_module,
        scgw_weight=args.scgw_weight,
    )


def _covariance_torch(x: torch.Tensor) -> torch.Tensor:
    """Compute feature covariance for CORAL on [num_nodes, feat_dim]."""
    n = x.shape[0]
    if n <= 1:
        return torch.zeros((x.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
    x_centered = x - x.mean(dim=0, keepdim=True)
    return (x_centered.t() @ x_centered) / float(n - 1)


def coral_loss_pair(x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
    """CORAL loss between two feature matrices with same feature dim."""
    c_a = _covariance_torch(x_a)
    c_b = _covariance_torch(x_b)
    d = c_a.shape[0]
    return ((c_a - c_b) ** 2).sum() / (4.0 * d * d)


def coral_loss_multi(features_list) -> torch.Tensor:
    """Average pairwise CORAL over multiple domains."""
    loss = 0.0
    pairs = 0
    for i in range(len(features_list)):
        for j in range(i + 1, len(features_list)):
            loss = loss + coral_loss_pair(features_list[i], features_list[j])
            pairs += 1
    if pairs == 0:
        return torch.tensor(0.0, device=features_list[0].device)
    return loss / pairs


def prototype_loss(embeds: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """L2 distance from sample embeddings to their class prototypes."""
    target_proto = prototypes[labels]
    return ((embeds - target_proto) ** 2).mean()


def cosine_proto_logits(query_embeds: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    q = F.normalize(query_embeds, dim=1)
    p = F.normalize(prototypes, dim=1)
    return torch.mm(q, p.t())


def gfmate_complementary_labels(embeds: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """GFMate TGCL: assign each test node the least similar prototype class."""
    return cosine_proto_logits(embeds, prototypes).argmin(dim=1)


def gfmate_tgcl_loss(
    embeds: torch.Tensor,
    prototypes: torch.Tensor,
    comp_labels: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Encourage centroids to be distant from complementary (least similar) classes."""
    logits = cosine_proto_logits(embeds, prototypes) / max(float(tau), 1e-6)
    probs = F.softmax(logits, dim=1)
    idx = torch.arange(comp_labels.shape[0], device=comp_labels.device)
    p_comp = probs[idx, comp_labels]
    return -torch.log(1.0 - p_comp + 1e-8).mean()


def run_gfmate_tgcl_phase(
    args,
    log,
    features,
    sp_adj,
    sparse,
    episode_gcn,
    downk,
    idx_train,
    idx_test,
    pretrain_embs,
    test_embs,
    train_lbls,
    homo_score,
    downstreamlr,
):
    """GFMate test-time complementary learning on unlabeled test nodes."""
    from models.paper_route_plugins import pivot_layer_from_entropy

    if args.gfmate_tgcl_steps <= 0:
        return
    if args.gfmate_tgcl_tune == 'centroid':
        tgcl_params = []
        if getattr(log, 'centroid_prompt', None) is not None:
            tgcl_params.append(log.centroid_prompt)
        if getattr(log, 'layer_prompt', None) is not None:
            tgcl_params.extend(list(log.layer_prompt.parameters()))
        if not tgcl_params:
            tgcl_params = list(log.parameters())
    else:
        tgcl_params = list(log.parameters())
    opt = torch.optim.Adam(tgcl_params, lr=downstreamlr)
    use_ap = args.use_ap or args.use_uniprop
    total = int(args.gfmate_tgcl_steps)
    for step_i in range(total):
        log.train()
        if use_ap:
            log._ap_homo_score = homo_score if (args.ap_homo_cond or args.use_uniprop) else None
        opt.zero_grad()
        log(
            features, sp_adj, sparse, episode_gcn, idx_train, pretrain_embs, downk,
            train_lbls, 1, tgcl_step=step_i, tgcl_total=total,
        )
        _, test_dyn, protos = log(
            features, sp_adj, sparse, episode_gcn, idx_test, test_embs, downk,
            train_lbls, 0, return_aux=True, tgcl_step=step_i, tgcl_total=total,
        )
        layer_embeds = getattr(log, "_last_query_layer_embeds", None)
        if layer_embeds is None:
            layer_embeds = getattr(log, "_last_layer_embeds", None)
        layer_protos = getattr(log, "_last_layer_protos", None)
        if log.use_layer_prompt and layer_embeds and layer_protos:
            pivot = pivot_layer_from_entropy(layer_embeds, layer_protos)
            comp_labels = gfmate_complementary_labels(layer_embeds[pivot], layer_protos[pivot])
            loss = args.gfmate_tgcl_weight * gfmate_tgcl_loss(
                layer_embeds[pivot], layer_protos[pivot], comp_labels, args.gfmate_tgcl_tau,
            )
        else:
            comp_labels = gfmate_complementary_labels(test_dyn, protos)
            loss = args.gfmate_tgcl_weight * gfmate_tgcl_loss(
                test_dyn, protos, comp_labels, args.gfmate_tgcl_tau,
            )
        loss.backward()
        opt.step()


def build_prototypes(embeds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    protos = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            protos.append(embeds[mask].mean(dim=0))
        else:
            protos.append(embeds.mean(dim=0))
    return torch.stack(protos, dim=0)


def _downprompt_token_kwargs(model):
    """Frozen pretrain tokens for downstream prompt heads."""
    return dict(
        token1=model.texttoken1.weight.detach(),
        token2=model.texttoken2.weight.detach(),
        token3=model.texttoken3.weight.detach(),
        token4=model.texttoken4.weight.detach(),
        token5=model.texttoken5.weight.detach(),
        sumtext=model.sumtext.weight.detach(),
        pretoken1=model.pretext1.weight.detach(),
        pretoken2=model.pretext2.weight.detach(),
        pretoken3=model.pretext3.weight.detach(),
        pretoken4=model.pretext4.weight.detach(),
        pretoken5=model.pretext5.weight.detach(),
        balancetoken1=model.balancetoken1.weight.detach(),
        balancetoken2=model.balancetoken2.weight.detach(),
        balancetoken3=model.balancetoken3.weight.detach(),
        balancetoken4=model.balancetoken4.weight.detach(),
        balancetoken5=model.balancetoken5.weight.detach(),
    )


def _scgw_downstream_kwargs(args, model) -> dict:
    """SCGW P3: structural prompt bias (requires P2/P4-pretrained geometric bases)."""
    if not args.scgw_p3:
        return {}
    return dict(
        scgw_module=getattr(model, "scgw", None),
        scgw_prompt_blend=args.scgw_prompt_blend,
    )


def _gcn_num_layers(model) -> int:
    gcn = getattr(model, "gcn", None)
    return int(getattr(gcn, "num_layers_num", 3))


def _paper_plugin_kwargs(args, model, hid_units) -> dict:
    from models.paper_route_plugins import GoGEncoder, RiemannianMoE

    gog = None
    riemann = None
    if args.use_rgfm_gog:
        gog = GoGEncoder(hid_units, edge_ratio=args.rgfm_gog_edge_ratio)
    if args.use_rgfm_riemann:
        riemann = RiemannianMoE(
            hid_units,
            router_hidden=args.rgfm_router_hidden,
            top_m=args.rgfm_moe_top_m,
        )
    return dict(
        use_gfmate_centroid=args.use_gfmate,
        use_layer_prompt=args.use_gfmate_layer,
        num_gcn_layers=_gcn_num_layers(model),
        decouple_meta=args.use_gfmate_decouple,
        decouple_blend=args.gfmate_decouple_blend,
        gog_encoder=gog,
        riemann_moe=riemann,
        rgfm_max_hop=args.rgfm_max_hop,
        rgfm_gog_homo_min=args.rgfm_gog_homo_min,
    )


def build_downstream_log(args, model, hid_units, nb_classes, unify_dim):
    """Instantiate original or paper-route downstream head on frozen MDGFM pretrain."""
    kw = _downprompt_token_kwargs(model)
    common = dict(
        ft_in=hid_units,
        nb_classes=nb_classes,
        type=args.combinetype,
        feature_dim=unify_dim,
    )
    scgw_kw = _scgw_downstream_kwargs(args, model)
    if args.use_bikt:
        log = downprompt_bikt(
            **kw,
            **common,
            **scgw_kw,
            bikt_weight=args.bikt_weight,
            use_hetero_film=args.use_hetero_film,
            hetero_film_scale=args.hetero_film_scale,
        ).cuda()
        if args.use_uniprop:
            log.ap = AdaptivePropagationPrompt(
                num_hops=args.ap_num_hops,
                homo_condition=True,
            ).cuda()
        return log
    if args.use_prograph:
        return downprompt_prograph(
            **kw,
            **common,
            prograph_subspaces=args.prograph_subspaces,
            view_weight=args.prograph_view_weight,
        ).cuda()
    if args.use_mfgia:
        return downprompt_mfgia(
            **kw,
            **common,
            mfgia_domain_dim=args.mfgia_domain_dim,
        ).cuda()
    if args.use_tri:
        return downprompt_tri(
            **kw,
            **common,
            tri_subspaces=args.tri_subspaces,
            tri_domain_dim=args.tri_domain_dim,
        ).cuda()
    log = downprompt(
        **kw,
        **common,
        **scgw_kw,
        **_paper_plugin_kwargs(args, model, hid_units),
    ).cuda()
    if args.use_ap:
        log.ap = AdaptivePropagationPrompt(
            num_hops=args.ap_num_hops,
            homo_condition=args.ap_homo_cond,
        ).cuda()
    return log


def needs_episode_homo_score(args) -> bool:
    return (
        args.use_homo_router
        or args.use_uniprop
        or args.use_hat_adapter
        or (args.use_ap and args.ap_homo_cond)
        or args.use_tri
        or args.use_bikt
        or args.use_prograph
        or args.use_mfgia
        or args.use_rgfm_gog
    )


def needs_adj_dense_for_homo(args) -> bool:
    from models.f2_downstream_plugins import f2_plugins_need_adj
    return (
        args.use_homo_router
        or args.use_uniprop
        or args.use_hat_adapter
        or args.use_ap
        or args.use_hs
        or args.use_tri
        or args.use_rgfm_gog
        or args.use_bikt
        or args.use_prograph
        or args.use_mfgia
        or f2_plugins_need_adj(args)
    )


def homo_dual_domain_allowlist(args) -> set[str] | None:
    raw = (args.homo_dual_domains or '').strip()
    if not raw:
        return None
    return {d.strip() for d in raw.split(',') if d.strip()}


def domain_allows_dual(args) -> bool:
    allow = homo_dual_domain_allowlist(args)
    if allow is None:
        return True
    return args.dataset in allow


_HOMO_SOFT_EPS = 1e-4


def blend_homo_router_logits(logits_dp, logits_dual, dual_w: float):
    """Blend downprompt (softmax probs) and Dual (logits) with soft router weight."""
    if logits_dp is None:
        return logits_dual
    if logits_dual is None:
        return logits_dp
    if dual_w <= _HOMO_SOFT_EPS:
        return logits_dp
    if dual_w >= 1.0 - _HOMO_SOFT_EPS:
        return logits_dual
    w_t = logits_dp.new_tensor(dual_w)
    return w_t * logits_dual + (1.0 - w_t) * logits_dp


def episode_dual_weight(args, homo_score) -> float:
    """Return soft Dual weight in [0, 1]; 0 when this domain cannot use Dual."""
    if not domain_allows_dual(args):
        return 0.0
    if homo_score is None:
        return 0.0
    h = float(homo_score.reshape(()))
    tau = float(args.homo_bypass_thresh)
    temp = max(float(args.homo_soft_temp), 1e-6)
    return 1.0 / (1.0 + math.exp(-(h - tau) / temp))


def effective_bikt_weight(args, homo_score=None) -> float:
    w = float(args.bikt_weight)
    scale = float(args.hetero_bikt_scale)
    if scale == 1.0:
        return w
    from models.branch_utils import heterophily_weight
    if homo_score is not None:
        hetero = float(heterophily_weight(homo_score, homo_score).reshape(()))
        return w * (1.0 + (scale - 1.0) * hetero)
    if args.dataset in ('Chameleon', 'Squirrel', 'Cornell'):
        w *= scale
    return w


def resolve_dual_episode(args, homo_score) -> bool:
    """True -> Dual branch; False -> downprompt (+ optional paper route)."""
    if not args.use_homo_router:
        return args.downstream_head == 'dual'
    from models.branch_utils import homophilic_episode
    if homo_score is None:
        return args.downstream_head == 'dual'
    if not domain_allows_dual(args):
        return False
    return homophilic_episode(homo_score, args.homo_bypass_thresh)


def resolve_dataset_scope_dual(args, mean_homo: float) -> bool:
    """Dataset-level router: mean episode homo -> all Dual or all downprompt/route."""
    if not domain_allows_dual(args):
        return False
    from models.branch_utils import homophilic_episode
    ref = torch.tensor(float(mean_homo), dtype=torch.float32)
    return homophilic_episode(ref, args.homo_bypass_thresh)


def prescan_episode_homo_scores(
    args,
    shotnum: int,
    n_eval: int,
    adj_dense_full,
    labels,
) -> list[float]:
    """Collect support homophily for each eval episode (for dataset-scope router)."""
    scores: list[float] = []
    for i in range(n_eval):
        idx_train = torch.load(
            "data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(
                args.dataset.lower(), shotnum, args.dataset.lower(), i
            )
        ).type(torch.long).cuda()
        train_lbls = torch.load(
            "data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(
                args.dataset.lower(), shotnum, args.dataset.lower(), i
            )
        ).type(torch.long).squeeze().cuda()
        homo_score = episode_homo_score(
            adj_dense_full, labels, idx_train, train_lbls, True
        )
        if homo_score is None:
            homo_score = train_lbls.new_tensor(0.5, dtype=torch.float32)
        scores.append(float(homo_score.reshape(())))
    return scores


def configure_downstream_episode(log, args, homo_score, idx_train):
    """Per-episode context for paper-route downstream heads."""
    if args.use_mfgia:
        log.set_episode_context(support_idx=idx_train, homo_score=homo_score)
        return
    if args.use_tri:
        log.set_episode_context(
            homo_score,
            args.tri_bikt_weight,
            args.tri_view_weight,
            idx_train,
        )
        return
    if args.use_bikt:
        log.set_episode_context(
            homo_score,
            idx_train,
            bikt_weight=effective_bikt_weight(args, homo_score),
            hetero_sim_temp=args.hetero_sim_temp,
            hetero_struct_boost=args.hetero_struct_boost,
        )
        return
    if args.use_prograph:
        log.set_episode_context(
            homo_score, idx_train, view_weight=args.prograph_view_weight
        )


def maybe_refresh_domain_embedding(log, args, step, features, sp_adj, sparse, gcn, downk, idx_train, train_lbls):
    """Refresh MF-GIA domain fingerprint outside the training autograd graph."""
    refresh_every = 0
    if args.use_mfgia:
        refresh_every = args.mfgia_refresh_every
    elif args.use_tri:
        refresh_every = args.tri_refresh_every
    if not (args.use_mfgia or args.use_tri):
        return
    if step == 0 or (refresh_every > 0 and step > 0 and step % refresh_every == 0):
        with torch.no_grad():
            log.refresh_domain_embedding(
                features, sp_adj, sparse, gcn, downk, idx_train, train_lbls
            )


def compute_downprompt_train_loss(out, train_lbls, args):
    """Classification + optional route aux / prototype regularizers."""
    if isinstance(out, tuple) and len(out) == 3 and not isinstance(out[1], dict):
        logits, train_embs_dyn, train_proto = out
        cls_loss = xent(logits, train_lbls)
        if args.proto_weight > 0:
            return cls_loss + args.proto_weight * prototype_loss(
                train_embs_dyn, train_lbls, train_proto
            )
        return cls_loss
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        logits, aux = out
        cls_loss = xent(logits, train_lbls)
        loss = cls_loss + aux.get("bikt_loss", 0) + aux.get("view_loss", 0)
        if "f2_plugin_loss" in aux:
            loss = loss + aux["f2_plugin_loss"]
        if args.proto_weight > 0 and "train_embs" in aux and "proto" in aux:
            loss = loss + args.proto_weight * prototype_loss(
                aux["train_embs"], train_lbls, aux["proto"]
            )
        return loss
    return xent(out, train_lbls)


class SupportGuidedFiLM(nn.Module):
    """Support-Guided Representation Modulation (SGRM).

    Few-shot episode summary h = mean(z_support) drives a lightweight MLP that outputs
    per-dimension gamma, beta for affine modulation: z' = gamma ⊙ z + beta.
    Trained only in the dual-head loop alongside the linear probe; backbone stays frozen.
    """

    def __init__(self, dim: int, bottleneck: int = 64, scale: float = 0.2):
        super().__init__()
        b = max(8, int(bottleneck))
        self.net = nn.Sequential(
            nn.Linear(dim, b),
            nn.ReLU(),
            nn.Linear(b, 2 * dim),
        )
        self.scale = float(scale)

    def forward(self, emb: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        if summary.dim() == 1:
            summary = summary.unsqueeze(0)
        gb = self.net(summary)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(gamma)
        beta = self.scale * torch.tanh(beta)
        return emb * gamma + beta


def prototype_graph_regularization(prototypes: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Penalize overly similar class prototypes using off-diagonal cosine similarity."""
    p = F.normalize(prototypes, dim=1)
    sim = torch.mm(p, p.t())
    c = sim.shape[0]
    if c <= 1:
        return torch.tensor(0.0, device=sim.device, dtype=sim.dtype)
    mask = ~torch.eye(c, dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]
    return F.relu(off_diag - margin).mean()


def lp_refine_probs(test_logits: torch.Tensor, adj_dense: torch.Tensor, beta: float = 0.3, steps: int = 2) -> torch.Tensor:
    """Refine test probabilities with simple label propagation on test subgraph."""
    p0 = F.softmax(test_logits, dim=1)
    p = p0
    n = p.shape[0]
    if n == 0:
        return p0
    a = adj_dense.float()
    deg = a.sum(dim=1, keepdim=True).clamp_min(1e-6)
    a_norm = a / deg
    for _ in range(max(steps, 0)):
        p = (1 - beta) * p0 + beta * torch.mm(a_norm, p)
    return p


def margin_ce_loss(proto_logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Cross-entropy over prototype logits with true-class additive margin."""
    if margin <= 0:
        return F.cross_entropy(proto_logits, labels)
    adjusted = proto_logits.clone()
    adjusted[torch.arange(labels.shape[0], device=labels.device), labels] -= margin
    return F.cross_entropy(adjusted, labels)


def dual_mix_logits(logits_linear, logits_proto, alpha_raw, fixed_alpha: float, alpha_override=None):
    """Fuse linear and prototype logits; optional learnable weight via sigmoid(alpha_raw)."""
    if alpha_override is not None:
        return alpha_override * logits_linear + (1 - alpha_override) * logits_proto
    if alpha_raw is not None:
        w = torch.sigmoid(alpha_raw)
        return w * logits_linear + (1 - w) * logits_proto
    return fixed_alpha * logits_linear + (1 - fixed_alpha) * logits_proto


def srm_episode_alpha(logits_linear: torch.Tensor, logits_proto: torch.Tensor, base_alpha: float, temp: float) -> torch.Tensor:
    """Episode-wise adaptive alpha from confidence gap (linear vs prototype branch)."""
    with torch.no_grad():
        conf_linear = F.softmax(logits_linear, dim=1).max(dim=1).values.mean()
        conf_proto = F.softmax(logits_proto, dim=1).max(dim=1).values.mean()
        base = float(max(min(base_alpha, 1.0 - 1e-4), 1e-4))
        base_logit = math.log(base / (1.0 - base))
        alpha = torch.sigmoid(logits_linear.new_tensor(base_logit) + float(temp) * (conf_linear - conf_proto))
        return alpha.clamp(0.05, 0.95)


def dual_cls_loss(logits, labels, smoothing: float):
    """Main classification loss for dual head (optional label smoothing)."""
    if smoothing and smoothing > 0:
        return F.cross_entropy(logits, labels, label_smoothing=float(smoothing))
    return xent(logits, labels)


def support_homophily_ratio(adj_sub: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Estimate homophily from (weighted) adjacency and labels on a node subset."""
    if adj_sub.numel() == 0 or labels.numel() <= 1:
        return labels.new_tensor(0.5, dtype=torch.float32)
    n = labels.shape[0]
    eye = torch.eye(n, device=adj_sub.device, dtype=adj_sub.dtype)
    edge_w = (adj_sub * (1.0 - eye)).clamp_min(0.0)
    same = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    denom = edge_w.sum().clamp_min(1e-8)
    return ((edge_w * same).sum() / denom).float().clamp(0.0, 1.0)


def support_neighborhood_homophily(
    adj_dense_full: torch.Tensor,
    all_labels: torch.Tensor,
    idx_train: torch.Tensor,
) -> torch.Tensor:
    """1-hop neighbor label agreement averaged over support nodes (full-graph adj)."""
    device = adj_dense_full.device
    all_labels = all_labels.reshape(-1).to(device=device)
    idx_train = idx_train.reshape(-1).long().to(device)

    rows = adj_dense_full.index_select(0, idx_train).clone()
    rows[torch.arange(rows.shape[0], device=device), idx_train] = 0.0
    edge_w = rows.clamp_min(0.0)
    support_lbls = all_labels.index_select(0, idx_train).unsqueeze(1)
    same = (all_labels.unsqueeze(0) == support_lbls).float()
    denom = edge_w.sum(dim=1)
    homo_per = (edge_w * same).sum(dim=1) / denom.clamp_min(1e-8)
    valid = denom > 1e-8
    if not valid.any():
        return support_lbls.new_tensor(0.5, dtype=torch.float32)
    return homo_per[valid].mean().float().clamp(0.0, 1.0)


def episode_homo_score(
    adj_dense_full: torch.Tensor | None,
    all_labels: torch.Tensor,
    idx_train: torch.Tensor,
    train_lbls: torch.Tensor,
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled or adj_dense_full is None:
        return None
    del train_lbls
    return support_neighborhood_homophily(adj_dense_full, all_labels, idx_train)


def ap_dual_encode(
    encoder,
    features: torch.Tensor,
    sp_adj,
    sparse: bool,
    gcn: nn.Module,
    downk: int,
    homo_score: torch.Tensor | None,
    idx_train: torch.Tensor,
    idx_test,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_embeds = encoder.encode_graph(features, sp_adj, sparse, gcn, downk, homo_score)
    return all_embeds[idx_train], all_embeds[idx_test]


def run_downprompt_episode(
    args,
    model,
    features,
    sp_adj,
    sparse,
    downk,
    hid_units,
    nb_classes,
    unify_dim,
    idx_train,
    idx_test,
    pretrain_embs,
    test_embs,
    train_lbls,
    homo_score,
    downstreamlr,
):
    """Train downprompt head for one episode; return test logits."""
    episode_gcn = gcn_for_downstream(model, use_dual_branch=False)
    log = build_downstream_log(args, model, hid_units, nb_classes, unify_dim)
    if hasattr(log, '_episode_homo_score'):
        log._episode_homo_score = homo_score
    optim_groups = [{"params": log.parameters(), "lr": downstreamlr}]
    if args.finetune_gcn:
        optim_groups.append({
            "params": model.gcn.parameters(),
            "lr": downstreamlr * args.finetune_gcn_lr_scale,
        })
    append_mtg_optimizer_group(optim_groups, model, args, downstreamlr)
    opt = torch.optim.Adam(optim_groups, lr=downstreamlr)
    configure_downstream_episode(log, args, homo_score, idx_train)
    from models.f2_downstream_plugins import f2_plugins_need_adj
    if f2_plugins_need_adj(args) and hasattr(log, "set_f2_plugin_graph"):
        adj_d = sp_adj.to_dense() if sparse else (
            sp_adj if isinstance(sp_adj, torch.Tensor) else torch.as_tensor(sp_adj)
        )
        log.set_f2_plugin_graph(args, adj_d, train_lbls)
    use_route_aux = args.use_bikt or args.use_prograph or args.use_tri
    return_aux = use_route_aux or args.proto_weight > 0 or any(_paper_plugin_flags)
    maybe_refresh_domain_embedding(
        log, args, 0, features, sp_adj, sparse, episode_gcn, downk, idx_train, train_lbls,
    )
    best = 1e9
    cnt_wait = 0
    for step in range(400):
        log.train()
        if args.use_ap or args.use_uniprop:
            log._ap_homo_score = homo_score if (args.ap_homo_cond or args.use_uniprop) else None
        if step > 0:
            maybe_refresh_domain_embedding(
                log, args, step, features, sp_adj, sparse, episode_gcn, downk, idx_train, train_lbls,
            )
        opt.zero_grad()
        out = log(
            features, sp_adj, sparse, episode_gcn, idx_train, pretrain_embs, downk, train_lbls, 1,
            return_aux=return_aux,
        )
        loss = compute_downprompt_train_loss(out, train_lbls, args)
        if loss < best:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
        loss.backward()
        opt.step()
    run_gfmate_tgcl_phase(
        args, log, features, sp_adj, sparse, episode_gcn, downk,
        idx_train, idx_test, pretrain_embs, test_embs, train_lbls, homo_score, downstreamlr,
    )
    return log(features, sp_adj, sparse, episode_gcn, idx_test, test_embs, downk)


def run_standard_dual_episode(
    args,
    pretrain_embs,
    test_embs,
    train_lbls,
    homo_score,
    downstreamlr,
    hid_units,
    nb_classes,
    seed,
    episode_i,
    shotnum,
    adj_dense=None,
    full_embeds=None,
    idx_train=None,
):
    """Standard Dual head (pretrain embeddings) for one episode; return test logits."""
    from models.f2_downstream_plugins import dual_plugin_aux_loss, subproto_on_dual, lsmo_on_dual, teacher_on_dual

    K = max(1, int(args.dual_ensemble))
    in_dim = int(pretrain_embs.shape[1]) if pretrain_embs is not None else int(hid_units)
    logits_accum = None
    ens_base = int(seed) + episode_i * 100003 + int(shotnum) * 9973
    use_f2_dual_aux = subproto_on_dual(args) or lsmo_on_dual(args) or teacher_on_dual(args)
    for k_m in range(K):
        torch.manual_seed(ens_base + k_m * 1_000_003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ens_base + k_m * 1_000_003)
        dual_head = nn.Linear(in_dim, nb_classes).cuda()
        dual_params = list(dual_head.parameters())
        alpha_raw = None
        if args.learn_dual_alpha:
            p = float(max(min(args.dual_alpha, 1.0 - 1e-4), 1e-4))
            init_logit = math.log(p / (1.0 - p))
            alpha_raw = nn.Parameter(
                torch.tensor(init_logit, device=pretrain_embs.device, dtype=torch.float32)
            )
            dual_params.append(alpha_raw)
        if args.dual_calib:
            calib_scale = nn.Parameter(torch.ones(1, nb_classes).cuda())
            calib_bias = nn.Parameter(torch.zeros(1, nb_classes).cuda())
            dual_params += [calib_scale, calib_bias]
        else:
            calib_scale = None
            calib_bias = None
        sgfm_mod = None
        if args.use_sgfm:
            sgfm_mod = SupportGuidedFiLM(
                in_dim, bottleneck=args.sgfm_bottleneck, scale=args.sgfm_scale,
            ).cuda()
            dual_params.extend(list(sgfm_mod.parameters()))
        gate_mlp = None
        if args.use_hat_adapter:
            gate_hidden = max(4, int(args.hat_gate_hidden))
            gate_mlp = nn.Sequential(
                nn.Linear(3, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1),
            ).cuda()
            dual_params.extend(list(gate_mlp.parameters()))
        opt = torch.optim.Adam(dual_params, lr=downstreamlr)
        for _ in range(args.dual_epochs):
            dual_head.train()
            if sgfm_mod is not None:
                sgfm_mod.train()
                summary_ctx = pretrain_embs.mean(dim=0, keepdim=True)
                pe = sgfm_mod(pretrain_embs, summary_ctx)
            else:
                pe = pretrain_embs
            opt.zero_grad()
            logits_linear = dual_head(pe)
            if use_f2_dual_aux:
                proto, f2_extra = dual_plugin_aux_loss(
                    args, pe, train_lbls, nb_classes,
                    adj_dense=adj_dense, full_embeds=full_embeds, idx_train=idx_train,
                )
            else:
                proto = build_prototypes(pe, train_lbls, nb_classes)
                f2_extra = pe.new_zeros(())
            logits_proto = cosine_proto_logits(pe, proto) / max(args.dual_tau, 1e-6)
            alpha_srm = None
            if args.use_srm:
                alpha_base = torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                alpha_srm = srm_episode_alpha(logits_linear, logits_proto, alpha_base, args.srm_temp)
            if gate_mlp is not None:
                logits_mix, gate_val = hat_mix_logits(
                    logits_linear, logits_proto, proto, gate_mlp, homo_score
                )
            else:
                logits_mix = dual_mix_logits(
                    logits_linear, logits_proto, alpha_raw, args.dual_alpha, alpha_override=alpha_srm
                )
            if args.dual_calib:
                logits_mix = logits_mix * calib_scale + calib_bias
            cls_loss = dual_cls_loss(logits_mix, train_lbls, args.dual_label_smoothing)
            if args.proto_margin_weight > 0:
                margin_loss = margin_ce_loss(logits_proto, train_lbls, args.proto_margin)
                cls_loss = cls_loss + args.proto_margin_weight * margin_loss
            if args.proto_weight > 0:
                proto_reg = prototype_loss(pe, train_lbls, proto)
                loss = cls_loss + args.proto_weight * proto_reg
            else:
                loss = cls_loss
            if args.pgr_weight > 0:
                pgr_reg = prototype_graph_regularization(proto, margin=args.pgr_margin)
                loss = loss + args.pgr_weight * pgr_reg
            if args.use_srm and args.srm_reg_weight > 0 and alpha_srm is not None:
                alpha_ref = torch.sigmoid(alpha_raw) if alpha_raw is not None else logits_linear.new_tensor(float(args.dual_alpha))
                loss = loss + args.srm_reg_weight * (alpha_srm - alpha_ref.detach()) ** 2
            if gate_mlp is not None and args.hat_reg_weight > 0:
                loss = loss + args.hat_reg_weight * (gate_val - 0.5) ** 2
            if use_f2_dual_aux:
                loss = loss + f2_extra
            loss.backward()
            opt.step()
        if args.self_train_steps > 0:
            for _ in range(args.self_train_steps):
                dual_head.train()
                if sgfm_mod is not None:
                    sgfm_mod.train()
                    summary_ctx = pretrain_embs.mean(dim=0, keepdim=True)
                    pe = sgfm_mod(pretrain_embs, summary_ctx)
                    te = sgfm_mod(test_embs, summary_ctx)
                else:
                    pe = pretrain_embs
                    te = test_embs
                opt.zero_grad()
                logits_linear_train = dual_head(pe)
                proto_train = build_prototypes(pe, train_lbls, nb_classes)
                logits_proto_train = cosine_proto_logits(pe, proto_train) / max(args.dual_tau, 1e-6)
                alpha_srm_train = None
                if args.use_srm:
                    alpha_base_train = torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                    alpha_srm_train = srm_episode_alpha(
                        logits_linear_train, logits_proto_train, alpha_base_train, args.srm_temp
                    )
                if gate_mlp is not None:
                    logits_mix_train, _ = hat_mix_logits(
                        logits_linear_train, logits_proto_train, proto_train, gate_mlp, homo_score
                    )
                else:
                    logits_mix_train = dual_mix_logits(
                        logits_linear_train, logits_proto_train, alpha_raw, args.dual_alpha,
                        alpha_override=alpha_srm_train,
                    )
                if args.dual_calib:
                    logits_mix_train = logits_mix_train * calib_scale + calib_bias
                sup_loss = dual_cls_loss(logits_mix_train, train_lbls, args.dual_label_smoothing)
                logits_linear_test_st = dual_head(te)
                logits_proto_test_st = cosine_proto_logits(te, proto_train) / max(args.dual_tau, 1e-6)
                alpha_srm_test = None
                if args.use_srm:
                    alpha_base_test = torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                    alpha_srm_test = srm_episode_alpha(
                        logits_linear_train, logits_proto_train, alpha_base_test, args.srm_temp
                    )
                if gate_mlp is not None:
                    logits_mix_test_st, _ = hat_mix_logits(
                        logits_linear_test_st, logits_proto_test_st, proto_train, gate_mlp, homo_score
                    )
                else:
                    logits_mix_test_st = dual_mix_logits(
                        logits_linear_test_st, logits_proto_test_st, alpha_raw, args.dual_alpha,
                        alpha_override=alpha_srm_test,
                    )
                if args.dual_calib:
                    logits_mix_test_st = logits_mix_test_st * calib_scale + calib_bias
                probs_test = F.softmax(logits_mix_test_st, dim=1)
                conf, pseudo_lbl = probs_test.max(dim=1)
                mask = conf >= args.self_train_thresh
                if mask.any():
                    pseudo_loss = xent(logits_mix_test_st[mask], pseudo_lbl[mask])
                    st_loss = sup_loss + args.self_train_weight * pseudo_loss
                else:
                    st_loss = sup_loss
                st_loss.backward()
                opt.step()
        dual_head.eval()
        if sgfm_mod is not None:
            sgfm_mod.eval()
        with torch.no_grad():
            if sgfm_mod is not None:
                summary_ctx = pretrain_embs.mean(dim=0, keepdim=True)
                pe = sgfm_mod(pretrain_embs, summary_ctx)
                te = sgfm_mod(test_embs, summary_ctx)
            else:
                pe = pretrain_embs
                te = test_embs
            logits_linear_test = dual_head(te)
            if use_f2_dual_aux:
                proto_test, _ = dual_plugin_aux_loss(
                    args, pe, train_lbls, nb_classes,
                    adj_dense=adj_dense, full_embeds=full_embeds, idx_train=idx_train,
                )
            else:
                proto_test = build_prototypes(pe, train_lbls, nb_classes)
            logits_proto_test = cosine_proto_logits(te, proto_test) / max(args.dual_tau, 1e-6)
            alpha_srm_eval = None
            if args.use_srm:
                logits_linear_train_eval = dual_head(pe)
                logits_proto_train_eval = cosine_proto_logits(pe, proto_test) / max(args.dual_tau, 1e-6)
                alpha_base_eval = torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                alpha_srm_eval = srm_episode_alpha(
                    logits_linear_train_eval, logits_proto_train_eval, alpha_base_eval, args.srm_temp
                )
            if gate_mlp is not None:
                logits_k, _ = hat_mix_logits(
                    logits_linear_test, logits_proto_test, proto_test, gate_mlp, homo_score
                )
            else:
                logits_k = dual_mix_logits(
                    logits_linear_test, logits_proto_test, alpha_raw, args.dual_alpha,
                    alpha_override=alpha_srm_eval,
                )
            if args.dual_calib:
                logits_k = logits_k * calib_scale + calib_bias
            if logits_accum is None:
                logits_accum = logits_k
            else:
                logits_accum = logits_accum + logits_k
    return logits_accum / float(K)


def run_dual_reencode_episode(
    args,
    model,
    features,
    sp_adj,
    sparse,
    downk,
    hid_units,
    nb_classes,
    unify_dim,
    idx_train,
    idx_test,
    train_lbls,
    homo_score,
    downstreamlr,
    seed,
    episode_i,
    shotnum,
    use_ap: bool,
):
    """Dual on per-step MDGFM prompt→GSL→(optional AP)→frozen GCN (Route A homophilic expert)."""
    K = max(1, int(args.dual_ensemble))
    ap_homo = homo_score if (use_ap and args.ap_homo_cond) else None
    device = features.device
    encoder = build_downstream_encoder_from_model(
        model,
        args.combinetype,
        unify_dim,
        hid_units,
        use_ap=use_ap,
        ap_num_hops=args.ap_num_hops,
        ap_homo_cond=args.ap_homo_cond,
    ).cuda()
    logits_accum = None
    ens_base = int(seed) + episode_i * 100003 + int(shotnum) * 9973
    mtg_params = (
        mtg_trainable_params(model.gcn)
        if args.use_mtg and not mtg_hetero_only_enabled(args)
        else []
    )
    for k_m in range(K):
        torch.manual_seed(ens_base + k_m * 1_000_003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ens_base + k_m * 1_000_003)
        dual_head = nn.Linear(hid_units, nb_classes).cuda()
        dual_params = list(dual_head.parameters()) + list(encoder.parameters()) + mtg_params
        alpha_raw = None
        if args.learn_dual_alpha:
            p = float(max(min(args.dual_alpha, 1.0 - 1e-4), 1e-4))
            init_logit = math.log(p / (1.0 - p))
            alpha_raw = nn.Parameter(torch.tensor(init_logit, device=device, dtype=torch.float32))
            dual_params.append(alpha_raw)
        if args.dual_calib:
            calib_scale = nn.Parameter(torch.ones(1, nb_classes, device=device))
            calib_bias = nn.Parameter(torch.zeros(1, nb_classes, device=device))
            dual_params += [calib_scale, calib_bias]
        else:
            calib_scale = None
            calib_bias = None
        sgfm_mod = None
        if args.use_sgfm:
            sgfm_mod = SupportGuidedFiLM(
                hid_units,
                bottleneck=args.sgfm_bottleneck,
                scale=args.sgfm_scale,
            ).cuda()
            dual_params.extend(list(sgfm_mod.parameters()))
        gate_mlp = None
        if args.use_hat_adapter:
            gate_hidden = max(4, int(args.hat_gate_hidden))
            gate_mlp = nn.Sequential(
                nn.Linear(3, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
            ).cuda()
            dual_params.extend(list(gate_mlp.parameters()))
        opt = torch.optim.Adam(dual_params, lr=downstreamlr)
        for _ in range(args.dual_epochs):
            encoder.train()
            dual_head.train()
            if sgfm_mod is not None:
                sgfm_mod.train()
            pe, te = ap_dual_encode(
                encoder, features, sp_adj, sparse, model.gcn, downk, ap_homo, idx_train, idx_test
            )
            if sgfm_mod is not None:
                summary_ctx = pe.mean(dim=0, keepdim=True)
                pe = sgfm_mod(pe, summary_ctx)
            opt.zero_grad()
            logits_linear = dual_head(pe)
            proto = build_prototypes(pe, train_lbls, nb_classes)
            logits_proto = cosine_proto_logits(pe, proto) / max(args.dual_tau, 1e-6)
            alpha_srm = None
            if args.use_srm:
                alpha_base = torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                alpha_srm = srm_episode_alpha(logits_linear, logits_proto, alpha_base, args.srm_temp)
            if gate_mlp is not None:
                logits_mix, gate_val = hat_mix_logits(
                    logits_linear, logits_proto, proto, gate_mlp, homo_score
                )
            else:
                logits_mix = dual_mix_logits(
                    logits_linear, logits_proto, alpha_raw, args.dual_alpha, alpha_override=alpha_srm
                )
            if args.dual_calib:
                logits_mix = logits_mix * calib_scale + calib_bias
            cls_loss = dual_cls_loss(logits_mix, train_lbls, args.dual_label_smoothing)
            if args.proto_margin_weight > 0:
                margin_loss = margin_ce_loss(logits_proto, train_lbls, args.proto_margin)
                cls_loss = cls_loss + args.proto_margin_weight * margin_loss
            if args.proto_weight > 0:
                proto_reg = prototype_loss(pe, train_lbls, proto)
                loss = cls_loss + args.proto_weight * proto_reg
            else:
                loss = cls_loss
            if args.pgr_weight > 0:
                pgr_reg = prototype_graph_regularization(proto, margin=args.pgr_margin)
                loss = loss + args.pgr_weight * pgr_reg
            if args.use_srm and args.srm_reg_weight > 0 and alpha_srm is not None:
                alpha_ref = (
                    torch.sigmoid(alpha_raw)
                    if alpha_raw is not None
                    else logits_linear.new_tensor(float(args.dual_alpha))
                )
                loss = loss + args.srm_reg_weight * (alpha_srm - alpha_ref.detach()) ** 2
            if gate_mlp is not None and args.hat_reg_weight > 0:
                loss = loss + args.hat_reg_weight * (gate_val - 0.5) ** 2
            if args.ap_reg_weight > 0 and encoder.ap is not None:
                loss = loss + args.ap_reg_weight * (encoder.ap.logits ** 2).mean()
            loss.backward()
            opt.step()
        if args.self_train_steps > 0:
            for _ in range(args.self_train_steps):
                encoder.train()
                dual_head.train()
                if sgfm_mod is not None:
                    sgfm_mod.train()
                pe, te = ap_dual_encode(
                    encoder, features, sp_adj, sparse, model.gcn, downk, ap_homo, idx_train, idx_test
                )
                if sgfm_mod is not None:
                    summary_ctx = pe.mean(dim=0, keepdim=True)
                    pe = sgfm_mod(pe, summary_ctx)
                    te = sgfm_mod(te, summary_ctx)
                opt.zero_grad()
                logits_linear_train = dual_head(pe)
                proto_train = build_prototypes(pe, train_lbls, nb_classes)
                logits_proto_train = cosine_proto_logits(pe, proto_train) / max(args.dual_tau, 1e-6)
                alpha_srm_train = None
                if args.use_srm:
                    alpha_base_train = (
                        torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                    )
                    alpha_srm_train = srm_episode_alpha(
                        logits_linear_train, logits_proto_train, alpha_base_train, args.srm_temp
                    )
                if gate_mlp is not None:
                    logits_mix_train, _ = hat_mix_logits(
                        logits_linear_train, logits_proto_train, proto_train, gate_mlp, homo_score
                    )
                else:
                    logits_mix_train = dual_mix_logits(
                        logits_linear_train, logits_proto_train, alpha_raw, args.dual_alpha,
                        alpha_override=alpha_srm_train,
                    )
                if args.dual_calib:
                    logits_mix_train = logits_mix_train * calib_scale + calib_bias
                sup_loss = dual_cls_loss(logits_mix_train, train_lbls, args.dual_label_smoothing)
                logits_linear_test_st = dual_head(te)
                logits_proto_test_st = cosine_proto_logits(te, proto_train) / max(args.dual_tau, 1e-6)
                alpha_srm_test = None
                if args.use_srm:
                    alpha_base_test = (
                        torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                    )
                    alpha_srm_test = srm_episode_alpha(
                        logits_linear_train, logits_proto_train, alpha_base_test, args.srm_temp
                    )
                if gate_mlp is not None:
                    logits_mix_test_st, _ = hat_mix_logits(
                        logits_linear_test_st, logits_proto_test_st, proto_train, gate_mlp, homo_score
                    )
                else:
                    logits_mix_test_st = dual_mix_logits(
                        logits_linear_test_st, logits_proto_test_st, alpha_raw, args.dual_alpha,
                        alpha_override=alpha_srm_test,
                    )
                if args.dual_calib:
                    logits_mix_test_st = logits_mix_test_st * calib_scale + calib_bias
                probs_test = F.softmax(logits_mix_test_st, dim=1)
                conf, pseudo_lbl = probs_test.max(dim=1)
                mask = conf >= args.self_train_thresh
                if mask.any():
                    pseudo_loss = xent(logits_mix_test_st[mask], pseudo_lbl[mask])
                    st_loss = sup_loss + args.self_train_weight * pseudo_loss
                else:
                    st_loss = sup_loss
                st_loss.backward()
                opt.step()
        dual_head.eval()
        encoder.eval()
        if sgfm_mod is not None:
            sgfm_mod.eval()
        with torch.no_grad():
            pe, te = ap_dual_encode(
                encoder, features, sp_adj, sparse, model.gcn, downk, ap_homo, idx_train, idx_test
            )
            if sgfm_mod is not None:
                summary_ctx = pe.mean(dim=0, keepdim=True)
                pe = sgfm_mod(pe, summary_ctx)
                te = sgfm_mod(te, summary_ctx)
            logits_linear_test = dual_head(te)
            proto_test = build_prototypes(pe, train_lbls, nb_classes)
            logits_proto_test = cosine_proto_logits(te, proto_test) / max(args.dual_tau, 1e-6)
            alpha_srm_eval = None
            if args.use_srm:
                logits_linear_train_eval = dual_head(pe)
                logits_proto_train_eval = cosine_proto_logits(pe, proto_test) / max(args.dual_tau, 1e-6)
                alpha_base_eval = (
                    torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                )
                alpha_srm_eval = srm_episode_alpha(
                    logits_linear_train_eval, logits_proto_train_eval, alpha_base_eval, args.srm_temp
                )
            if gate_mlp is not None:
                logits_k, _ = hat_mix_logits(
                    logits_linear_test, logits_proto_test, proto_test, gate_mlp, homo_score
                )
            else:
                logits_k = dual_mix_logits(
                    logits_linear_test, logits_proto_test, alpha_raw, args.dual_alpha,
                    alpha_override=alpha_srm_eval,
                )
            if args.dual_calib:
                logits_k = logits_k * calib_scale + calib_bias
        if logits_accum is None:
            logits_accum = logits_k
        else:
            logits_accum = logits_accum + logits_k
    return logits_accum / float(K)


def run_dual_ap_episode(
    args,
    model,
    features,
    sp_adj,
    sparse,
    downk,
    hid_units,
    nb_classes,
    unify_dim,
    idx_train,
    idx_test,
    train_lbls,
    homo_score,
    downstreamlr,
    seed,
    episode_i,
    shotnum,
):
    """MDGFM-AP + Dual: per-step prompt→GSL→AP→frozen GCN re-encode."""
    return run_dual_reencode_episode(
        args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
        unify_dim, idx_train, idx_test, train_lbls, homo_score, downstreamlr,
        seed, episode_i, shotnum, use_ap=True,
    )


def run_dual_adapted_episode(
    args,
    model,
    features,
    sp_adj,
    sparse,
    downk,
    hid_units,
    nb_classes,
    unify_dim,
    idx_train,
    idx_test,
    train_lbls,
    homo_score,
    downstreamlr,
    seed,
    episode_i,
    shotnum,
):
    """MDGFM 4.3 prompt+GSL+frozen GCN per step, then Dual (vanilla pretrain, Route A)."""
    return run_dual_reencode_episode(
        args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
        unify_dim, idx_train, idx_test, train_lbls, homo_score, downstreamlr,
        seed, episode_i, shotnum, use_ap=False,
    )


def run_dual_hs_episode(
    args,
    model,
    features,
    sp_adj,
    sparse,
    downk,
    hid_units,
    nb_classes,
    unify_dim,
    idx_train,
    idx_test,
    train_lbls,
    homo_score,
    downstreamlr,
    seed,
    episode_i,
    shotnum,
):
    """MDGFM-HS + Dual: phase-1 spectral prompt tune, then frozen-z Dual (Ours-stable)."""
    device = features.device
    encoder = build_downstream_hs_encoder_from_model(
        model,
        args.combinetype,
        unify_dim,
        hid_units,
        num_prompt=args.hs_num_prompt,
        tau_inner=args.hs_tau_inner,
        tau_cross=args.hs_tau_cross,
        freeze_band_fusion=True,
    ).cuda()

    prompt_opt = torch.optim.Adam(encoder.spectral_trainable_parameters(), lr=downstreamlr)
    for _ in range(max(1, int(args.hs_prompt_epochs))):
        encoder.train()
        prompt_opt.zero_grad()
        all_emb = encoder.encode_graph(features, sp_adj, sparse, model.gcn, downk, None)
        pe = all_emb[idx_train]
        proto = build_prototypes(pe, train_lbls, nb_classes)
        logits_proto = cosine_proto_logits(pe, proto) / max(args.dual_tau, 1e-6)
        loss = xent(logits_proto, train_lbls)
        if args.hs_reg_weight > 0:
            loss = loss + args.hs_reg_weight * (
                encoder.prompt_low.prompt_feats.pow(2).mean()
                + encoder.prompt_high.prompt_feats.pow(2).mean()
            )
        loss.backward()
        prompt_opt.step()

    encoder.eval()
    with torch.no_grad():
        all_emb = encoder.encode_graph(features, sp_adj, sparse, model.gcn, downk, None)
        pretrain_embs = all_emb[idx_train].detach()
        test_embs = all_emb[idx_test].detach()

    K = max(1, int(args.dual_ensemble))
    logits_accum = None
    ens_base = int(seed) + episode_i * 100003 + int(shotnum) * 9973
    for k_m in range(K):
        torch.manual_seed(ens_base + k_m * 1_000_003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ens_base + k_m * 1_000_003)
        dual_head = nn.Linear(hid_units, nb_classes).cuda()
        dual_params = list(dual_head.parameters())
        alpha_raw = None
        if args.learn_dual_alpha:
            p = float(max(min(args.dual_alpha, 1.0 - 1e-4), 1e-4))
            init_logit = math.log(p / (1.0 - p))
            alpha_raw = nn.Parameter(torch.tensor(init_logit, device=device, dtype=torch.float32))
            dual_params.append(alpha_raw)
        if args.dual_calib:
            calib_scale = nn.Parameter(torch.ones(1, nb_classes, device=device))
            calib_bias = nn.Parameter(torch.zeros(1, nb_classes, device=device))
            dual_params += [calib_scale, calib_bias]
        else:
            calib_scale = None
            calib_bias = None
        sgfm_mod = None
        if args.use_sgfm:
            sgfm_mod = SupportGuidedFiLM(
                hid_units,
                bottleneck=args.sgfm_bottleneck,
                scale=args.sgfm_scale,
            ).cuda()
            dual_params.extend(list(sgfm_mod.parameters()))
        gate_mlp = None
        if args.use_hat_adapter:
            gate_hidden = max(4, int(args.hat_gate_hidden))
            gate_mlp = nn.Sequential(
                nn.Linear(3, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
            ).cuda()
            dual_params.extend(list(gate_mlp.parameters()))
        opt = torch.optim.Adam(dual_params, lr=downstreamlr)
        for _ in range(args.dual_epochs):
            dual_head.train()
            if sgfm_mod is not None:
                sgfm_mod.train()
                summary_ctx = pretrain_embs.mean(dim=0, keepdim=True)
                pe = sgfm_mod(pretrain_embs, summary_ctx)
            else:
                pe = pretrain_embs
            opt.zero_grad()
            logits_linear = dual_head(pe)
            proto = build_prototypes(pe, train_lbls, nb_classes)
            logits_proto = cosine_proto_logits(pe, proto) / max(args.dual_tau, 1e-6)
            alpha_srm = None
            if args.use_srm:
                alpha_base = torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                alpha_srm = srm_episode_alpha(logits_linear, logits_proto, alpha_base, args.srm_temp)
            if gate_mlp is not None:
                logits_mix, gate_val = hat_mix_logits(
                    logits_linear, logits_proto, proto, gate_mlp, homo_score
                )
            else:
                logits_mix = dual_mix_logits(
                    logits_linear, logits_proto, alpha_raw, args.dual_alpha, alpha_override=alpha_srm
                )
            if args.dual_calib:
                logits_mix = logits_mix * calib_scale + calib_bias
            cls_loss = dual_cls_loss(logits_mix, train_lbls, args.dual_label_smoothing)
            if args.proto_margin_weight > 0:
                margin_loss = margin_ce_loss(logits_proto, train_lbls, args.proto_margin)
                cls_loss = cls_loss + args.proto_margin_weight * margin_loss
            if args.proto_weight > 0:
                proto_reg = prototype_loss(pe, train_lbls, proto)
                loss = cls_loss + args.proto_weight * proto_reg
            else:
                loss = cls_loss
            if args.pgr_weight > 0:
                pgr_reg = prototype_graph_regularization(proto, margin=args.pgr_margin)
                loss = loss + args.pgr_weight * pgr_reg
            if args.use_srm and args.srm_reg_weight > 0 and alpha_srm is not None:
                alpha_ref = (
                    torch.sigmoid(alpha_raw)
                    if alpha_raw is not None
                    else logits_linear.new_tensor(float(args.dual_alpha))
                )
                loss = loss + args.srm_reg_weight * (alpha_srm - alpha_ref.detach()) ** 2
            if gate_mlp is not None and args.hat_reg_weight > 0:
                loss = loss + args.hat_reg_weight * (gate_val - 0.5) ** 2
            loss.backward()
            opt.step()
        if args.self_train_steps > 0:
            for _ in range(args.self_train_steps):
                dual_head.train()
                if sgfm_mod is not None:
                    sgfm_mod.train()
                    summary_ctx = pretrain_embs.mean(dim=0, keepdim=True)
                    pe = sgfm_mod(pretrain_embs, summary_ctx)
                    te = sgfm_mod(test_embs, summary_ctx)
                else:
                    pe = pretrain_embs
                    te = test_embs
                opt.zero_grad()
                logits_linear_train = dual_head(pe)
                proto_train = build_prototypes(pe, train_lbls, nb_classes)
                logits_proto_train = cosine_proto_logits(pe, proto_train) / max(args.dual_tau, 1e-6)
                alpha_srm_train = None
                if args.use_srm:
                    alpha_base_train = (
                        torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                    )
                    alpha_srm_train = srm_episode_alpha(
                        logits_linear_train, logits_proto_train, alpha_base_train, args.srm_temp
                    )
                if gate_mlp is not None:
                    logits_mix_train, _ = hat_mix_logits(
                        logits_linear_train, logits_proto_train, proto_train, gate_mlp, homo_score
                    )
                else:
                    logits_mix_train = dual_mix_logits(
                        logits_linear_train, logits_proto_train, alpha_raw, args.dual_alpha,
                        alpha_override=alpha_srm_train,
                    )
                if args.dual_calib:
                    logits_mix_train = logits_mix_train * calib_scale + calib_bias
                sup_loss = dual_cls_loss(logits_mix_train, train_lbls, args.dual_label_smoothing)
                logits_linear_test_st = dual_head(te)
                logits_proto_test_st = cosine_proto_logits(te, proto_train) / max(args.dual_tau, 1e-6)
                alpha_srm_test = None
                if args.use_srm:
                    alpha_base_test = (
                        torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                    )
                    alpha_srm_test = srm_episode_alpha(
                        logits_linear_train, logits_proto_train, alpha_base_test, args.srm_temp
                    )
                if gate_mlp is not None:
                    logits_mix_test_st, _ = hat_mix_logits(
                        logits_linear_test_st, logits_proto_test_st, proto_train, gate_mlp, homo_score
                    )
                else:
                    logits_mix_test_st = dual_mix_logits(
                        logits_linear_test_st, logits_proto_test_st, alpha_raw, args.dual_alpha,
                        alpha_override=alpha_srm_test,
                    )
                if args.dual_calib:
                    logits_mix_test_st = logits_mix_test_st * calib_scale + calib_bias
                probs_test = F.softmax(logits_mix_test_st, dim=1)
                conf, pseudo_lbl = probs_test.max(dim=1)
                mask = conf >= args.self_train_thresh
                if mask.any():
                    pseudo_loss = xent(logits_mix_test_st[mask], pseudo_lbl[mask])
                    st_loss = sup_loss + args.self_train_weight * pseudo_loss
                else:
                    st_loss = sup_loss
                st_loss.backward()
                opt.step()
        dual_head.eval()
        if sgfm_mod is not None:
            sgfm_mod.eval()
        with torch.no_grad():
            if sgfm_mod is not None:
                summary_ctx = pretrain_embs.mean(dim=0, keepdim=True)
                pe = sgfm_mod(pretrain_embs, summary_ctx)
                te = sgfm_mod(test_embs, summary_ctx)
            else:
                pe = pretrain_embs
                te = test_embs
            logits_linear_test = dual_head(te)
            proto_test = build_prototypes(pe, train_lbls, nb_classes)
            logits_proto_test = cosine_proto_logits(te, proto_test) / max(args.dual_tau, 1e-6)
            alpha_srm_eval = None
            if args.use_srm:
                logits_linear_train_eval = dual_head(pe)
                logits_proto_train_eval = cosine_proto_logits(pe, proto_test) / max(args.dual_tau, 1e-6)
                alpha_base_eval = (
                    torch.sigmoid(alpha_raw).item() if alpha_raw is not None else args.dual_alpha
                )
                alpha_srm_eval = srm_episode_alpha(
                    logits_linear_train_eval, logits_proto_train_eval, alpha_base_eval, args.srm_temp
                )
            if gate_mlp is not None:
                logits_k, _ = hat_mix_logits(
                    logits_linear_test, logits_proto_test, proto_test, gate_mlp, homo_score
                )
            else:
                logits_k = dual_mix_logits(
                    logits_linear_test, logits_proto_test, alpha_raw, args.dual_alpha,
                    alpha_override=alpha_srm_eval,
                )
            if args.dual_calib:
                logits_k = logits_k * calib_scale + calib_bias
        if logits_accum is None:
            logits_accum = logits_k
        else:
            logits_accum = logits_accum + logits_k
    return logits_accum / float(K)


def hat_mix_logits(
    logits_topo: torch.Tensor,
    logits_feat: torch.Tensor,
    prototypes: torch.Tensor,
    gate_mlp: nn.Module,
    homo_score: torch.Tensor,
):
    """Homophily-adaptive blending between topology and feature branches."""
    conf_topo = F.softmax(logits_topo, dim=1).max(dim=1).values.mean()
    if prototypes.shape[0] > 1:
        p = F.normalize(prototypes, dim=1)
        sim = torch.mm(p, p.t())
        c = sim.shape[0]
        off_diag = sim[~torch.eye(c, dtype=torch.bool, device=sim.device)]
        proto_sep = (1.0 - off_diag.mean()).clamp(0.0, 1.0)
    else:
        proto_sep = conf_topo.new_tensor(0.5)
    stats = torch.stack([homo_score, conf_topo, proto_sep], dim=0).unsqueeze(0)
    gate = torch.sigmoid(gate_mlp(stats))  # [1, 1]
    mixed = gate * logits_topo + (1.0 - gate) * logits_feat
    return mixed, gate.squeeze()


os.makedirs(CKPT_DIR, exist_ok=True)
a = os.path.basename(args.save_name)
n_=0
for lr in [lr_list]:
    # 这里写成 [lr_list] 是为了保留“可扩展成多学习率网格搜索”的结构
    time_=time.localtime()
    n_+=1
    best = 1e9
    firstbest = 0
    if args.load_pretrained:
        args.save_name = args.load_pretrained
    elif args.fixed_ckpt:
        # Keep relative paths (incl. subdirs); only absolutize for mkdir.
        ck = args.fixed_ckpt
        args.save_name = ck
        parent = os.path.dirname(os.path.abspath(ck))
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        args.save_name = _resolve_ckpt(str(time_) + a)
    if args.load_pretrained:
        # Downstream-only: skip source-domain PCA / pretrain graph prep.
        if not os.path.isfile(args.load_pretrained):
            raise FileNotFoundError(
                '[load_pretrained] checkpoint not found: {}'.format(args.load_pretrained)
            )
        if args.use_graver:
            raise ValueError(
                '--load_pretrained currently incompatible with --use_graver '
                '(needs source-domain features).'
            )
        features1 = features2 = features3 = features4 = features5 = None
        labels1 = labels2 = labels3 = labels4 = labels5 = None
        sp_adj1 = sp_adj2 = sp_adj3 = sp_adj4 = sp_adj5 = None
        negative_sample = None
        scgw_module = None
        pre_i = 5
    else:
        # 构建 5 个预训练域输入（特征 + 邻接）
        # 若 target 在 source 集合中，则做一换一替换，避免“看见目标域”
        for step, (data1,data2,data3,data4,data5,data6) in enumerate(zip(loader1,loader2,loader3,loader4,loader5,loader6)):

            features11,adj1= process.process_tu(data1,data1.x.shape[1])
            features22,adj2= process.process_tu(data2,data2.x.shape[1])
            features33,adj3= process.process_tu(data3,data3.x.shape[1])
            features44,adj4= process.process_tu(data4,data4.x.shape[1])
            features55,adj5= process.process_tu(data5,data5.x.shape[1])
            # pre_i: 记录被替换的域索引，后续会影响 PrePrompt 内部的 kNN 图构建参数
            pre_i=5
            if args.dataset=='Cora':
                features11,adj1= process.process_tu(data6,data6.x.shape[1])
                pre_i=0
            elif args.dataset=='Pubmed':
                features22,adj2= process.process_tu(data6,data6.x.shape[1])
                pre_i=1
            elif args.dataset=='Citeseer':
                features33,adj3= process.process_tu(data6,data6.x.shape[1])
                pre_i=2
            elif args.dataset=='Chameleon':
                features44,adj4= process.process_tu(data6,data6.x.shape[1])
                pre_i=3
            elif args.dataset=='Squirrel':
                features55,adj5= process.process_tu(data6,data6.x.shape[1])
                pre_i=4
            src_d1, src_d2, src_d3, src_d4, src_d5 = data1, data2, data3, data4, data5
            if args.dataset == 'Cora':
                src_d1 = data6
            elif args.dataset == 'Pubmed':
                src_d2 = data6
            elif args.dataset == 'Citeseer':
                src_d3 = data6
            elif args.dataset == 'Chameleon':
                src_d4 = data6
            elif args.dataset == 'Squirrel':
                src_d5 = data6
            labels1 = torch.LongTensor(np.array(src_d1.y)).cuda()
            labels2 = torch.LongTensor(np.array(src_d2.y)).cuda()
            labels3 = torch.LongTensor(np.array(src_d3.y)).cuda()
            labels4 = torch.LongTensor(np.array(src_d4.y)).cuda()
            labels5 = torch.LongTensor(np.array(src_d5.y)).cuda()
            # 论文式 (1) 的投影入口：
            # - 默认：PCA（原始 MDGFM）
            # - [ALL-IN]：随机投影 + NodeCov + 压缩（启用 --feature_adapter allin）
            print('[pretrain] PCA domain 1/5 (Cora)', flush=True)
            features1 = adapt_features_for_mdgfm(features11, unify_dim, args, seed_offset=11)
            print('[pretrain] PCA domain 2/5 (Pubmed)', flush=True)
            features2 = adapt_features_for_mdgfm(features22, unify_dim, args, seed_offset=22)
            print('[pretrain] PCA domain 3/5 (slot3)', flush=True)
            features3 = adapt_features_for_mdgfm(features33, unify_dim, args, seed_offset=33)
            print('[pretrain] PCA domain 4/5 (Chameleon)', flush=True)
            features4 = adapt_features_for_mdgfm(features44, unify_dim, args, seed_offset=44)
            print('[pretrain] PCA domain 5/5 (Squirrel)', flush=True)
            features5 = adapt_features_for_mdgfm(features55, unify_dim, args, seed_offset=55)

            scgw_module = create_scgw_module(args, unify_dim)
            if scgw_module is not None:
                print('[pretrain] SCGFM geometric bases enabled', flush=True)
                features1 = apply_scgw_p1_if_needed(features1, adj1, scgw_module, args)
                features2 = apply_scgw_p1_if_needed(features2, adj2, scgw_module, args)
                features3 = apply_scgw_p1_if_needed(features3, adj3, scgw_module, args)
                features4 = apply_scgw_p1_if_needed(features4, adj4, scgw_module, args)
                features5 = apply_scgw_p1_if_needed(features5, adj5, scgw_module, args)

            print('[pretrain] moving features to GPU', flush=True)
            features1 = torch.FloatTensor(features1).cuda()
            features2 = torch.FloatTensor(features2).cuda()
            features3 = torch.FloatTensor(features3).cuda()
            features4 = torch.FloatTensor(features4).cuda()
            features5 = torch.FloatTensor(features5).cuda()

            # Legacy negative_sample table for PrePrompt ctor only (unused in forward).
            # Skip combine_dataset + O(n^2) row setdiff — unrelated to downstream router.
            nodenum = (
                adj1.shape[0] + adj2.shape[0] + adj3.shape[0]
                + adj4.shape[0] + adj5.shape[0]
            )
            print('[pretrain] negative_sample stub nodenum={}'.format(nodenum), flush=True)
            negative_sample = preprompt.prompt_pretrain_sample_fast(nodenum, 50, seed=int(args.seed))

        # 每个域分别加自环并归一化；后续以稀疏张量形式送入 GNN
        adj2 = process.normalize_adj(adj2 + sp.eye(adj2.shape[0]))
        adj1 = process.normalize_adj(adj1 + sp.eye(adj1.shape[0]))
        adj3 = process.normalize_adj(adj3 + sp.eye(adj3.shape[0]))
        adj4 = process.normalize_adj(adj4 + sp.eye(adj4.shape[0]))
        adj5 = process.normalize_adj(adj5 + sp.eye(adj5.shape[0]))
        if sparse:
            sp_adj1 = process.sparse_mx_to_torch_sparse_tensor(adj1)
            sp_adj2 = process.sparse_mx_to_torch_sparse_tensor(adj2)
            sp_adj3 = process.sparse_mx_to_torch_sparse_tensor(adj3)
            sp_adj4 = process.sparse_mx_to_torch_sparse_tensor(adj4)
            sp_adj5 = process.sparse_mx_to_torch_sparse_tensor(adj5)

    if args.load_pretrained:
        args.save_name = args.load_pretrained
        print('[pretrain] SKIP stage-1; reuse ckpt {}'.format(args.save_name), flush=True)
        if args.pretrain_only:
            print('[pretrain_only] nothing to do (already have ckpt); exit', flush=True)
            sys.exit(0)
    else:
        # 预训练模型（论文 4.2）：
        # `PrePrompt.forward` 内部会执行
        # 1) 域 token + 共享 token（式 (2)）
        # 2) GSL 构造 refined adjacency A'（式 (3)）
        # 3) 以 I_e 和 A' 为正样本关系做对比下界损失（式 (4)）
        model = PrePrompt(
            unify_dim,
            hid_units,
            nonlinearity,
            negative_sample,
            3,
            0.1,
            args.combinetype,
            **preprompt_ctor_kwargs(args, scgw_module),
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        if torch.cuda.is_available():
            model = model.cuda()
            features1 = features1.cuda()
            features2 = features2.cuda()
            features3 = features3.cuda()
            features4 = features4.cuda()
            features5 = features5.cuda()

            if sparse:
                sp_adj1 = sp_adj1.cuda()
                sp_adj2 = sp_adj2.cuda()
                sp_adj3 = sp_adj3.cuda()
                sp_adj4 = sp_adj4.cuda()
                sp_adj5 = sp_adj5.cuda()

        ema_shadow = None
        if args.pretrain_ema:
            ema_shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

        # ==========================================================
        # 阶段一：多域预训练（论文 4.1 + 4.2）
        # 目标：学习可迁移的 token/prompt 与结构感知图编码器参数
        # ==========================================================
        for epoch in range(nb_epochs):
            torch.cuda.empty_cache()
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            loss = 0
            regloss = 0
            model.train()
            optimiser.zero_grad()
            # 返回标量对齐损失（非分类 logits）；
            # 其定义对应论文式 (4) 的两项互信息下界近似。
            base_loss, dpc_loss = model( features1,features2,features3,features4,features5,
                        sp_adj1 if sparse else adj1, sp_adj2 if sparse else adj2,sp_adj3 if sparse else adj3,sp_adj4 if sparse else adj4,sp_adj5 if sparse else adj5,
                        sparse, None, None, None,pre_i, compute_dpc=(args.dpc_weight > 0))
            coral_reg = torch.tensor(0.0, device=base_loss.device)
            if args.coral_weight > 0:
                coral_reg = coral_loss_multi([features1, features2, features3, features4, features5])
            loss = base_loss + args.coral_weight * coral_reg + args.dpc_weight * dpc_loss
            loss.backward()
            optimiser.step()
            if args.pretrain_ema and ema_shadow is not None:
                d_ema = args.pretrain_ema_decay
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        ema_shadow[n].mul_(d_ema).add_(p.data, alpha=1.0 - d_ema)
            if args.coral_weight > 0:
                print('Loss:[{:.4f}] Base:[{:.4f}] Coral:[{:.4f}] DPC:[{:.4f}]'.format(
                    loss.item(), base_loss.item(), coral_reg.item(), dpc_loss.item()
                ))
            else:
                print('Loss:[{:.4f}] Base:[{:.4f}] DPC:[{:.4f}]'.format(
                    loss.item(), base_loss.item(), dpc_loss.item()
                ))
            if loss < best:
                firstbest = 1
                best = loss
                best_t = epoch
                cnt_wait = 0
                # 保存当前最优预训练参数（供下游阶段加载）
                torch.save(model.state_dict(), args.save_name)
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('Early stopping!')
                break
            print('Loading {}th epoch'.format(best_t))

        if args.pretrain_ema and ema_shadow is not None:
            sd = model.state_dict()
            for n in ema_shadow:
                sd[n] = ema_shadow[n].clone()
            torch.save(sd, args.save_name)
            print('[pretrain_ema] Overwrote {} with EMA-smoothed parameters.'.format(args.save_name))
        elif not os.path.isfile(args.save_name):
            torch.save(model.state_dict(), args.save_name)
            print('[pretrain] Saved final weights to {}'.format(args.save_name), flush=True)

        print('[pretrain] ckpt ready: {}'.format(args.save_name), flush=True)
        if args.pretrain_only:
            print('[pretrain_only] exit before downstream', flush=True)
            sys.exit(0)

    # ==========================================================
    # 阶段二：下游 few-shot 迁移（论文 4.3）
    # 流程：重建同构模型 -> 加载预训练权重 -> 提取目标域表示 -> 训练下游 prompt 头
    # ==========================================================
    scgw_module_ds = create_scgw_module(args, unify_dim)
    model = PrePrompt(
        unify_dim,
        hid_units,
        nonlinearity,
        1,
        3,
        0.1,
        args.combinetype,
        **preprompt_ctor_kwargs(args, scgw_module_ds),
    )

    print('#'*50)
    print('Downastream dataset is ',args.dataset)

    # 选择目标域并设置 downk（下游 GSL 的 kNN 超参）
    if args.dataset == 'Cora' or args.dataset =='Citeseer' or args.dataset =='Pubmed':
        dataset = Planetoid(root='data', name=args.dataset)                                                                                         
        downk=30
        if args.dataset =='Pubmed':
            testnum=100
    if args.dataset == 'Chameleon' or args.dataset == 'Squirrel':
        dataset=WikipediaNetwork(root='data',name=args.dataset)
        downk=15
    if args.dataset == 'Cornell':
        dataset=WebKB(root='data', name=args.dataset)
        downk=15

    print(args.dataset)
    loader = DataLoader(dataset)
    for data in loader:
        print(data)
        # 与上游一致的数值空间对齐（默认 PCA；ALL-IN 分支可选）-> 归一化邻接 -> 稀疏化
        features,adj= process.process_tu(data,data.x.shape[1])
        features = adapt_features_for_mdgfm(features, unify_dim, args, seed_offset=999)
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_adj = sp_adj.cuda()
        print(features.shape)
        # 测试集划分：脚本采用尾部切分（工程复现策略）
        # 若后续要和其他仓库严格对齐，可替换为固定 split 文件。
        ln=data.y.shape[0]-testnum
        if ln<0:
            ln=0
        idx_test = range(ln,data.y.shape[0])
        labels = data.y
        data=np.array(data.y)
        np.unique(data)
        nb_classes=len(np.unique(data))
        print(nb_classes)
        
    model = model.cuda()
    model.load_state_dict(torch.load(args.save_name), strict=False)
    apply_mtg_if_needed(model, args)
    if args.scgw_p1 and getattr(model, "scgw", None) is not None:
        features = apply_scgw_p1_if_needed(features, adj, model.scgw, args)
    features = torch.FloatTensor(features).cuda()

    use_ap_dual = args.use_ap and args.downstream_head == 'dual'
    use_hs_dual = args.use_hs and args.downstream_head == 'dual'
    use_graver_dual = args.use_graver and args.downstream_head == 'dual'
    # 提取目标域节点表示 Z（对应论文式 (6) 中 GE(·) 的输出 z_x）
    # MDGFM-AP/HS：每个 episode 内 prompt→GSL→重编码，不做一次性 embed
    # MDGFM-GRAVER：一次性 embed 用于 query；support 每 episode 做词汇增强
    if use_ap_dual or use_hs_dual:
        embeds = None
        for p in model.gcn.parameters():
            p.requires_grad_(False)
        model.gcn.eval()
    else:
        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None, LP)
        if use_graver_dual:
            for p in model.gcn.parameters():
                p.requires_grad_(False)
            model.gcn.eval()
    graver_vocab_bank = None
    if use_graver_dual:
        src_feats = [features1, features2, features3, features4, features5]
        src_adjs = [sp_adj1, sp_adj2, sp_adj3, sp_adj4, sp_adj5]
        src_lbls = [labels1, labels2, labels3, labels4, labels5]
        graver_vocab_bank = build_vocabulary_bank(
            src_feats,
            src_adjs,
            src_lbls,
            nb_classes,
            vocab_size=args.graver_vocab_size,
            ego_hop=args.graver_ego_hop,
            max_ego_nodes=args.graver_max_ego_nodes,
            samples_per_class=args.graver_samples_per_class,
            wildcard_per_domain=args.graver_wildcard_per_domain,
            seed=seed,
        )
        wc = graver_vocab_bank.counts[:, graver_vocab_bank.wildcard_class].sum().item()
        print(f'[MDGFM-GRAVER] vocabulary bank built (wildcard templates={int(wc)})')
    adj_dense_full = None
    if sparse and needs_adj_dense_for_homo(args):
        adj_dense_full = sp_adj.to_dense()
    acclist = torch.FloatTensor(100,).cuda()
    router_dual_n = 0
    router_dp_n = 0
    router_dual_w_sum = 0.0
    router_homo_scores = []

    for downstreamlr in [downstreamlrlist]:
        
        print(labels.shape)
        test_lbls = labels[idx_test].cuda()
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        print('-' * 100)
        for shotnum in range(shot_num,shot_num+1):
            tot = torch.zeros(1)
            tot = tot.cuda()
            accs = []
            cnt_wait = 0
            best = 1e9
            best_t = 0
            print("shotnum",shotnum)
            n_eval = max(1, int(args.eval_episodes))
            dataset_use_dual = None
            dataset_mean_homo = None
            if (
                args.use_homo_router
                and args.homo_router_scope == 'dataset'
                and not args.homo_router_soft
            ):
                prescan = prescan_episode_homo_scores(
                    args, shotnum, n_eval, adj_dense_full, labels
                )
                dataset_mean_homo = sum(prescan) / max(len(prescan), 1)
                dataset_use_dual = resolve_dataset_scope_dual(args, dataset_mean_homo)
                dom_gate = ''
                if homo_dual_domain_allowlist(args) is not None:
                    dom_gate = ' domain_dual={}'.format(domain_allows_dual(args))
                branch = 'dual' if dataset_use_dual else 'downprompt'
                print(
                    '[Homo router dataset] mean_homo={:.4f} over {} episodes -> all {} '
                    '(thresh={:.2f}{})'.format(
                        dataset_mean_homo, len(prescan), branch,
                        args.homo_bypass_thresh, dom_gate,
                    ),
                    flush=True,
                )
            # few-shot episodes（默认 50；快筛可用 --eval_episodes 10）
            for i in tqdm(range(n_eval)):
                idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()
                train_lbls = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()
                if not use_ap_dual and not use_hs_dual:
                    pretrain_embs = embeds[0, idx_train]
                    test_embs = embeds[0, idx_test]
                else:
                    pretrain_embs = None
                    test_embs = None
                global_test_embs = embeds[0, idx_test] if use_graver_dual else None
                global_support_embs = embeds[0, idx_train] if use_graver_dual else None
                homo_score = None
                if needs_episode_homo_score(args):
                    homo_score = episode_homo_score(
                        adj_dense_full, labels, idx_train, train_lbls, True
                    )
                    if homo_score is None:
                        homo_score = train_lbls.new_tensor(0.5, dtype=torch.float32)
                # F2 E1/E2/E3 on Dual: prepare separate embeds (do not mutate downprompt path).
                dual_pretrain_embs = pretrain_embs
                dual_test_embs = test_embs
                dual_in_dim = hid_units
                from models.f2_downstream_plugins import (
                    apply_f2_dual_plugins,
                    f2_plugins_need_adj,
                    gee_on_dual,
                    leaky_on_dual,
                    node_w_on_dual,
                )
                if (
                    embeds is not None
                    and adj_dense_full is not None
                    and f2_plugins_need_adj(args)
                    and (gee_on_dual(args) or leaky_on_dual(args) or node_w_on_dual(args))
                ):
                    dual_pretrain_embs, dual_test_embs, dual_in_dim = apply_f2_dual_plugins(
                        args, embeds[0], adj_dense_full, labels, idx_train, idx_test,
                        nb_classes, homo_score,
                    )
                dual_w = None
                if args.use_homo_router and args.homo_router_soft:
                    dual_w = episode_dual_weight(args, homo_score)
                    use_dual_episode = dual_w >= 1.0 - _HOMO_SOFT_EPS
                elif (
                    args.use_homo_router
                    and args.homo_router_scope == 'dataset'
                    and not args.homo_router_soft
                    and dataset_use_dual is not None
                ):
                    use_dual_episode = dataset_use_dual
                else:
                    use_dual_episode = resolve_dual_episode(args, homo_score)
                soft_blend = (
                    args.use_homo_router
                    and args.homo_router_soft
                    and domain_allows_dual(args)
                )
                if args.use_homo_router:
                    router_homo_scores.append(float(homo_score.reshape(())))
                    if i < 3:
                        dom_gate = ''
                        if homo_dual_domain_allowlist(args) is not None:
                            dom_gate = ' domain_dual={}'.format(domain_allows_dual(args))
                        if args.use_uniprop:
                            branch = 'dual' if use_dual_episode else 'downprompt'
                            print(
                                '[UniProp] episode={} homo={:.4f} -> {} (thresh={:.2f}{})'.format(
                                    i, float(homo_score), branch,
                                    args.homo_bypass_thresh, dom_gate,
                                )
                            )
                        elif soft_blend:
                            print(
                                '[Homo router soft] episode={} homo={:.4f} -> w_dual={:.4f} '
                                '(thresh={:.2f}, temp={:.3f}{})'.format(
                                    i, float(homo_score), dual_w,
                                    args.homo_bypass_thresh, args.homo_soft_temp, dom_gate,
                                )
                            )
                        elif (
                            args.homo_router_scope == 'dataset'
                            and dataset_mean_homo is not None
                        ):
                            branch = 'dual' if use_dual_episode else 'downprompt'
                            if i < 3:
                                print(
                                    '[Homo router dataset] episode={} homo={:.4f} -> {} '
                                    '(fixed; mean={:.4f}, thresh={:.2f}{})'.format(
                                        i, float(homo_score), branch, dataset_mean_homo,
                                        args.homo_bypass_thresh, dom_gate,
                                    )
                                )
                        else:
                            branch = 'dual' if use_dual_episode else 'downprompt'
                            print(
                                '[Homo router] episode={} homo={:.4f} -> {} (thresh={:.2f}{})'.format(
                                    i, float(homo_score), branch, args.homo_bypass_thresh, dom_gate
                                )
                            )
                    if soft_blend:
                        router_dual_w_sum += dual_w
                    elif use_dual_episode:
                        router_dual_n += 1
                    else:
                        router_dp_n += 1
                best = 1e9
                pat_steps = 0
                best_acc = torch.zeros(1)
                best_acc = best_acc.cuda()

                if soft_blend:
                    logits_dp = None
                    logits_dual = None
                    if dual_w < 1.0 - _HOMO_SOFT_EPS:
                        logits_dp = run_downprompt_episode(
                            args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                            unify_dim, idx_train, idx_test, pretrain_embs, test_embs, train_lbls,
                            homo_score, downstreamlr,
                        )
                    if dual_w > _HOMO_SOFT_EPS:
                        if dual_uses_reencode(args):
                            logits_dual = run_dual_adapted_episode(
                                args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                                unify_dim, idx_train, idx_test, train_lbls, homo_score, downstreamlr,
                                seed, i, shotnum,
                            )
                        else:
                            logits_dual = run_standard_dual_episode(
                                args, dual_pretrain_embs, dual_test_embs, train_lbls, homo_score, downstreamlr,
                                dual_in_dim, nb_classes, seed, i, shotnum,
                                adj_dense=adj_dense_full,
                                full_embeds=embeds[0] if embeds is not None else None,
                                idx_train=idx_train,
                            )
                    if dual_w <= _HOMO_SOFT_EPS:
                        logits = logits_dp
                    elif dual_w >= 1.0 - _HOMO_SOFT_EPS:
                        logits = logits_dual
                    else:
                        logits = blend_homo_router_logits(logits_dp, logits_dual, dual_w)
                elif use_dual_episode:
                    if use_graver_dual:
                        logits = run_dual_graver_episode(
                            args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                            unify_dim, idx_train, idx_test, train_lbls, homo_score, downstreamlr,
                            seed, i, shotnum, graver_vocab_bank, global_test_embs,
                            global_support_embs=global_support_embs,
                        )
                    elif use_hs_dual:
                        logits = run_dual_hs_episode(
                            args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                            unify_dim, idx_train, idx_test, train_lbls, homo_score, downstreamlr,
                            seed, i, shotnum,
                        )
                    elif use_ap_dual:
                        logits = run_dual_ap_episode(
                            args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                            unify_dim, idx_train, idx_test, train_lbls, homo_score, downstreamlr,
                            seed, i, shotnum,
                        )
                    elif dual_uses_reencode(args):
                        logits = run_dual_adapted_episode(
                            args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                            unify_dim, idx_train, idx_test, train_lbls, homo_score, downstreamlr,
                            seed, i, shotnum,
                        )
                    else:
                        # Prefer shared Dual helper (supports F2 E1/E2/E3 dim changes).
                        logits = run_standard_dual_episode(
                            args, dual_pretrain_embs, dual_test_embs, train_lbls, homo_score, downstreamlr,
                            dual_in_dim, nb_classes, seed, i, shotnum,
                            adj_dense=adj_dense_full,
                            full_embeds=embeds[0] if embeds is not None else None,
                            idx_train=idx_train,
                        )
                else:
                    logits = run_downprompt_episode(
                        args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                        unify_dim, idx_train, idx_test, pretrain_embs, test_embs, train_lbls,
                        homo_score, downstreamlr,
                    )
                preds = torch.argmax(logits, dim=1).cuda()
                if args.lp_refine:
                    idx_test_t = torch.tensor(list(idx_test), device=sp_adj.device, dtype=torch.long)
                    adj_dense_test = sp_adj.to_dense().index_select(0, idx_test_t).index_select(1, idx_test_t)
                    probs_refined = lp_refine_probs(
                        logits,
                        adj_dense_test,
                        beta=args.lp_beta,
                        steps=args.lp_steps,
                    )
                    preds = torch.argmax(probs_refined, dim=1).cuda()
                # episode 精度：query/test 节点上的分类正确率
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                accs.append(acc * 100)
                print('acc:[{:.4f}]'.format(acc))
                tot += acc
            print('-' * 100)
            print('Average accuracy:[{:.4f}]'.format(tot.item() / n_eval))
            accs = torch.stack(accs)
            print('Mean:[{:.4f}]'.format(accs.mean().item()))
            print('Std :[{:.4f}]'.format(accs.std().item()))
            if args.use_homo_router:
                if router_homo_scores:
                    hs = torch.tensor(router_homo_scores)
                    print(
                        'Homo scores: min={:.4f} mean={:.4f} max={:.4f}'.format(
                            hs.min().item(), hs.mean().item(), hs.max().item()
                        )
                    )
                dom_note = ''
                allow = homo_dual_domain_allowlist(args)
                if allow is not None:
                    dom_note = ' dual_domains={}'.format(','.join(sorted(allow)))
                if args.homo_router_soft:
                    n_ep = len(router_homo_scores)
                    mean_w = router_dual_w_sum / max(n_ep, 1)
                    print(
                        'Homo router soft: mean w_dual={:.3f} (thresh={:.2f}, temp={:.3f}{})'.format(
                            mean_w, args.homo_bypass_thresh, args.homo_soft_temp, dom_note,
                        )
                    )
                else:
                    label = 'UniProp' if args.use_uniprop else 'Homo router'
                    if args.homo_router_scope == 'dataset' and not args.homo_router_soft:
                        label = 'Homo router dataset'
                    print(
                        '{} episodes: dual={} downprompt={} (thresh={:.2f}{})'.format(
                            label, router_dual_n, router_dp_n, args.homo_bypass_thresh, dom_note,
                        )
                    )
                router_homo_scores = []
                router_dual_n = 0
                router_dp_n = 0
                router_dual_w_sum = 0.0
            print('-' * 100)
            row = ['Final:',"lr",lr,"downstreamlr",downstreamlr,"nb_epochs",nb_epochs,hid_units,accs.mean().item(),accs.std().item()]
            tag = "_{}".format(args.result_tag) if args.result_tag else ""
            out = open("data/ICML25_{}{}_fewshot.csv".format(args.dataset.lower(), tag), "a", newline="")
            csv_writer = csv.writer(out, dialect="excel")
            # 结果落盘：每次实验追加一行（便于后续汇总）
            csv_writer.writerow(row)
