"""SpecGFM pre-training (BandGSL) and homophily-guided dual-branch few-shot evaluation."""

from __future__ import annotations

import sys
import numpy as np
import scipy.sparse as sp
import random
import time
from preprompt import PrePrompt, pca_compression
import preprompt as preprompt
from scgw_utils import maybe_create_scgw
from utils import process
import os
CKPT_DIR = 'checkpoints'


def _resolve_ckpt(name: str) -> str:
    if os.path.isabs(name):
        return name
    return os.path.join(CKPT_DIR, os.path.basename(name))


import argparse
from downprompt_bikt import downprompt_bikt
import csv
from tqdm import tqdm














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
parser.add_argument("--f2_gee_branch", type=str, default="off", choices=["off", "dual"],
                    help="support-GEE. Paper full model uses dual (homophilic branch only).")

args = parser.parse_args()
if args.scgw_p4 and not args.use_band_gsl:
    raise ValueError("--scgw_p4 requires --use_band_gsl")

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
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from torch_geometric.loader import DataLoader
print('-' * 100)
nb_epochs = args.epochs
patience = 500
lr_list=args.lr
l2_coef = 0.0001
hid_units = 256
sparse = True
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
xent = nn.CrossEntropyLoss()
unify_dim = 50


def adapt_features_for_mdgfm(raw_features, unify_dim, args, seed_offset=0):
    """PCA projection used before pre-training and downstream evaluation."""
    del args, seed_offset
    return pca_compression(raw_features, k=unify_dim)


def create_scgw_module(args, feat_dim: int):
    mod = maybe_create_scgw(args, feat_dim)
    if mod is not None and torch.cuda.is_available():
        mod = mod.cuda()
    return mod




def preprompt_ctor_kwargs(args, scgw_module):
    return dict(
        dpc_temp=args.dpc_temp,
        use_band_gsl=args.use_band_gsl,
        band_init_alpha=args.band_init_alpha,
        band_v2_adaptive_gate=args.band_v2_adaptive_gate,
        band_v2_cross_domain=args.band_v2_cross_domain,
        band_trust_low=args.band_trust_low,
        band_gate_clamp=args.band_gate_clamp,
        band_cd_weight=args.band_cd_weight,
        use_feat_dv_inv=args.use_feat_dv_inv,
        feat_dv_weight=args.feat_dv_weight,
        feat_dv_dropout=args.feat_dv_dropout,
        use_scgw_p4=args.scgw_p4,
        scgw_module=scgw_module,
        scgw_weight=args.scgw_weight,
    )








def prototype_loss(embeds: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """L2 distance from sample embeddings to their class prototypes."""
    target_proto = prototypes[labels]
    return ((embeds - target_proto) ** 2).mean()


def cosine_proto_logits(query_embeds: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    q = F.normalize(query_embeds, dim=1)
    p = F.normalize(prototypes, dim=1)
    return torch.mm(q, p.t())








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








def build_downstream_log(args, model, hid_units, nb_classes, unify_dim):
    """Heterophilic branch: BiKT dual-view head on frozen pre-trained tokens."""
    kw = _downprompt_token_kwargs(model)
    return downprompt_bikt(
        **kw,
        ft_in=hid_units,
        nb_classes=nb_classes,
        type=args.combinetype,
        feature_dim=unify_dim,
        bikt_weight=args.bikt_weight,
    ).cuda()


def needs_episode_homo_score(args) -> bool:
    return args.use_homo_router or args.use_bikt


def needs_adj_dense_for_homo(args) -> bool:
    from models.f2_downstream_plugins import f2_plugins_need_adj
    return args.use_homo_router or args.use_bikt or f2_plugins_need_adj(args)


def resolve_dual_episode(args, homo_score) -> bool:
    """True -> homophilic dual head; False -> heterophilic BiKT head."""
    if not args.use_homo_router:
        return True
    from models.branch_utils import homophilic_episode
    if homo_score is None:
        return True
    return homophilic_episode(homo_score, args.homo_bypass_thresh)


def configure_downstream_episode(log, args, homo_score, idx_train):
    """Pass h_e into the heterophilic branch."""
    if args.use_bikt:
        log.set_episode_context(
            homo_score,
            idx_train,
            bikt_weight=float(args.bikt_weight),
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
    """Train the heterophilic BiKT head for one episode; return test logits."""
    episode_gcn = model.gcn
    log = build_downstream_log(args, model, hid_units, nb_classes, unify_dim)
    opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
    configure_downstream_episode(log, args, homo_score, idx_train)
    from models.f2_downstream_plugins import f2_plugins_need_adj
    if f2_plugins_need_adj(args) and hasattr(log, "set_f2_plugin_graph"):
        adj_d = sp_adj.to_dense() if sparse else (
            sp_adj if isinstance(sp_adj, torch.Tensor) else torch.as_tensor(sp_adj)
        )
        log.set_f2_plugin_graph(args, adj_d, train_lbls)
    return_aux = bool(args.use_bikt or args.proto_weight > 0)
    best = 1e9
    cnt_wait = 0
    for _step in range(400):
        log.train()
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
    return log(features, sp_adj, sparse, episode_gcn, idx_test, test_embs, downk)




def run_standard_dual_episode(
    args,
    pretrain_embs,
    test_embs,
    train_lbls,
    downstreamlr,
    nb_classes,
    seed,
    episode_i,
    shotnum,
):
    """Homophilic branch: frozen embeddings, averaged linear heads, prototype head."""
    k_heads = max(1, int(args.dual_ensemble))
    in_dim = int(pretrain_embs.shape[1])
    logits_accum = None
    ens_base = int(seed) + episode_i * 100003 + int(shotnum) * 9973
    for k_m in range(k_heads):
        torch.manual_seed(ens_base + k_m * 1_000_003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ens_base + k_m * 1_000_003)
        dual_head = nn.Linear(in_dim, nb_classes).cuda()
        opt = torch.optim.Adam(dual_head.parameters(), lr=downstreamlr)
        for _ in range(args.dual_epochs):
            dual_head.train()
            opt.zero_grad()
            logits_linear = dual_head(pretrain_embs)
            proto = build_prototypes(pretrain_embs, train_lbls, nb_classes)
            logits_proto = cosine_proto_logits(pretrain_embs, proto)
            logits_mix = args.dual_alpha * logits_linear + (1.0 - args.dual_alpha) * logits_proto
            loss = xent(logits_mix, train_lbls)
            if args.proto_weight > 0:
                loss = loss + args.proto_weight * prototype_loss(pretrain_embs, train_lbls, proto)
            loss.backward()
            opt.step()
        dual_head.eval()
        with torch.no_grad():
            logits_linear_test = dual_head(test_embs)
            proto_test = build_prototypes(pretrain_embs, train_lbls, nb_classes)
            logits_proto_test = cosine_proto_logits(test_embs, proto_test)
            logits_k = args.dual_alpha * logits_linear_test + (1.0 - args.dual_alpha) * logits_proto_test
            logits_accum = logits_k if logits_accum is None else logits_accum + logits_k
    return logits_accum / float(k_heads)














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
        args.save_name = _resolve_ckpt(time.strftime("%Y%m%d_%H%M%S_", time_) + a)
    if args.load_pretrained:
        # Downstream-only: skip source-domain PCA / pretrain graph prep.
        if not os.path.isfile(args.load_pretrained):
            raise FileNotFoundError(
                '[load_pretrained] checkpoint not found: {}'.format(args.load_pretrained)
            )
        features1 = features2 = features3 = features4 = features5 = None
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
                print('[pretrain] structural coordinate bases enabled', flush=True)

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
            loss = base_loss + args.dpc_weight * dpc_loss
            loss.backward()
            optimiser.step()
            if args.pretrain_ema and ema_shadow is not None:
                d_ema = args.pretrain_ema_decay
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        ema_shadow[n].mul_(d_ema).add_(p.data, alpha=1.0 - d_ema)
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
    features = torch.FloatTensor(features).cuda()

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None, LP)
    adj_dense_full = None
    if sparse and needs_adj_dense_for_homo(args):
        adj_dense_full = sp_adj.to_dense()
    router_dual_n = 0
    router_dp_n = 0
    router_homo_scores = []

    for downstreamlr in [downstreamlrlist]:
        print(labels.shape)
        test_lbls = labels[idx_test].cuda()
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        print('-' * 100)
        for shotnum in range(shot_num, shot_num + 1):
            tot = torch.zeros(1)
            tot = tot.cuda()
            accs = []
            print("shotnum", shotnum)
            n_eval = max(1, int(args.eval_episodes))
            for i in tqdm(range(n_eval)):
                idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i)).type(torch.long).cuda()
                train_lbls = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i)).type(torch.long).squeeze().cuda()
                pretrain_embs = embeds[0, idx_train]
                test_embs = embeds[0, idx_test]
                homo_score = None
                if needs_episode_homo_score(args):
                    homo_score = episode_homo_score(
                        adj_dense_full, labels, idx_train, train_lbls, True
                    )
                    if homo_score is None:
                        homo_score = train_lbls.new_tensor(0.5, dtype=torch.float32)
                dual_pretrain_embs = pretrain_embs
                dual_test_embs = test_embs
                from models.f2_downstream_plugins import apply_f2_dual_plugins, f2_plugins_need_adj, gee_on_dual
                if (
                    adj_dense_full is not None
                    and f2_plugins_need_adj(args)
                    and gee_on_dual(args)
                ):
                    dual_pretrain_embs, dual_test_embs, _dual_in_dim = apply_f2_dual_plugins(
                        args, embeds[0], adj_dense_full, labels, idx_train, idx_test,
                        nb_classes, homo_score,
                    )
                use_dual_episode = resolve_dual_episode(args, homo_score)
                if args.use_homo_router:
                    router_homo_scores.append(float(homo_score.reshape(())))
                    branch = 'homophilic' if use_dual_episode else 'heterophilic'
                    if i < 3:
                        print(
                            '[Homo router] episode={} homo={:.4f} -> {} (thresh={:.2f})'.format(
                                i, float(homo_score), branch, args.homo_bypass_thresh,
                            )
                        )
                    if use_dual_episode:
                        router_dual_n += 1
                    else:
                        router_dp_n += 1
                if use_dual_episode:
                    logits = run_standard_dual_episode(
                        args, dual_pretrain_embs, dual_test_embs, train_lbls, downstreamlr,
                        nb_classes, seed, i, shotnum,
                    )
                else:
                    logits = run_downprompt_episode(
                        args, model, features, sp_adj, sparse, downk, hid_units, nb_classes,
                        unify_dim, idx_train, idx_test, pretrain_embs, test_embs, train_lbls,
                        homo_score, downstreamlr,
                    )
                preds = torch.argmax(logits, dim=1).cuda()
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
                print(
                    'Homo router episodes: homophilic={} heterophilic={} (thresh={:.2f})'.format(
                        router_dual_n, router_dp_n, args.homo_bypass_thresh,
                    )
                )
                router_homo_scores = []
                router_dual_n = 0
                router_dp_n = 0
            print('-' * 100)
            row = ['Final:', "lr", lr, "downstreamlr", downstreamlr, "nb_epochs", nb_epochs, hid_units, accs.mean().item(), accs.std().item()]
            tag = "_{}".format(args.result_tag) if args.result_tag else ""
            out = open("data/ICML25_{}{}_fewshot.csv".format(args.dataset.lower(), tag), "a", newline="")
            csv_writer = csv.writer(out, dialect="excel")
            csv_writer.writerow(row)
