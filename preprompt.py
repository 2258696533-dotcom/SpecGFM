"""Pre-training: BandGSL, domain tokens, and contrastive topology alignment."""


import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers
from models.scale_encoder import build_gcn_encoder, resolve_scale_encoder
from layers import GCN, AvgReadout
import tqdm
import numpy as np
import dgl
from utils import Calbound
from sklearn.decomposition import PCA
from layers import Attentivemod
from tools import *


class ATT_learner(nn.Module):
    """图结构学习（GSL）子模块的骨干 MLP（逐维 Attentive）。

    `graph_process` 输出论文中的 refined 邻接 A'_i：相似度/kNN + 对称 + 激活 + 归一化（稠密情形），
    或基于 `knn_fast` 的局部近似（稀疏情形，和论文 Appendix 中“局部敏感近似”思路一致）。
    """
    def __init__(self, nlayers, isize, i, dropedge_rate, sparse, act):
        super(ATT_learner, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentivemod.Attentive(isize))

        self.non_linearity = 'relu'
        self.i = i
        self.sparse = sparse
        self.act = act
        self.dropedge_rate = dropedge_rate

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)

        return h

    def forward(self, features):
       
        embeddings = self.internal_forward(features)

        return embeddings

    def graph_process(self, k, embeddings):
        # 由 Hi 构造 A'_i：kNN 稀疏化 + 对称 + 非线性 +（对称）归一化；训练期带 dropout 增强稳健性。
        if self.sparse:
            rows, cols, values = knn_fast(embeddings, k, 1000)      
            values[torch.isnan(values)] = 0  
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            values_ = F.dropout(values_, p=self.dropedge_rate, training=self.training)

            num_nodes = embeddings.shape[0]
            learned_adj = torch.zeros((num_nodes, num_nodes), device='cuda') 
            learned_adj[rows_, cols_] = values_ 

            return learned_adj
        else:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, k + 1)
            similarities = symmetrize(similarities)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            learned_adj = normalize(similarities, 'sym')
            learned_adj = F.dropout(learned_adj, p=self.dropedge_rate, training=self.training)

            return learned_adj



class BandGSL(nn.Module):
    """Low/high-frequency graph structure learner with learnable fusion."""

    def __init__(
        self,
        nlayers,
        isize,
        i,
        dropedge_rate,
        sparse,
        act,
        init_alpha=0.5,
        adaptive_gate=False,
        trust_low=0.0,
        gate_clamp=False,
    ):
        super(BandGSL, self).__init__()
        self.low_learner = ATT_learner(nlayers, isize, i, dropedge_rate, sparse, act)
        self.sparse = sparse
        self.non_linearity = 'relu'
        self.i = i
        self.dropedge_rate = dropedge_rate
        self.adaptive_gate = adaptive_gate
        self.trust_low = float(max(0.0, min(trust_low, 1.0)))
        self.gate_clamp = gate_clamp
        p = float(max(min(init_alpha, 1.0 - 1e-4), 1e-4))
        self.alpha_raw = nn.Parameter(torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32))
        if self.adaptive_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    def _build_high_adj(self, embeddings: torch.Tensor, low_adj: torch.Tensor, k: int):
        # Laplacian residual branch: emphasize high-frequency components.
        residual = embeddings - torch.mm(low_adj, embeddings)
        res_norm = F.normalize(residual, dim=1, p=2)
        similarities = cal_similarity_graph(res_norm)
        similarities = top_k(similarities, k + 1)
        similarities = symmetrize(similarities)
        similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
        high_adj = normalize(similarities, 'sym')
        high_adj = F.dropout(high_adj, p=self.dropedge_rate, training=self.training)
        return high_adj, residual

    def _fuse(self, low_adj: torch.Tensor, high_adj: torch.Tensor, embeddings: torch.Tensor, residual: torch.Tensor):
        if self.adaptive_gate:
            stats = torch.stack(
                [
                    embeddings.norm(dim=1),
                    residual.norm(dim=1),
                    low_adj.sum(dim=1),
                ],
                dim=1,
            )
            node_gate = torch.sigmoid(self.gate_mlp(stats)).squeeze(-1)  # [N]
            if self.gate_clamp:
                # Keep a floor/ceiling on low-freq weight to avoid collapsing to pure high-freq.
                node_gate = 0.15 + 0.70 * node_gate
            gate_mat = 0.5 * (node_gate.unsqueeze(1) + node_gate.unsqueeze(0))
            refined = gate_mat * low_adj + (1.0 - gate_mat) * high_adj
            gate_summary = node_gate.mean()
            return refined, gate_summary, node_gate
        alpha = torch.sigmoid(self.alpha_raw)
        refined = alpha * low_adj + (1.0 - alpha) * high_adj
        return refined, alpha, None

    def graph_process(self, k, embeddings):
        refined, _, _, _, _ = self.graph_process_with_views(k, embeddings)
        return refined

    def graph_process_with_bands(self, k, embeddings):
        # Low branch: original local-similarity topology.
        low_adj = self.low_learner.graph_process(k, embeddings)
        # High branch: Laplacian residual topology (I - A_low)X.
        high_adj, residual = self._build_high_adj(embeddings, low_adj, k)
        refined, gate_summary, node_gate = self._fuse(low_adj, high_adj, embeddings, residual)
        if self.trust_low > 0:
            refined = (1.0 - self.trust_low) * refined + self.trust_low * low_adj
        low_view = torch.mm(low_adj, embeddings)
        high_view = residual
        return refined, low_adj, high_adj, low_view, high_view, gate_summary, node_gate

    def graph_process_with_views(self, k, embeddings):
        refined, _, _, low_view, high_view, gate_summary, node_gate = self.graph_process_with_bands(
            k, embeddings
        )
        return refined, low_view, high_view, gate_summary, node_gate


class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        

        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


def matrixsquare(matrix):
    if matrix.is_sparse:
        square = torch.sparse.mm(matrix, matrix)
    else:
        square = torch.mm(matrix, matrix)
    return square


class FiLMPrompt(nn.Module):
    """Node-wise FiLM. `replace`: full gamma/beta map; `delta`: bounded residual on h."""

    def __init__(self, dim: int, mode: str = "replace"):
        super().__init__()
        self.mode = mode
        hid = max(64, min(256, int(dim))) if mode == "delta" else max(128, min(512, int(dim)))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 2 * dim),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def delta(self, h: torch.Tensor) -> torch.Tensor:
        """Small perturbation: tanh(g)*h + tanh(b); scaled outside by film_residual_scale."""
        gb = self.mlp(h)
        gamma, beta = gb.chunk(2, dim=-1)
        return torch.tanh(gamma) * h + torch.tanh(beta)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "delta":
            return self.delta(h)
        gb = self.mlp(h)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + 0.15 * torch.tanh(gamma)
        return gamma * h + 0.15 * torch.tanh(beta)


class PrePrompt(nn.Module):
    """多源域预训练主体：Prompt + GSL + 拓扑对齐损失（论文式 (2)(3)(4) 的工程实现）。

    - 每个源域一套 `pretext*`（域 token）+ 共享 `sumtext`（共享 token）。
    - `balancetoken*`：对拼接特征 [X', A X'] 的 balance token（式 (3) 的 t_B）。
    - `learner`：生成 refined 图 A'（式 (3) 之后、式 (4) 之前的 GSL）。
    - `negative_sample`：由 `prompt_pretrain_sample` 生成；在本 forward 中未直接使用，但保留接口/历史实验兼容。
    """
    def __init__(
        self,
        n_in,
        n_h,
        activation,
        sample,
        num_layers_num,
        p,
        type,
        use_domain_gate=False,
        domain_gate_gamma=0.3,
        dpc_temp=0.2,
        use_band_gsl=False,
        band_init_alpha=0.5,
        band_v2_adaptive_gate=False,
        band_v2_cross_domain=False,
        band_trust_low=0.0,
        band_gate_clamp=False,
        band_cd_weight=0.1,
        band_deg_anchor_weight=0.0,
        band_gate_entropy_weight=0.0,
        use_graph_mae=False,
        mae_weight=0.25,
        mae_mask_ratio=0.2,
        use_feat_dv_inv=False,
        feat_dv_weight=0.15,
        feat_dv_dropout=0.25,
        use_film_prompt=False,
        use_film_residual=False,
        film_residual_scale=0.15,
        use_gcil=False,
        gcil_weight=0.1,
        use_gcil_spectral=False,
        gcil_inv_weight=1.0,
        gcil_indep_weight=0.1,
        use_scale_gnn=False,
        scale_gnn_hops=3,
        scale_encoder="none",
        scale_residual_gamma=0.15,
        use_hybrid_spectral=False,
        use_scgw_p2=False,
        use_scgw_p4=False,
        scgw_module=None,
        scgw_weight=0.1,
    ):
        super(PrePrompt, self).__init__()
        self.lp = Lp(n_in, n_h)
        enc = resolve_scale_encoder(use_scale_gnn, scale_encoder)
        self.gcn = build_gcn_encoder(
            n_in,
            n_h,
            num_layers_num,
            p,
            scale_encoder=enc,
            num_hops=scale_gnn_hops,
            scale_residual_gamma=scale_residual_gamma,
        )
        self.read = AvgReadout()
        self.prompttype = type

        self.pretext1 = textprompt(n_in,type)
        self.pretext2 = textprompt(n_in,type)
        self.pretext3 = textprompt(n_in,type)
        self.pretext4 = textprompt(n_in,type)
        self.pretext5 = textprompt(n_in,type)

        
        self.sumtext = textprompt(n_in,type)

        self.texttoken1 = textprompt(n_h,type)
        self.texttoken2 = textprompt(n_h,type)
        self.texttoken3 = textprompt(n_h,type)
        self.texttoken4 = textprompt(n_h,type)
        self.texttoken5 = textprompt(n_h,type)


        self.balancetoken1 = textprompt(2*n_in,type)
        self.balancetoken2 = textprompt(2*n_in,type)
        self.balancetoken3 = textprompt(2*n_in,type)
        self.balancetoken4 = textprompt(2*n_in,type)
        self.balancetoken5 = textprompt(2*n_in,type)
       
        self.use_band_gsl = use_band_gsl
        self.band_v2_adaptive_gate = band_v2_adaptive_gate
        self.band_v2_cross_domain = band_v2_cross_domain
        self.band_cd_weight = float(band_cd_weight)
        self.band_deg_anchor_weight = float(band_deg_anchor_weight)
        self.band_gate_entropy_weight = float(band_gate_entropy_weight)
        if self.use_band_gsl:
            self.learner = BandGSL(
                2,
                50,
                6,
                0.5,
                sparse=True,
                act='relu',
                init_alpha=band_init_alpha,
                adaptive_gate=band_v2_adaptive_gate,
                trust_low=band_trust_low,
                gate_clamp=band_gate_clamp,
            )
        else:
            self.learner = ATT_learner(2, 50, 6, 0.5, sparse = True, act = 'relu')

        self.negative_sample = torch.tensor(sample,dtype=int).cuda()
        self.loss = nn.BCEWithLogitsLoss()
        self.use_domain_gate = use_domain_gate
        self.domain_gate_gamma = domain_gate_gamma
        self.dpc_temp = dpc_temp
        if self.use_domain_gate:
            self.domain_gate_mlp = nn.Sequential(
                nn.Linear(n_in, max(16, n_in // 2)),
                nn.ReLU(),
                nn.Linear(max(16, n_in // 2), 1),
            )

        self.use_graph_mae = use_graph_mae
        self.mae_weight = float(mae_weight)
        self.mae_mask_ratio = float(min(max(mae_mask_ratio, 0.01), 0.9))
        if self.use_graph_mae:
            self.mae_decoder = nn.Linear(n_h, n_in)
            self.mae_mask_token = nn.Parameter(torch.zeros(1, n_in))
            nn.init.xavier_uniform_(self.mae_decoder.weight)
            nn.init.zeros_(self.mae_decoder.bias)

        self.use_feat_dv_inv = use_feat_dv_inv
        self.feat_dv_weight = float(feat_dv_weight)
        self.feat_dv_dropout = float(min(max(feat_dv_dropout, 0.05), 0.9))
        self.use_film_prompt = bool(use_film_prompt)
        self.use_film_residual = bool(use_film_residual)
        self.film_residual_scale = float(max(0.0, film_residual_scale))
        if self.use_film_prompt and self.use_film_residual:
            raise ValueError("use_film_prompt (replace) and use_film_residual are mutually exclusive")
        if self.use_film_prompt:
            self.film1 = FiLMPrompt(n_in, mode="replace")
            self.film2 = FiLMPrompt(n_in, mode="replace")
            self.film3 = FiLMPrompt(n_in, mode="replace")
            self.film4 = FiLMPrompt(n_in, mode="replace")
            self.film5 = FiLMPrompt(n_in, mode="replace")
        elif self.use_film_residual:
            self.film1 = FiLMPrompt(n_in, mode="delta")
            self.film2 = FiLMPrompt(n_in, mode="delta")
            self.film3 = FiLMPrompt(n_in, mode="delta")
            self.film4 = FiLMPrompt(n_in, mode="delta")
            self.film5 = FiLMPrompt(n_in, mode="delta")

        self.use_gcil = bool(use_gcil)
        self.gcil_weight = float(gcil_weight)
        self.use_gcil_spectral = bool(use_gcil_spectral)
        self.gcil_inv_weight = float(gcil_inv_weight)
        self.gcil_indep_weight = float(gcil_indep_weight)
        self.use_scale_gnn = enc != "none"
        self.use_hybrid_spectral = bool(use_hybrid_spectral)
        if self.use_hybrid_spectral and not self.use_band_gsl:
            raise ValueError("use_hybrid_spectral requires use_band_gsl")
        if self.use_hybrid_spectral:
            from models.hybrid_spectral_encoder import HybridSpectralFusion
            self.band_fusion = HybridSpectralFusion(num_bands=2, init_equal=True)
        self.scale_encoder = enc
        self.use_scgw_p2 = bool(use_scgw_p2)
        self.use_scgw_p4 = bool(use_scgw_p4)
        self.scgw_weight = float(scgw_weight)
        if use_scgw_p4 and not self.use_band_gsl:
            raise ValueError("use_scgw_p4 requires use_band_gsl")
        if scgw_module is not None:
            self.scgw = scgw_module
        else:
            self.scgw = None
        if (self.use_scgw_p2 or self.use_scgw_p4) and self.scgw is None:
            raise ValueError("scgw_module required when use_scgw_p2 or use_scgw_p4 is enabled")

    def _gcil_on_domain(self, z_refined, z_orig, adj, preseq, sparse):
        raise ValueError("GCIL is not included in this SpecGFM release")
        from gcil_utils import apply_lowpass, gcil_pair_loss
        """GCIL invariance/independence between refined-graph and original-graph views."""
        loss = gcil_pair_loss(
            z_refined,
            z_orig,
            inv_weight=self.gcil_inv_weight,
            indep_weight=self.gcil_indep_weight,
        )
        if self.use_gcil_spectral and adj.is_sparse:
            adj_lp = apply_lowpass(adj)
            z_spec = self.lp(self.gcn, preseq, adj_lp, sparse)
            loss = loss + gcil_pair_loss(
                z_orig,
                z_spec,
                inv_weight=self.gcil_inv_weight,
                indep_weight=self.gcil_indep_weight,
            )
        return loss

    def _align_preseq_after_relu(self, preseq, film_mod=None, use_sumtext=True, use_film_replace=False, use_film_res=False):
        """Eq. (2): sumtext (optional) + optional FiLM replace or residual."""
        if use_film_replace and film_mod is not None:
            return film_mod(preseq)
        h = self.sumtext(preseq) if use_sumtext else preseq
        if use_film_res and film_mod is not None:
            h = h + self.film_residual_scale * film_mod.delta(h)
        return h

    def _hybrid_spectral_encode(self, preseq, refined_adj, low_adj, high_adj, sparse):
        """Dual-band GCN encode + learned fusion (HS-GPPT §4.1)."""
        if self.use_hybrid_spectral and low_adj is not None and high_adj is not None:
            z_low = self.lp(self.gcn, preseq, low_adj, sparse)
            z_high = self.lp(self.gcn, preseq, high_adj, sparse)
            return self.band_fusion.fuse([z_low, z_high])
        return self.lp(self.gcn, preseq, refined_adj, sparse)

    def _feat_view_inv_single(self, preseq, z_refined, refined_adj, sparse):
        """1 - cos(z, z') with feature dropout on preseq; same A'. Stabilizes tail seeds."""
        x2 = F.dropout(preseq, p=self.feat_dv_dropout, training=self.training)
        z2 = self.lp(self.gcn, x2, refined_adj, sparse)
        z1n = F.normalize(z_refined, dim=1)
        z2n = F.normalize(z2, dim=1)
        return (1.0 - (z1n * z2n).sum(dim=1)).mean()

    def _domain_proto_contrast(self, pre_logits_list, logits_list):
        """InfoNCE over per-domain prototypes: align refined/original domain representations."""
        pre_proto = torch.stack([x.mean(dim=0) for x in pre_logits_list], dim=0)
        ori_proto = torch.stack([x.mean(dim=0) for x in logits_list], dim=0)
        pre_proto = F.normalize(pre_proto, dim=1)
        ori_proto = F.normalize(ori_proto, dim=1)
        sim = torch.mm(pre_proto, ori_proto.t()) / max(self.dpc_temp, 1e-6)
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_a = F.cross_entropy(sim, labels)
        loss_b = F.cross_entropy(sim.t(), labels)
        return 0.5 * (loss_a + loss_b)

    def _band_cross_domain_loss(self, low_views, high_views):
        """Align domain prototypes between low/high frequency branches."""
        low_proto = torch.stack([x.mean(dim=0) for x in low_views], dim=0)
        high_proto = torch.stack([x.mean(dim=0) for x in high_views], dim=0)
        low_proto = F.normalize(low_proto, dim=1)
        high_proto = F.normalize(high_proto, dim=1)
        sim = torch.mm(low_proto, high_proto.t()) / max(self.dpc_temp, 1e-6)
        labels = torch.arange(sim.shape[0], device=sim.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

    def _band_degree_anchor(self, refined: torch.Tensor, adj_sp) -> torch.Tensor:
        """L1 between normalized row-sum profiles of A' and input sparse A (per domain)."""
        dr = refined.sum(dim=1)
        n = refined.size(0)
        ones = torch.ones(n, 1, device=refined.device, dtype=refined.dtype)
        adj_f = adj_sp.float() if adj_sp.dtype != refined.dtype else adj_sp
        do = torch.sparse.mm(adj_f, ones).squeeze(1)
        dr = dr / (dr.mean() + 1e-6)
        do = do / (do.mean() + 1e-6)
        return F.l1_loss(dr, do)

    def _graph_mae_loss_single(self, x, refined_adj, sparse):
        """Feature masking + decode on refined graph (Graph-MAE style, pretrain only)."""
        n, d = x.shape
        p = self.mae_mask_ratio
        mask = torch.rand(n, d, device=x.device, dtype=x.dtype) < p
        if not mask.any():
            mask[0, 0] = True
        token = self.mae_mask_token.to(dtype=x.dtype, device=x.device).expand_as(x)
        x_in = torch.where(mask, token, x)
        h = self.lp(self.gcn, x_in, refined_adj, sparse)
        x_hat = self.mae_decoder(h)
        return F.mse_loss(x_hat[mask], x.detach()[mask])

    def forward(self, seq1,seq2,seq3,seq4,seq5,adj1,adj2,adj3,adj4,adj5,
                sparse, msk, samp_bias1, samp_bias2,i, compute_dpc=True):
        """返回标量损失 lploss（对 5 个源域求和）。

        对每个域 i：
        1) 先得到 X'_i：域 token ⊙ σ(·) + 共享 token（见式 (2) 的实现对应）。
        2) 构造 Hi = t_B ⊙ [X'_i, A_i X'_i]（式 (3)）。
        3) 由 Hi 得到 A'_i（`learner.graph_process`）。
        4) 用两组表示 z：在 A_i 上与在 A'_i 上跑同一个 GCN（`Lp` 包装 `GcnLayers`），
           通过 `calc_lower_bound` 对齐（式 (4) 的两项：I_e 与 A'_i）。

        参数 i：用于在异质/同质数据集上调整 kNN 的 k（论文实验中对不同图结构常用不同局部性）。
        """

        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        seq5 = torch.squeeze(seq5,0)

        # --- 式 (2)：X'_i = t_S ⊙ σ(t_{D_i} ⊙ X_i) ---
        # 代码顺序：先逐域 token（pretext*），再 ReLU（σ），再共享 token（sumtext）。
        preseq1 = self.pretext1(seq1)
        preseq2 = self.pretext2(seq2)
        preseq3 = self.pretext3(seq3)
        preseq4 = self.pretext4(seq4)
        preseq5 = self.pretext5(seq5)

        preseq1 = F.relu(preseq1)
        preseq2 = F.relu(preseq2)
        preseq3 = F.relu(preseq3)
        preseq4 = F.relu(preseq4)
        preseq5 = F.relu(preseq5)

        preseq1 = self._align_preseq_after_relu(
            preseq1, self.film1 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq2 = self._align_preseq_after_relu(
            preseq2, self.film2 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq3 = self._align_preseq_after_relu(
            preseq3, self.film3 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq4 = self._align_preseq_after_relu(
            preseq4, self.film4 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq5 = self._align_preseq_after_relu(
            preseq5, self.film5 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        if self.use_domain_gate:
            domain_means = torch.stack(
                [
                    seq1.mean(dim=0),
                    seq2.mean(dim=0),
                    seq3.mean(dim=0),
                    seq4.mean(dim=0),
                    seq5.mean(dim=0),
                ],
                dim=0,
            )
            gate_logits = self.domain_gate_mlp(domain_means).squeeze(-1)
            gate_weights = F.softmax(gate_logits, dim=0)
            # Domains have different node counts, so mix domain-level prototypes
            # (feature vectors) instead of full node matrices.
            mixed_shared = (
                gate_weights[0] * preseq1.mean(dim=0)
                + gate_weights[1] * preseq2.mean(dim=0)
                + gate_weights[2] * preseq3.mean(dim=0)
                + gate_weights[3] * preseq4.mean(dim=0)
                + gate_weights[4] * preseq5.mean(dim=0)
            ).unsqueeze(0)
            g = float(self.domain_gate_gamma)
            preseq1 = (1.0 - g) * preseq1 + g * mixed_shared
            preseq2 = (1.0 - g) * preseq2 + g * mixed_shared
            preseq3 = (1.0 - g) * preseq3 + g * mixed_shared
            preseq4 = (1.0 - g) * preseq4 + g * mixed_shared
            preseq5 = (1.0 - g) * preseq5 + g * mixed_shared

        # --- 式 (3)：Hi = t_B ⊙ [X'_i, A_i^r X'_i] ---
        # 这里 r 默认为 1：直接做一次稀疏矩阵乘法 A_i X'_i（等价于 1-hop 聚合特征）。
        reseq1 = torch.sparse.mm(adj1,preseq1)
        reseq1 = torch.cat((preseq1, reseq1), dim = 1)
        reseq2 = torch.sparse.mm(adj2,preseq2)
        reseq2 = torch.cat((preseq2, reseq2), dim = 1)
        reseq3 = torch.sparse.mm(adj3,preseq3)
        reseq3 = torch.cat((preseq3, reseq3), dim = 1)
        reseq4 = torch.sparse.mm(adj4,preseq4)
        reseq4 = torch.cat((preseq4, reseq4), dim = 1)
        reseq5 = torch.sparse.mm(adj5,preseq5)
        reseq5 = torch.cat((preseq5, reseq5), dim = 1)

        # balancetoken* ≈ t_B：在 2d 维拼接空间上逐域调制
        reseq1 = self.balancetoken1(reseq1)
        reseq2 = self.balancetoken2(reseq2)
        reseq3 = self.balancetoken3(reseq3)
        reseq4 = self.balancetoken4(reseq4)
        reseq5 = self.balancetoken5(reseq5)


        # kNN 的 k：论文实验里对不同拓扑/规模的图会用不同局部性超参。
        pre_k=[30,30,30,15,15,15]
        pre_k[i]=15
        # A'_i：由 Hi 经 kNN-GSL 得到（论文 4.2 节）
        if self.use_band_gsl:
            refinedadj1, low_adj1, high_adj1, low1, high1, _, bg1 = self.learner.graph_process_with_bands(pre_k[0], reseq1)
            refinedadj2, low_adj2, high_adj2, low2, high2, _, bg2 = self.learner.graph_process_with_bands(pre_k[1], reseq2)
            refinedadj3, low_adj3, high_adj3, low3, high3, _, bg3 = self.learner.graph_process_with_bands(pre_k[2], reseq3)
            refinedadj4, low_adj4, high_adj4, low4, high4, _, bg4 = self.learner.graph_process_with_bands(pre_k[3], reseq4)
            refinedadj5, low_adj5, high_adj5, low5, high5, _, bg5 = self.learner.graph_process_with_bands(pre_k[4], reseq5)
            refinedadj1 = refinedadj1.to('cuda')
            refinedadj2 = refinedadj2.to('cuda')
            refinedadj3 = refinedadj3.to('cuda')
            refinedadj4 = refinedadj4.to('cuda')
            refinedadj5 = refinedadj5.to('cuda')
        else:
            refinedadj1 = self.learner.graph_process(pre_k[0], reseq1).to('cuda')
            refinedadj2 = self.learner.graph_process(pre_k[1], reseq2).to('cuda')
            refinedadj3 = self.learner.graph_process(pre_k[2], reseq3).to('cuda')
            refinedadj4 = self.learner.graph_process(pre_k[3], reseq4).to('cuda')
            refinedadj5 = self.learner.graph_process(pre_k[4], reseq5).to('cuda')
            low_adj1 = low_adj2 = low_adj3 = low_adj4 = low_adj5 = None
            high_adj1 = high_adj2 = high_adj3 = high_adj4 = high_adj5 = None
            low1 = high1 = low2 = high2 = low3 = high3 = low4 = high4 = low5 = high5 = None
            bg1 = bg2 = bg3 = bg4 = bg5 = None

        num1, _ = refinedadj1.size()
        num2, _ = refinedadj2.size()
        num3, _ = refinedadj3.size()
        num4, _ = refinedadj4.size()
        num5, _ = refinedadj5.size()

        # 正样本关系矩阵 pos：
        # - I_e（单位阵）=> 仅把“同一个节点在两视图下”视为正样本（式 (4) 第一项）
        # - A'（refined adjacency）=> 把结构上可信的邻居也视为正样本（式 (4) 第二项）
        pos_eye1 = torch.eye(num1).to(refinedadj1.device)
        pos_eye2 = torch.eye(num2).to(refinedadj1.device)
        pos_eye3 = torch.eye(num3).to(refinedadj1.device)
        pos_eye4 = torch.eye(num4).to(refinedadj1.device)
        pos_eye5 = torch.eye(num5).to(refinedadj1.device)

        # z(A'_i)：在 refined 图上的嵌入
        prelogits1 = self._hybrid_spectral_encode(preseq1, refinedadj1, low_adj1, high_adj1, sparse)
        prelogits2 = self._hybrid_spectral_encode(preseq2, refinedadj2, low_adj2, high_adj2, sparse)
        prelogits3 = self._hybrid_spectral_encode(preseq3, refinedadj3, low_adj3, high_adj3, sparse)
        prelogits4 = self._hybrid_spectral_encode(preseq4, refinedadj4, low_adj4, high_adj4, sparse)
        prelogits5 = self._hybrid_spectral_encode(preseq5, refinedadj5, low_adj5, high_adj5, sparse)

        # z(A_i)：在原始图上的嵌入
        logits1 = self.lp(self.gcn,preseq1,adj1,sparse)
        logits2 = self.lp(self.gcn,preseq2,adj2,sparse)
        logits3 = self.lp(self.gcn,preseq3,adj3,sparse)
        logits4 = self.lp(self.gcn,preseq4,adj4,sparse)
        logits5 = self.lp(self.gcn,preseq5,adj5,sparse)

        # 式 (4)：两项对齐目标
        # - lploss1: I(G_i1; G_i2 † I_e)  近似/下界实现（pos=I）
        # - lploss2: I(G_i1; G_i2 † A'_i) 近似/下界实现（pos=A'；detach 防止 A' 学习不稳定带来震荡）
        lploss1 = Calbound.calc_lower_bound(prelogits1, logits1, pos_eye1)+Calbound.calc_lower_bound(prelogits2, logits2, pos_eye2)+Calbound.calc_lower_bound(prelogits3, logits3, pos_eye3)+Calbound.calc_lower_bound(prelogits4, logits4, pos_eye4)+Calbound.calc_lower_bound(prelogits5, logits5, pos_eye5)
        lploss2 = Calbound.calc_lower_bound(prelogits1, logits1, refinedadj1.detach())+Calbound.calc_lower_bound(prelogits2, logits2, refinedadj2.detach())+Calbound.calc_lower_bound(prelogits3, logits3, refinedadj3.detach())+Calbound.calc_lower_bound(prelogits4, logits4, refinedadj4.detach())+Calbound.calc_lower_bound(prelogits5, logits5, refinedadj5.detach())
        
        lploss = lploss1 + lploss2
        if compute_dpc:
            dpc_loss = self._domain_proto_contrast(
                [prelogits1, prelogits2, prelogits3, prelogits4, prelogits5],
                [logits1, logits2, logits3, logits4, logits5],
            )
        else:
            dpc_loss = torch.tensor(0.0, device=lploss.device, dtype=lploss.dtype)
        if self.use_band_gsl and self.band_v2_cross_domain:
            band_cd = self._band_cross_domain_loss(
                [low1, low2, low3, low4, low5],
                [high1, high2, high3, high4, high5],
            )
            lploss = lploss + self.band_cd_weight * band_cd
        if self.use_band_gsl and self.band_deg_anchor_weight > 0:
            da = (
                self._band_degree_anchor(refinedadj1, adj1)
                + self._band_degree_anchor(refinedadj2, adj2)
                + self._band_degree_anchor(refinedadj3, adj3)
                + self._band_degree_anchor(refinedadj4, adj4)
                + self._band_degree_anchor(refinedadj5, adj5)
            )
            lploss = lploss + self.band_deg_anchor_weight * da
        if self.use_band_gsl and self.band_gate_entropy_weight > 0 and self.band_v2_adaptive_gate:
            ent_acc = torch.zeros((), device=refinedadj1.device, dtype=refinedadj1.dtype)
            n_g = 0
            for g in (bg1, bg2, bg3, bg4, bg5):
                if g is None:
                    continue
                p = g.clamp(1e-4, 1.0 - 1e-4)
                ent_acc = ent_acc + (-(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean())
                n_g += 1
            if n_g > 0:
                lploss = lploss - self.band_gate_entropy_weight * (ent_acc / float(n_g))
        if self.use_graph_mae:
            mae = (
                self._graph_mae_loss_single(preseq1, refinedadj1, sparse)
                + self._graph_mae_loss_single(preseq2, refinedadj2, sparse)
                + self._graph_mae_loss_single(preseq3, refinedadj3, sparse)
                + self._graph_mae_loss_single(preseq4, refinedadj4, sparse)
                + self._graph_mae_loss_single(preseq5, refinedadj5, sparse)
            ) / 5.0
            lploss = lploss + self.mae_weight * mae
        if self.use_feat_dv_inv:
            inv = (
                self._feat_view_inv_single(preseq1, prelogits1, refinedadj1, sparse)
                + self._feat_view_inv_single(preseq2, prelogits2, refinedadj2, sparse)
                + self._feat_view_inv_single(preseq3, prelogits3, refinedadj3, sparse)
                + self._feat_view_inv_single(preseq4, prelogits4, refinedadj4, sparse)
                + self._feat_view_inv_single(preseq5, prelogits5, refinedadj5, sparse)
            ) / 5.0
            lploss = lploss + self.feat_dv_weight * inv
        if self.use_gcil:
            gcil = (
                self._gcil_on_domain(prelogits1, logits1, adj1, preseq1, sparse)
                + self._gcil_on_domain(prelogits2, logits2, adj2, preseq2, sparse)
                + self._gcil_on_domain(prelogits3, logits3, adj3, preseq3, sparse)
                + self._gcil_on_domain(prelogits4, logits4, adj4, preseq4, sparse)
                + self._gcil_on_domain(prelogits5, logits5, adj5, preseq5, sparse)
            ) / 5.0
            lploss = lploss + self.gcil_weight * gcil
        if self.use_scgw_p2 and self.scgw is not None:
            lploss = lploss + self.scgw_weight * self.scgw.pretrain_loss(
                [adj1, adj2, adj3, adj4, adj5]
            )
        if self.use_scgw_p4 and self.scgw is not None and self.use_band_gsl:
            band_pairs = [
                (low_adj1, high_adj1),
                (low_adj2, high_adj2),
                (low_adj3, high_adj3),
                (low_adj4, high_adj4),
                (low_adj5, high_adj5),
            ]
            p4 = lploss.new_tensor(0.0)
            n_p4 = 0
            for low_a, high_a in band_pairs:
                if low_a is not None and high_a is not None:
                    p4 = p4 + self.scgw.band_coord_loss(low_a, high_a)
                    n_p4 += 1
            if n_p4:
                lploss = lploss + self.scgw_weight * (p4 / float(n_p4))
        lploss.requires_grad_(True)
        return lploss, dpc_loss
  
    def embedding(self, seq1,seq2,seq3,seq4,seq5,
                adj1,adj2,adj3,adj4,adj5,
                sparse, msk, samp_bias1, samp_bias2):
        
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        seq5 = torch.squeeze(seq5,0)


        preseq1 = self.pretext1(seq1)
        preseq2 = self.pretext2(seq2)
        preseq3 = self.pretext3(seq3)
        preseq4 = self.pretext4(seq4)
        preseq5 = self.pretext5(seq5)

        preseq1 = F.relu(preseq1)
        preseq2 = F.relu(preseq2)
        preseq3 = F.relu(preseq3)
        preseq4 = F.relu(preseq4)
        preseq5 = F.relu(preseq5)
        preseq1 = self._align_preseq_after_relu(
            preseq1, self.film1 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq2 = self._align_preseq_after_relu(
            preseq2, self.film2 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq3 = self._align_preseq_after_relu(
            preseq3, self.film3 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq4 = self._align_preseq_after_relu(
            preseq4, self.film4 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)
        preseq5 = self._align_preseq_after_relu(
            preseq5, self.film5 if (self.use_film_prompt or self.use_film_residual) else None,
            use_film_replace=self.use_film_prompt, use_film_res=self.use_film_residual)

        prelogits1 = self.lp(self.gcn,preseq1,adj1,sparse)
        prelogits2 = self.lp(self.gcn,preseq2,adj2,sparse)
        prelogits3 = self.lp(self.gcn,preseq3,adj3,sparse)
        prelogits4 = self.lp(self.gcn,preseq4,adj4,sparse)
        prelogits5 = self.lp(self.gcn,preseq5,adj5,sparse)

        return prelogits1.detach(),prelogits2.detach(),prelogits3.detach(),prelogits4.detach(),prelogits5.detach()

    def embed(self, seq, adj, sparse, msk,LP):

        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class textprompt(nn.Module):
    """可学习 prompt 向量。

    - combinetype == 'mul'：Hadamard 乘法（论文默认的 ⊙ 形式）。
    - combinetype == 'add'：加法形式（论文 ablation/实现变体时常用）。
    """
    def __init__(self,hid_units,type):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype =type
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding = self.weight * graph_embedding

        return graph_embedding


def mygather(feature, index):
 
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))



def compareloss(feature,tuples,temperature):
    h_tuples=mygather(feature,tuples)

    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))

    temp=temp.cuda()
    h_i = mygather(feature, temp)

    sim = F.cosine_similarity(h_i, h_tuples, dim=2)

    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj, n):
    """Legacy per-node pos/neg index table (O(n·deg); kept for API compat)."""
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n), dtype=np.int64)
    for i in range(nodenum):
        neighbors = indices[indptr[i]:indptr[i + 1]]
        if neighbors.size == 0:
            res[i, 0] = i
        else:
            res[i, 0] = neighbors[0]
        # Sample non-neighbors without O(n) setdiff1d per row.
        picked = []
        need = n
        tries = 0
        max_tries = max(need * 32, nodenum)
        while len(picked) < need and tries < max_tries:
            j = np.random.randint(0, nodenum)
            tries += 1
            if j == i or j in neighbors or j in picked:
                continue
            picked.append(j)
        while len(picked) < need:
            j = np.random.randint(0, nodenum)
            if j not in picked:
                picked.append(j)
        res[i, 1:1 + n] = picked[:n]
    return res


def prompt_pretrain_sample_fast(nodenum: int, n: int, seed: int = 0) -> np.ndarray:
    """O(n) legacy buffer for PrePrompt (forward never reads negative_sample)."""
    rng = np.random.default_rng(seed)
    res = np.zeros((nodenum, 1 + n), dtype=np.int64)
    res[:, 0] = np.arange(nodenum)
    if nodenum > 1:
        res[:, 1:1 + n] = rng.integers(0, nodenum, size=(nodenum, n))
    return res


def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    print(pca.explained_variance_ratio_.sum(), flush=True)
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
 
    return res