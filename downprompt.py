"""Downstream few-shot head: prompts, dual-branch routing hooks, prototype classifier."""


import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL
from layers import Attentivemod

from layers import GCN, AvgReadout
import torch_scatter
from tools import *

class prefeatureprompt(nn.Module):
    """把 meta prompt 分支与 specific prompt 分支做融合（论文式 (5) 的双流思想）。

    - `precomposedfeature`：多源 token 组合后再与输入相乘/相加（Hadamard 或 add）。
    - `preopenfeature`：任务特定提示（类似 p_s 作用的简化参数化）。
    - `combineprompt`：学习两者权重（相当于学习 β 的同类机制，但作用在两条特征流上）。
    """
    def __init__(self,texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,sumtext,dim,type:str,head_num=8,
                 decouple_meta=False, decouple_blend=0.0):
        super(prefeatureprompt, self).__init__()
        self.decouple_meta = bool(decouple_meta)
        self.decouple_blend = float(max(0.0, min(1.0, decouple_blend)))
        if not self.decouple_meta:
            self.precomposedfeature = composedtoken(texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,type)
            self.preopenfeature = downstreamprompt(dim)
            self.sumtext = sumtext
            self.combineprompt = combineprompt()
        else:
            from models.paper_route_plugins import TargetOnlyPrompt
            self.target_only = TargetOnlyPrompt(dim, prompt_type=type)
            if self.decouple_blend > 0.0:
                self.precomposedfeature = composedtoken(texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,type)
                self.preopenfeature = downstreamprompt(dim)
                self.sumtext = sumtext
                self.combineprompt = combineprompt()
    def forward(self,seq):
        if self.decouple_meta:
            target = self.target_only(seq)
            if self.decouple_blend <= 0.0 or not hasattr(self, 'precomposedfeature'):
                return target
            seq1 = self.precomposedfeature(seq)
            seq1 = F.relu(seq1)
            seq3 = self.sumtext * seq1
            seq2 = self.preopenfeature(seq)
            meta = self.combineprompt(seq3, seq2)
            b = self.decouple_blend
            return b * meta + (1.0 - b) * target
        seq1 = self.precomposedfeature(seq)
        seq1 = F.relu(seq1)
        seq3 = self.sumtext * seq1
        seq2 = self.preopenfeature(seq)
        ret =  self.combineprompt(seq3 ,seq2) 
        return ret

class composedtoken(nn.Module):
    """多源 domain token 的线性组合（论文式 (5) 的 Σ α_i t_{D_i} 部分）。

    实现细节：先把 5 个域 token 拼成矩阵，再用 `weighted_prompt(5)` 学习 1×5 权重并映射到 1×d，
    最后按 `type` 与节点特征做 Hadamard 或加法。
    """
    def __init__(self,texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,type:str,head_num=8):
        super(composedtoken, self).__init__()   
        self.texttoken = torch.cat((texttoken1,texttoken2,texttoken3,texttoken4,texttoken5),dim=0)
        self.prompt = weighted_prompt(5).cuda()
        self.type = type
        self._scgw_bias = None
        self._scgw_blend = 0.0

    def set_scgw_bias(self, domain_bias: torch.Tensor, blend: float):
        self._scgw_bias = domain_bias
        self._scgw_blend = float(max(0.0, min(1.0, blend)))

    def forward(self,seq):
        texttoken = self.prompt(self.texttoken, scgw_bias=self._scgw_bias, scgw_blend=self._scgw_blend)
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets

class ATT_learner(nn.Module):
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
        if self.sparse:
            rows, cols, values = knn_fast(embeddings, k, 100)      
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

class textprompt(nn.Module):
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

class downprompt(nn.Module):
    def __init__(self,token1,token2,token3,token4,token5,
            sumtext,pretoken1,pretoken2,pretoken3,pretoken4,pretoken5,balancetoken1,balancetoken2,balancetoken3,balancetoken4,balancetoken5,
            ft_in, nb_classes, type,feature_dim, scgw_module=None, scgw_prompt_blend=0.3,
            use_gfmate_centroid=False, use_layer_prompt=False, num_gcn_layers=3,
            decouple_meta=False, decouple_blend=0.0, gog_encoder=None, riemann_moe=None,
            rgfm_max_hop=3, rgfm_gog_homo_min=0.35):
        super(downprompt, self).__init__()
        self.downprompt = downstreamprompt(ft_in)
        self.composedprompt = composedtoken(token1,token2,token3,token4,token5,type=type)
        self.prefeature = prefeatureprompt(
            pretoken1, pretoken2, pretoken3, pretoken4, pretoken5, sumtext,
            dim=feature_dim, type=type, head_num=4, decouple_meta=decouple_meta,
            decouple_blend=decouple_blend,
        )
        self.combineprompt1 = combineprompt()
        self.combineprompt2 = combineprompt()
        self.balancedprompt = textprompt(2 * feature_dim,type)
        self.learner = ATT_learner(2, 50, 6, 0.5, sparse = True, act = 'relu')
        self.struct_alpha = nn.Parameter(torch.tensor(0.5))
        self.ap = None

        self.nb_classes = nb_classes
        self.ft_in = ft_in
        self.leakyrelu = nn.ELU() 
        self.one = torch.ones(1,ft_in).cuda()
        self.ave = torch.FloatTensor(nb_classes,ft_in).cuda()
        self.ave_layers = None
        self.scgw_module = scgw_module
        self.scgw_prompt_blend = float(scgw_prompt_blend)
        self.use_layer_prompt = bool(use_layer_prompt)
        self.num_gcn_layers = max(1, int(num_gcn_layers))
        self.gog_encoder = gog_encoder
        self.riemann_moe = riemann_moe
        self.rgfm_max_hop = max(1, int(rgfm_max_hop))
        self.rgfm_gog_homo_min = float(rgfm_gog_homo_min)
        self._episode_homo_score = None
        self.layer_prompt = None
        if self.use_layer_prompt:
            from models.paper_route_plugins import LayerPromptEnsemble
            self.layer_prompt = LayerPromptEnsemble(self.num_gcn_layers)
        if use_gfmate_centroid:
            if self.use_layer_prompt:
                self.centroid_prompt = nn.Parameter(
                    torch.zeros(self.num_gcn_layers, nb_classes, ft_in)
                )
            else:
                self.centroid_prompt = nn.Parameter(torch.zeros(nb_classes, ft_in))
        else:
            self.centroid_prompt = None

    def refined_prototypes(self, layer_id=None):
        """Class centroids with optional GFMate centroid prompt offset."""
        if self.use_layer_prompt and layer_id is not None:
            base = self.ave_layers[layer_id]
            if self.centroid_prompt is not None:
                return base + self.centroid_prompt[layer_id]
            return base
        if self.centroid_prompt is not None and self.centroid_prompt.dim() == 2:
            return self.ave + self.centroid_prompt
        return self.ave

    def refined_prototypes_all_layers(self):
        if not self.use_layer_prompt or self.ave_layers is None:
            return [self.refined_prototypes()]
        out = []
        for l in range(len(self.ave_layers)):
            out.append(self.refined_prototypes(layer_id=l))
        return out

    def _encode_nodes(self, features, adj, sparse, gcn, downk):
        self._apply_scgw_prompt_bias(adj)
        features1 = self.prefeature(features)
        reseq1 = torch.sparse.mm(adj, features1)
        reseq1 = torch.cat((features1, reseq1), dim=1)
        reseq111 = self.balancedprompt(reseq1)
        adj1 = self.learner.graph_process(downk, reseq111).to('cuda')
        adjtot = self.struct_alpha * adj.to_dense() + (1 - self.struct_alpha) * adj1
        if self.ap is not None:
            adjtot = self.ap(adjtot, homo_score=getattr(self, "_ap_homo_score", None))
        from models.paper_route_plugins import gcn_encode
        final, layer_list = gcn_encode(
            gcn, features1, adjtot, sparse, lp=False, return_all_layers=self.use_layer_prompt,
        )
        return features1, adjtot, final, layer_list

    def _select_layer_embeds(self, layer_list, idx):
        if self.use_layer_prompt:
            return [h[idx] for h in layer_list[: self.num_gcn_layers]]
        return [layer_list[-1][idx]]

    def _proto_classify(self, query_emb, layer_embeds=None):
        from models.paper_route_plugins import layer_proto_probs, cosine_proto_matrix
        if self.use_layer_prompt and layer_embeds is not None and self.ave_layers is not None:
            offset = self.centroid_prompt if self.centroid_prompt is not None else None
            return layer_proto_probs(
                layer_embeds, self.ave_layers, self.layer_prompt, centroid_offset=offset,
            )
        protos = self.refined_prototypes()
        logits = cosine_proto_matrix(query_emb, protos)
        return F.softmax(logits, dim=1)

    def _apply_scgw_prompt_bias(self, adj) -> None:
        """SCGW P3: blend target-graph structural coords into meta prompt."""
        if self.scgw_module is None or getattr(self.prefeature, "decouple_meta", False):
            return
        if not hasattr(self.prefeature, "precomposedfeature"):
            return
        with torch.no_grad():
            bias = self.scgw_module.domain_prompt_bias(adj)
        self.prefeature.precomposedfeature.set_scgw_bias(bias, self.scgw_prompt_blend)

    def forward(self,features,adj,sparse,gcn,idx,seq,downk,labels=None,train=0,return_aux=False,
                tgcl_step=0, tgcl_total=0):
        """Few-shot 前向。

        1) `prefeature`：把预训练阶段学到的 token/prompt 注入到下游特征（论文 4.3 的 prompt-tuning）。
        2) 构造 [X', A X'] 并经 `balancedprompt`：与上游式 (3) 相同的“特征-结构平衡”机制，用于下游 GSL。
        3) `adjtot`：原始邻接与学习邻接凸组合，得到用于编码的拓扑。
        4) `gcn`：得到节点表示 z（论文式 (6) 的 GE 部分；此处 θ_pre 来自预训练）。
        5) 用 support 节点计算每类原型均值（式 (7) 的 z̄_y），再对 query 做余弦相似度分类。
        """
        _, adjtot, embeds_all, layer_list = self._encode_nodes(features, adj, sparse, gcn, downk)
        from models.paper_route_plugins import build_layer_prototypes, refine_node_embeds
        idx_t = torch.as_tensor(list(idx), device=embeds_all.device, dtype=torch.long)
        layer_embeds_clean = self._select_layer_embeds(layer_list, idx_t)
        rawret_clean = layer_embeds_clean[-1].cuda()

        if train == 1:
            if self.use_layer_prompt:
                self.ave_layers = build_layer_prototypes(layer_embeds_clean, labels, self.nb_classes)
                self.ave = self.ave_layers[-1]
            else:
                self.ave = averageemb(labels=labels, rawret=rawret_clean, nb_class=self.nb_classes)

        self._last_layer_embeds = layer_embeds_clean if self.use_layer_prompt else None
        self._last_layer_protos = self.ave_layers if self.use_layer_prompt else None

        use_refine = self.gog_encoder is not None or self.riemann_moe is not None
        homo = self._episode_homo_score
        refine_kw = dict(
            max_hop=self.rgfm_max_hop, tgcl_step=tgcl_step, tgcl_total=tgcl_total,
            homo_score=homo, homo_min=self.rgfm_gog_homo_min,
        )
        if use_refine:
            embeds_ref = refine_node_embeds(
                embeds_all, adjtot, idx_t, self.gog_encoder, self.riemann_moe, **refine_kw,
            )
            if self.use_layer_prompt:
                layer_embeds = [
                    refine_node_embeds(
                        h, adjtot, idx_t, self.gog_encoder, self.riemann_moe, **refine_kw,
                    )[idx_t]
                    for h in layer_list[: self.num_gcn_layers]
                ]
            else:
                layer_embeds = [embeds_ref[idx_t]]
            rawret = layer_embeds[-1].cuda()
        else:
            layer_embeds = layer_embeds_clean
            rawret = rawret_clean

        if self.use_layer_prompt:
            self._last_query_layer_embeds = layer_embeds
        else:
            self._last_query_layer_embeds = None

        ret = self._proto_classify(rawret, layer_embeds=layer_embeds if self.use_layer_prompt else None)
        proto_out = self.refined_prototypes_all_layers()[-1] if self.use_layer_prompt else self.refined_prototypes()

        if return_aux:
            return ret, rawret, proto_out
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)


    def forward(self, graph_embedding1, graph_embedding2):
        
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)

def averageemb(labels,rawret,nb_class):
    retlabel = torch_scatter.scatter(src=rawret,index=labels,dim=0,reduce='mean')
    return retlabel


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        x = self.fc(attended_values) + x
        x = torch.squeeze(x)

        x = torch.sum(x,dim=0)
        return x


class weighted_prompt(nn.Module):
    def __init__(self,weightednum):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, scgw_bias=None, scgw_blend=0.0):

        w = self.weight
        if scgw_bias is not None and scgw_blend > 0.0:
            b = float(max(0.0, min(1.0, scgw_blend)))
            w = (1.0 - b) * w + b * scgw_bias.view_as(w)
        graph_embedding=torch.mm(w,graph_embedding)
        return graph_embedding
   

class weighted_feature(nn.Module):
    def __init__(self,weightednum):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)
    def forward(self, graph_embedding1,graph_embedding2):
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    
class downstreamprompt(nn.Module):
    def __init__(self,hid_units):
        super(downstreamprompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        graph_embedding=self.weight * graph_embedding
        return graph_embedding
    
class featureprompt(nn.Module):
    def __init__(self,prompt1,prompt2,prompt3):
        super(featureprompt, self).__init__()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.weightprompt = weighted_prompt(3)
    def forward(self,feature):
        weight = self.weightprompt(self.prompt)
        feature = weight * feature
        return feature