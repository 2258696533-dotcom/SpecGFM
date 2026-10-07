"""图结构与相似度计算工具模块（论文公式的底层算子集合）。

本文件主要服务于论文第 4.2 节 Graph Topology-aware Alignment 与损失计算（式 (3)(4)(7)）：

- GSL 从 Hi 构造 refined adjacency A'：
  - 相似度矩阵 S = sim(H, H)（`cal_similarity_graph` / `knn_fast`）
  - kNN 稀疏化（`top_k` / `knn_fast`）
  - 对称化/激活/归一化（`symmetrize` / `apply_non_linearity` / `normalize`）
- 对比下界损失（`utils/Calbound.py`）：
  - 全对全余弦相似度 sim(·,·)（`sim_con`）
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres

EOS = 1e-10

def knn_fast(X, k, b):
    """基于余弦相似度的近似 kNN（论文 4.2 节提到的局部近似思想）。

    - X: 节点表征矩阵（如 Hi），形状 [N, d]
    - k: 每个节点保留的近邻数（实际取 top-(k+1)，包含 self）
    - b: 分块大小，避免一次性计算 N×N 相似度带来的显存压力

    返回 rows/cols/values 形式的边列表，可用于快速构造 A'（refined adjacency）。
    """

    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values

def apply_non_linearity(tensor, non_linearity, i):
    """GSL 后处理中的 Activation（论文 4.2 节 Symmetrization/Activation/Normalization 的 Activation）。"""
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')

def symmetrize(adj): 
    """对称化：A <- (A + A^T)/2（论文 4.2 节 Symmetrization）。"""
    return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    """稠密相似度矩阵 S = H H^T（若 H 已做 L2 normalize，则点积≈余弦相似度）。"""
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    """top-k 稀疏化（kNN sparsification，论文 4.2 节）。"""
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def normalize(adj, mode, sparse=False):
    """邻接归一化（Normalization）。

    - sym: D^{-1/2} A D^{-1/2}
    - row: D^{-1} A
    """
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()

def sim_con(z1, z2, temperature):
    """构造对比学习用相似度矩阵：sim(z1, z2)/temperature。

    - sim 使用余弦相似度（论文式 (7) 的 sim(·,·)）
    - temperature 对应论文的 τ（温度超参），在 `Calbound.calc_lower_bound` 里默认 0.2
    """
    
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EOS
    sim_matrix = dot_numerator / dot_denominator / temperature
    return sim_matrix



def dense_to_sparse(dense_matrix):
    
    if not isinstance(dense_matrix, torch.Tensor):
        raise ValueError("输入必须是一个 PyTorch 张量。")
    
    if dense_matrix.dim() != 2:
        raise ValueError("输入的张量必须是二维的。")
    
    dense_np = dense_matrix.cpu().detach().numpy() 
    indices = np.nonzero(dense_np)  
    values = dense_np[indices]


    indices = torch.tensor(indices, dtype=torch.long) 
    values = torch.tensor(values, dtype=torch.float32)  

    sparse_matrix = torch.sparse.FloatTensor(indices, values, dense_matrix.size())

    return sparse_matrix