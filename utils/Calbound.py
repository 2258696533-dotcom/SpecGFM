"""对比下界损失计算模块（论文式 (4) 的具体实现入口）。

论文在式 (4) 用互信息视角写出两项对齐；代码里用对称的对比目标来近似/下界该项：
给定两组节点表示 `z_1, z_2`（可理解为同一批节点在两种图视图/两种拓扑下得到的 embedding），
以及正样本关系矩阵 `pos`（可为单位阵 I_e，也可为 refined 邻接 A'），
`calc_lower_bound` 先构造全对全相似度，再按行/列做 softmax 归一，最后最大化正边概率（最小化负对数似然）。

注意：`pos` 不必是 0/1；实现里用逐元素乘法把“正样本权重”乘进 softmax 后的矩阵。
"""

import math
import torch
from tools import *

def calc_lower_bound(z_1, z_2, pos, temperature = 0.2):
    EOS = 1e-10    
    z_1=z_1.cuda()
    z_2=z_2.cuda()
    pos=pos.cuda()

    # sim_con：余弦相似度 / temperature（见 `tools.sim_con`）
    matrix_1 = torch.exp(sim_con(z_1, z_2, temperature))
    matrix_2 = matrix_1.t()

    # 对称的两个方向：p(j|i) 与 p(i|j)，对应对比学习里常见的 row/col 归一
    matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EOS)
    lori_1 = -torch.log(torch.clamp(matrix_1.mul(pos).sum(dim=-1), min=1e-10)).mean()

    matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EOS)
    lori_2 = -torch.log(torch.clamp(matrix_2.mul(pos).sum(dim=-1), min=1e-10)).mean()

    return (lori_1 + lori_2) / 2