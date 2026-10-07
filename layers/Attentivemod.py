"""Attention 基础层模块。

定义逐维可学习缩放层 `Attentive`，用于对输入特征维度加权，
常被上层结构学习器（ATT_learner）用于图关系重构前的表征变换。
"""

import dgl.function as fn
import torch
import torch.nn as nn



class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)