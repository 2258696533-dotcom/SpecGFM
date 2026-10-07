"""图读出（Readout）层模块。

提供节点到图级表示的聚合操作（均值或掩码均值），
用于图对比学习与上下文向量构建。
"""

import torch
import torch.nn as nn


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

