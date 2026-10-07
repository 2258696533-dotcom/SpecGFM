"""few-shot 索引生成脚本。

按数据集、shot 数与重复次数生成训练索引 `idx.pt` 与标签 `labels.pt`，
为下游 few-shot episode 评估提供输入文件。
"""

import torch
import numpy as np
import os
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB

# 全数据集 + 全shot配置
DATASET_LIST = ["Cora", "Pubmed", "Citeseer", "Chameleon", "Squirrel", "Cornell"]
SHOT_LIST = [1, 5]
NUM_RUNS = 50
SEED_BASE = 1024

# 批量生成
for dataset_name in DATASET_LIST:
    for shot_num in SHOT_LIST:
        print(f"\n===== 修复生成 {dataset_name} - {shot_num}-shot =====")
        
        # 创建目录
        save_root = f"data/fewshot_{dataset_name.lower()}/{shot_num}-shot_{dataset_name.lower()}"
        for run_id in range(NUM_RUNS):
            os.makedirs(f"{save_root}/{run_id}", exist_ok=True)
        
        # 加载数据集
        if dataset_name in ["Cora", "Pubmed", "Citeseer"]:
            dataset = Planetoid(root='data', name=dataset_name)
        elif dataset_name in ["Chameleon", "Squirrel"]:
            dataset = WikipediaNetwork(root='data', name=dataset_name)
        elif dataset_name == "Cornell":
            dataset = WebKB(root='data', name=dataset_name)
        data = dataset[0]
        nb_classes = len(np.unique(data.y.cpu().numpy()))
        
        # 生成文件（核心修复：只存train_idx张量，不存字典！）
        for run_id in range(NUM_RUNS):
            np.random.seed(SEED_BASE + run_id)
            train_idx = []
            
            # 每类采样
            for cls in range(nb_classes):
                cls_idx = np.where(data.y.cpu().numpy() == cls)[0]
                train_cls_idx = np.random.choice(cls_idx, size=shot_num, replace=False)
                train_idx.extend(train_cls_idx)
            
            # 🔥 修复点：直接保存train_idx张量，不是字典！
            torch.save(torch.LongTensor(train_idx), f"{save_root}/{run_id}/idx.pt")
            # labels.pt保持不变
            torch.save(data.y[train_idx], f"{save_root}/{run_id}/labels.pt")
        
        print(f"✅ {dataset_name} {shot_num}-shot 修复完成！")

print("\n🎉 全部修复完成！现在完美适配原代码！")