"""Build the few-shot support splits used by SpecGFM.

Each episode samples k labeled nodes per class once, with seed 1024 plus the episode index,
and writes idx.pt and labels.pt. Training loads these files and does not resample.
"""

import torch
import numpy as np
import os
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB

DATASET_LIST = ["Cora", "Pubmed", "Citeseer", "Chameleon", "Squirrel", "Cornell"]
SHOT_LIST = [1, 5]
NUM_RUNS = 50
SEED_BASE = 1024

for dataset_name in DATASET_LIST:
    for shot_num in SHOT_LIST:
        print(f"\n===== {dataset_name} {shot_num}-shot =====")

        save_root = f"data/fewshot_{dataset_name.lower()}/{shot_num}-shot_{dataset_name.lower()}"
        for run_id in range(NUM_RUNS):
            os.makedirs(f"{save_root}/{run_id}", exist_ok=True)

        if dataset_name in ["Cora", "Pubmed", "Citeseer"]:
            dataset = Planetoid(root='data', name=dataset_name)
        elif dataset_name in ["Chameleon", "Squirrel"]:
            dataset = WikipediaNetwork(root='data', name=dataset_name)
        elif dataset_name == "Cornell":
            dataset = WebKB(root='data', name=dataset_name)
        data = dataset[0]
        nb_classes = len(np.unique(data.y.cpu().numpy()))

        for run_id in range(NUM_RUNS):
            np.random.seed(SEED_BASE + run_id)
            train_idx = []

            for cls in range(nb_classes):
                cls_idx = np.where(data.y.cpu().numpy() == cls)[0]
                train_cls_idx = np.random.choice(cls_idx, size=shot_num, replace=False)
                train_idx.extend(train_cls_idx)

            torch.save(torch.LongTensor(train_idx), f"{save_root}/{run_id}/idx.pt")
            torch.save(data.y[train_idx], f"{save_root}/{run_id}/labels.pt")

        print(f"{dataset_name} {shot_num}-shot done")

print("\nFew-shot splits written.")
