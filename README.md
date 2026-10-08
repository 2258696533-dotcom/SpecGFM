# SpecGFM

Code for **Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation** (IEEE TKDE).

The protocol matches the paper. One of the six graphs is the unseen target, and the other five are the pre-training sources. The main experiment reports 1-shot and 5-shot node classification. RQ2 uses the same unseen-target protocol as the 1-shot table.

This repository contains the SpecGFM main run and the four RQ2 variants.

---

## 1. Hardware

The paper experiments were run on the following machine:

| Item | Configuration |
|------|----------------|
| GPU | NVIDIA A800 80GB × 1 |
| CPU | 2 × Intel Xeon Gold 6348 @ 2.60 GHz |
| Memory | 1 TB |
| OS | Ubuntu 18.04.5 LTS |

---

## 2. Software

| Component | Version |
|-----------|---------|
| Python | 3.9.20 |
| PyTorch | 1.10.1+cu113 |
| CUDA | 11.3 |
| NumPy | 1.x (for example 1.26.4; NumPy 2.x is incompatible) |
| PyTorch Geometric | 2.1.0 |
| torch-scatter / sparse / cluster / spline-conv | 2.0.9 / 0.6.13 / 1.6.0 / 1.2.1 |
| DGL | 0.9.1 |
| SciPy, scikit-learn, tqdm | a recent stable release |

```bash
conda create -n specgfm python=3.9 -y && conda activate specgfm
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.26.4 scipy scikit-learn tqdm
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric==2.1.0
pip install dgl==0.9.1 -f https://data.dgl.ai/wheels/repo.html
python check_env.py
```

Checkpoints go in `checkpoints/` and logs go in `logs/`. All six graphs are public datasets. Section 3 explains how to either unpack the copies in this repository or download those public datasets and reproduce the same few-shot splits.

---

## 3. Datasets

All six graphs are public. Cora, Citeseer, and Pubmed are the Planetoid citation networks. Cornell is a WebKB network. Chameleon and Squirrel are the WikipediaNetwork graphs used by Geom-GCN. This repository does not claim ownership of those graphs. Two ways to obtain them are below. Use either one. The code reads `Planetoid(root='data', ...)`, `WebKB(root='data', ...)`, and `WikipediaNetwork(root='data', ...)` from the repository root, the directory that contains `SpecGFM.py`.

The few-shot files are not part of the public graph releases. Each episode was sampled once, before training, and saved. For dataset, shot `k` in `{1, 5}`, and episode `i` in `0..49`, the support set is

```text
data/fewshot_<name>/<k>-shot_<name>/<i>/idx.pt
data/fewshot_<name>/<k>-shot_<name>/<i>/labels.pt
```

`<name>` is `cora`, `citeseer`, `pubmed`, `cornell`, `chameleon`, or `squirrel`. Episode `i` uses NumPy seed `1024 + i` and draws `k` nodes from each class. Training only loads these files. It does not sample a new support set.

### Option A. Unpack the archives in this repository

The copies used for the paper are the zip archives already in `data/`. From the repository root:

```bash
cd data
unzip -o Cora.zip Citeseer.zip Pubmed.zip cornell.zip chameleon.zip squirrel.zip fewshot.zip
cd ..
```

Unpack them inside `data/`, not in the repository root and not in a new folder. Cornell, Chameleon, and Squirrel use lowercase archive names because those are the PyTorch Geometric directory names. After unpacking, these files must exist:

```text
data/Cora/processed/data.pt
data/Citeseer/processed/data.pt
data/Pubmed/processed/data.pt
data/cornell/processed/data.pt
data/chameleon/geom_gcn/processed/data.pt
data/squirrel/geom_gcn/processed/data.pt
```

`processed/data.pt` is already inside each graph archive, so PyTorch Geometric reads the local files and does not download them. `fewshot.zip` is the paper split. With this option, do not run `generate_idx.py`.

### Option B. Download the public datasets

Use PyTorch Geometric 2.1.0, as in Section 2. Leave the six graph directories absent and let the loader download the public releases into `data/` on first use. You can start that download by running `generate_idx.py` from the repository root:

```bash
python generate_idx.py
```

That script loads Cora, Citeseer, and Pubmed through `Planetoid`, Cornell through `WebKB`, and Chameleon and Squirrel through `WikipediaNetwork`. It then writes the same 50 episodes with seed `1024 + i`. If you would rather keep the paper split exactly as released, unpack only `fewshot.zip` and do not run `generate_idx.py`.

| Dataset | Folder | PyG class | Nodes |
|---------|--------|-----------|-------|
| Cora | `data/Cora` | `Planetoid` | 2,708 |
| Citeseer | `data/Citeseer` | `Planetoid` | 3,327 |
| Pubmed | `data/Pubmed` | `Planetoid` | 19,717 |
| Cornell | `data/cornell` | `WebKB` | 183 |
| Chameleon | `data/chameleon` | `WikipediaNetwork` | 2,277 |
| Squirrel | `data/squirrel` | `WikipediaNetwork` | 5,201 |

## 4. Running the experiments

These commands are the paper protocol. `run_specgfm.py` sets the pre-training length, learning rates, and episode count from the target name, so they do not need to be typed on the command line.

| Target | Pre-training epochs | Pre-training lr | Downstream lr |
|--------|---------------------|-----------------|---------------|
| Cora | 60 | 0.0075 | 0.001 |
| Citeseer | 60 | 0.001 | 0.001 |
| Pubmed | 60 | 0.0001 | 0.0014 |
| Cornell | 100 | 0.02 | 0.0003 |
| Chameleon | 100 | 0.02 | 0.02 |
| Squirrel | 100 | 0.01 | 0.0003 |

Every command evaluates 50 few-shot episodes. The homophilic branch trains for 300 steps; the heterophilic branch trains for 400 steps. The paper tables average five seeds, `512 1024 2048 4096 8192`. One `--seeds` value is one of those repeats.

Main model (BandGSL, structural coordinate alignment, homophily-guided dual-branch routing, and support-GEE on the homophilic branch):

```bash
python run_specgfm.py --mode specgfm --dataset Cora --seeds 512 1024 2048 4096 8192 --shot_num 1
python run_specgfm.py --mode specgfm --dataset Cora --seeds 512 1024 2048 4096 8192 --shot_num 5
```

All six targets, 1-shot:

```bash
for ds in Cora Citeseer Pubmed Cornell Chameleon Squirrel; do
  python run_specgfm.py --mode specgfm --dataset "$ds" --seeds 512 1024 2048 4096 8192 --shot_num 1
done
```

RQ2 (1-shot, same unseen-target protocol as the main table):

| `--mode` | Variant | What is removed |
|----------|---------|-----------------|
| `rq2_wo_band` | w/o-Band | BandGSL, including structural coordinate alignment |
| `rq2_wo_he` | w/o-He | Heterophilic branch; every episode uses the homophilic branch |
| `rq2_wo_ho` | w/o-Ho | Homophilic branch, equivalent to locking the routing threshold at \(\tau=1\) |
| `rq2_wo_gee` | w/o-GEE | Support-GEE on the homophilic branch |

```bash
python run_specgfm.py --mode rq2_wo_band --dataset Cora --seeds 512 1024 2048 4096 8192 --shot_num 1
python run_specgfm.py --mode rq2_wo_he   --dataset Cora --seeds 512 1024 2048 4096 8192 --shot_num 1
python run_specgfm.py --mode rq2_wo_ho   --dataset Cora --seeds 512 1024 2048 4096 8192 --shot_num 1
python run_specgfm.py --mode rq2_wo_gee  --dataset Cora --seeds 512 1024 2048 4096 8192 --shot_num 1
```

On the homophilic branch, the logits of three independently initialized linear heads are averaged (`--dual_ensemble 3`). The routing threshold \(\tau\) is `--homo_bypass_thresh` (0.52 in the main run): an episode uses the homophilic branch when \(h_e>\tau\), and the heterophilic branch otherwise. Support-GEE is applied only on the homophilic branch (`--f2_gee_branch dual`).

---

## 5. Files

| File | Role |
|------|------|
| `run_specgfm.py` | Launcher for the main run and the RQ2 variants |
| `SpecGFM.py` | Pre-training and few-shot evaluation |
| `preprompt.py` | BandGSL and multi-domain pre-training |
| `scgw_utils.py` | Structural coordinate alignment of the two BandGSL adjacencies |
| `dual_training.py` | Homophilic branch (linear head, prototype head, support-GEE) |
| `downprompt_bikt.py` | Heterophilic branch |
| `downprompt.py` | Prompts and prototypes used by the heterophilic branch |
| `models/f2_downstream_plugins.py` | Support-GEE |
| `models/branch_utils.py` | Routing by the episode homophily \(h_e\) |
| `aug.py`, `tools.py`, `generate_idx.py` | Augmentation, graph utilities, few-shot splits |
| `layers/`, `models/`, `utils/` | GCN backbone and pre-training losses |
| `check_env.py` | Checks the software versions above |
