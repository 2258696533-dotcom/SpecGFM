# SpecGFM

Code for **Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation** (IEEE TKDE).

Cross-domain few-shot node classification: one dataset is the **target**, the other five are **sources** for pre-training.

---

## 1. Environment & configuration

Experiments in the paper were run with the following stack (GPU recommended).

### 1.1 Verified versions (reference)

| Component | Version used |
|-----------|----------------|
| Python | **3.9.20** |
| PyTorch | **1.10.1+cu113** |
| CUDA (toolkit matching torch) | **11.3** |
| NumPy | **1.x** (e.g. 1.26.4; avoid NumPy 2.x) |
| PyTorch Geometric | **2.1.0** |
| torch-scatter | **2.0.9** |
| torch-sparse | **0.6.13** |
| torch-cluster | **1.6.0** |
| torch-spline-conv | **1.2.1** |
| DGL | **0.9.1** |
| SciPy | ~1.13 |
| scikit-learn | ~1.6 |
| tqdm | recent |

Also needed: a NVIDIA GPU with a driver that works with the above CUDA build.

### 1.2 Suggested install outline

Create a conda/venv, then install **PyTorch cu113** first, then PyG wheels matching that torch/CUDA, then DGL:

```bash
# example — adjust CUDA/torch URLs to your machine
conda create -n specgfm python=3.9 -y
conda activate specgfm

pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.26.4 scipy scikit-learn tqdm

# PyG 2.1.0 + extensions (use the wheel index that matches torch 1.10.1 + cu113)
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric==2.1.0

pip install dgl==0.9.1 -f https://data.dgl.ai/wheels/repo.html
```

If wheel links 404 on your platform, open [PyG install](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) / [DGL install](https://www.dgl.ai/pages/start.html) and pick the row for **torch 1.10 + CUDA 11.3**.

### 1.3 Check before training

```bash
python check_env.py
```

This prints Python / NumPy / PyTorch / CUDA / PyG / DGL versions against the reference above.

### 1.4 Runtime layout

| Path | Purpose |
|------|---------|
| `data/` | Graphs + few-shot splits (create locally; not shipped) |
| `checkpoints/` | Saved pretrain weights (created when you train) |
| `logs/` | Optional run logs |

Default CLI knobs (also set by `run_specgfm.py`): target `--dataset`, `--seed`, `--shot_num`, pretrain `--lr` / `--epochs`, downstream `--downstreamlr`, GPU `--gpu 0`.

---

## 2. How to run (paper SpecGFM)

```bash
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 5

# all six targets
for ds in Cora Citeseer Pubmed Cornell Chameleon Squirrel; do
  python run_specgfm.py --mode specgfm --dataset "$ds" --seeds 1024 --shot_num 1
done
```

RQ2 ablations: `--mode rq2_wo_band | rq2_wo_gee | rq2_wo_he | rq2_wo_ho`

---

## 3. Public datasets (not in this repo)

Download / process locally into `data/`:

| Dataset | Role in paper | PyG loader |
|---------|---------------|------------|
| **Cora** | homophilic | `Planetoid` |
| **Citeseer** | homophilic | `Planetoid` |
| **Pubmed** | homophilic | `Planetoid` |
| **Cornell** | heterophilic | `WebKB` |
| **Chameleon** | heterophilic | `WikipediaNetwork` |
| **Squirrel** | heterophilic | `WikipediaNetwork` |

Few-shot splits live under `data/fewshot_*` (create with `generate_idx.py` if missing).

---

## 4. Main files (you mainly care about these)


| File | What it does |
|------|----------------|
| **`run_specgfm.py`** | Paper launcher: builds SpecGFM / RQ2 flags and calls `MDGFM.py`. |
| **`MDGFM.py`** | Main program: load graphs, pre-train, few-shot evaluate. |
| **`preprompt.py`** | Pre-training (BandGSL, domain tokens, contrastive alignment). |
| **`downprompt.py`** | Downstream prompt + prototype classifier (base head). |
| **`downprompt_bikt.py`** | Heterophilic-branch head used by SpecGFM (`--use_bikt`). |
| **`tools.py`** | Graph helpers (kNN, normalize, similarity, …). |
| **`aug.py`** | Graph augmentation used in pre-training. |
| **`generate_idx.py`** | Build few-shot `idx.pt` / `labels.pt` splits. |

---

## 5. Other files — by module / when they are used

### 4.1 Always part of the SpecGFM stack (pulled in by the main run)

| File / folder | Belongs to | Role |
|---------------|------------|------|
| `layers/gcn.py`, `readout.py`, `Attentivemod.py`, … | **Encoder layers** | GCN / attention / readout used by pretrain & downstream. |
| `models/gcnlayers.py`, `dgi.py`, `graphcl.py`, `LP.py`, `logreg.py` | **Backbone models** | Core GCN stack and classic SSL heads wired in `preprompt` / `downprompt`. |
| `utils/process.py` | **Data utils** | Adjacency / feature processing. |
| `utils/Calbound.py` | **Pretrain loss** | Contrastive lower-bound used in `preprompt`. |
| `scgw_utils.py` | **SpecGFM pretrain** | Structural-coordinate alignment (`--scgw_p4`). |
| `dual_training.py` | **SpecGFM downstream** | Dual-head training helpers. |
| `models/f2_downstream_plugins.py` | **SpecGFM downstream** | support-GEE and related Dual-branch plugins (`--f2_gee_branch dual`). |
| `models/mdgfm_tri.py` | **Router / BiKT** | Homophily episode helpers shared with BiKT path. |
| `downstream_encoder.py` | **Downstream encode** | Builds frozen / dual encoders for episodes. |

### 4.2 Imported by `MDGFM.py`, but only active for other variants (not default SpecGFM)

Default SpecGFM does **not** need you to turn these on; they stay for optional / historical modes.

| File | Variant / switch | What it is for |
|------|------------------|----------------|
| `downprompt_prograph.py` | `--use_prograph` | ProGraph-style downstream head. |
| `downprompt_mfgia.py` | `--use_mfgia` | MFGIA-style downstream head. |
| `downprompt_tri.py` | `--use_mdgfm_tri` | Tri-fusion downstream head. |
| `allin_adapter.py` | `--feature_adapter allin` | ALL-IN style feature adapter (default is `pca`). |
| `gcil_utils.py` | `--use_gcil` | GCIL spectral regularizer. |
| `graver_downstream.py`, `models/graver_vocab.py` | `--use_mdgfm_graver` | GRAVER vocabulary / MoE downstream. |
| `models/message_tuning.py`, `models/mtg_utils.py` | `--use_mtg` | Message-Tuning (MTG) adapter. |
| `models/scale_encoder.py`, `scale_gcn_layers*.py`, `gpr_gcn_layers*.py` | `--scale_encoder …` | ScaleGNN / GPR encoder variants. |
| `models/hybrid_spectral_encoder.py`, `spectral_prompt_graph.py` | `--use_hybrid_spectral_pretrain` / HS modes | Hybrid spectral pretrain & spectral prompt. |
| `models/adaptive_prop.py` | `--use_mdgfm_ap` | Adaptive propagation prompt. |
| `models/paper_route_plugins.py` | `--paper_plugins_all` / R-GFM flags | Extra paper-route plugins (GoG, etc.). |

### 4.3 Optional tooling

| File | When to use |
|------|-------------|
| `runexp.py` | Older wrapper around `MDGFM.py`; prefer `run_specgfm.py`. |
| `models/gcnformal.py` | Legacy formal GCN helper; not on the SpecGFM default path. |

---

## 5. Short map

```
run_specgfm.py  →  MDGFM.py
                      ├─ preprompt.py  (+ BandGSL, scgw_utils, layers, models/gcn*)
                      └─ downprompt / downprompt_bikt
                           (+ dual_training, f2_downstream_plugins, …)
```

You only need section **3** to understand and reproduce SpecGFM.  
Section **4.2** is “other variants behind flags”; ignore them unless you open those switches.
