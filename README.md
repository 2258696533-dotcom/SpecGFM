# SpecGFM

Code for **Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation** (IEEE TKDE).

Leave-one-out cross-domain few-shot node classification: one **target**, five **sources**.

---

## 1. Environment & configuration

Paper runs used this stack (GPU recommended):

| Component | Version |
|-----------|---------|
| Python | **3.9.20** |
| PyTorch | **1.10.1+cu113** |
| CUDA | **11.3** |
| NumPy | **1.x** (e.g. 1.26.4; avoid 2.x) |
| PyTorch Geometric | **2.1.0** |
| torch-scatter / sparse / cluster / spline-conv | **2.0.9 / 0.6.13 / 1.6.0 / 1.2.1** |
| DGL | **0.9.1** |
| SciPy, scikit-learn, tqdm | as needed |

```bash
conda create -n specgfm python=3.9 -y && conda activate specgfm
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.26.4 scipy scikit-learn tqdm
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric==2.1.0
pip install dgl==0.9.1 -f https://data.dgl.ai/wheels/repo.html
```

```bash
python check_env.py
```

Runtime dirs: `data/` (graphs + few-shot splits), `checkpoints/`, `logs/`.

---

## 2. How to run

```bash
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 5

for ds in Cora Citeseer Pubmed Cornell Chameleon Squirrel; do
  python run_specgfm.py --mode specgfm --dataset "$ds" --seeds 1024 --shot_num 1
done
```

RQ2: `--mode rq2_wo_band | rq2_wo_gee | rq2_wo_he | rq2_wo_ho`

---

## 3. Public datasets (not shipped)

| Dataset | Type | PyG |
|---------|------|-----|
| Cora / Citeseer / Pubmed | homophilic | `Planetoid` |
| Cornell | heterophilic | `WebKB` |
| Chameleon / Squirrel | heterophilic | `WikipediaNetwork` |

Place under `data/`. Few-shot files under `data/fewshot_*` (`generate_idx.py` if needed).

---

## 4. Code layout (paper SpecGFM only)

### Entry / core

| File | Role |
|------|------|
| `run_specgfm.py` | Paper launcher → `SpecGFM.py` |
| `SpecGFM.py` | Pre-train + few-shot evaluation |
| `preprompt.py` | BandGSL + multi-domain pre-training |
| `downprompt.py` | Downstream prompts + prototype head |
| `downprompt_bikt.py` | Heterophilic branch (`--use_bikt`) |
| `dual_training.py` | Dual-head training helpers |
| `downstream_encoder.py` | Episode encoder (prompt + GSL + GCN) |
| `scgw_utils.py` | Structural-coordinate alignment (`--scgw_p4`) |
| `tools.py` / `aug.py` | Graph ops / augmentation |
| `generate_idx.py` | Few-shot split generation |
| `check_env.py` | Dependency version check |

### Supporting packages

| Path | Role |
|------|------|
| `layers/` | GCN, attention, readout |
| `models/gcnlayers.py`, `dgi.py`, `graphcl.py`, `LP.py`, `logreg.py` | Backbone |
| `models/branch_utils.py` | Homophily routing helpers |
| `models/f2_downstream_plugins.py` | support-GEE / Dual plugins |
| `models/downstream_gcn.py` | Pick GCN for an episode |
| `models/scale_encoder.py` | Default GCN factory (paper uses this path) |
| `utils/process.py`, `utils/Calbound.py` | Data + contrastive bound |

This repository **does not** include exploratory variants (ALL-IN, GCIL, GRAVER, MTG, ScaleGNN ablations, ProGraph/MFGIA/Tri heads, etc.).
