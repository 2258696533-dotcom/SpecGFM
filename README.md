# SpecGFM

Code for **Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation** (IEEE TKDE).

Leave-one-out cross-domain few-shot: one **target**, five **sources**.

---

## 1. Environment & configuration

| Component | Version (paper runs) |
|-----------|----------------------|
| Python | **3.9.20** |
| PyTorch | **1.10.1+cu113** |
| CUDA | **11.3** |
| NumPy | **1.x** (e.g. 1.26.4; avoid 2.x) |
| PyTorch Geometric | **2.1.0** |
| torch-scatter / sparse / cluster / spline-conv | **2.0.9 / 0.6.13 / 1.6.0 / 1.2.1** |
| DGL | **0.9.1** |
| SciPy, scikit-learn, tqdm | recent |

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

Dirs: `data/` (graphs + few-shot), `checkpoints/`, `logs/`.

---

## 2. Run paper SpecGFM

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

---

## 4. Main files

| File | Role |
|------|------|
| `run_specgfm.py` | Paper launcher |
| `SpecGFM.py` | Train / few-shot eval entry |
| `preprompt.py` | BandGSL + multi-domain pre-training |
| `downprompt.py` | Downstream prompts + prototypes |
| `downprompt_bikt.py` | Heterophilic branch |
| `dual_training.py` / `downstream_encoder.py` | Dual head / episode encoder |
| `scgw_utils.py` | Structural-coordinate alignment (`--scgw_p4`) |
| `tools.py` / `aug.py` / `generate_idx.py` | Helpers / few-shot splits |
| `check_env.py` | Version check |
| `layers/` `models/` `utils/` | Backbone & losses |

---

## 5. Optional variants (extra modules / flags)

Not required for paper SpecGFM; available if you pass the matching flags in `SpecGFM.py` / `runexp.py`:

| Module / file | Flag(s) | Notes |
|---------------|---------|--------|
| `allin_adapter.py` | `--feature_adapter allin` | Feature adapter instead of PCA |
| `gcil_utils.py` | `--use_gcil` | GCIL regularizer |
| `graver_downstream.py`, `models/graver_vocab.py` | `--use_graver` | GRAVER vocabulary path |
| `models/message_tuning.py`, `mtg_utils.py` | `--use_mtg` | Message-Tuning |
| `models/scale_encoder.py` + `scale_gcn_*` / `gpr_*` | `--scale_encoder …` | Scale / GPR encoders |
| `models/hybrid_spectral_encoder.py`, `spectral_prompt_graph.py` | `--use_hs`, `--use_hybrid_spectral_pretrain` | Hybrid spectral |
| `models/adaptive_prop.py` | `--use_ap` | Adaptive propagation |
| `downprompt_prograph.py` | `--use_prograph` | ProGraph head |
| `downprompt_mfgia.py` | `--use_mfgia` | MFGIA head |
| `downprompt_tri.py` | `--use_tri` | Tri-fusion head |
| `models/paper_route_plugins.py` | `--paper_plugins_all` / R-GFM flags | Extra paper-route plugins |
| `runexp.py` | (wrapper) | Older CLI wrapper around `SpecGFM.py` |

Default paper path only needs §4 + `run_specgfm.py --mode specgfm`.
