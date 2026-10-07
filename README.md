# SpecGFM

Code for **Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation** (IEEE TKDE).

Setting: leave-one-out cross-domain few-shot node classification (one target, five sources).

## Public datasets (download locally; not in this repo)

| Dataset | Homophily | PyG class |
|---------|-----------|-----------|
| Cora | homophilic | `Planetoid` |
| Citeseer | homophilic | `Planetoid` |
| Pubmed | homophilic | `Planetoid` |
| Cornell | heterophilic | `WebKB` |
| Chameleon | heterophilic | `WikipediaNetwork` |
| Squirrel | heterophilic | `WikipediaNetwork` |

Put processed graphs under `data/` as expected by `MDGFM.py`.  
Few-shot splits: `data/fewshot_<name>/<k>-shot_.../` (build with `generate_idx.py` if needed).

## Environment

Python 3.8+, PyTorch, PyTorch Geometric, DGL, numpy, scipy, scikit-learn, tqdm. GPU recommended.

## Run SpecGFM (paper)

```bash
# 1-shot, single target
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 1

# 5-shot
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 5

# All six targets
for ds in Cora Citeseer Pubmed Cornell Chameleon Squirrel; do
  python run_specgfm.py --mode specgfm --dataset "$ds" --seeds 1024 --shot_num 1
done

# Multi-seed (paper tables)
python run_specgfm.py --mode specgfm --dataset Cora --seeds 512 1024 2048 4096 8192 --shot_num 1

# Print command only
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 5 --dry_run
```

`--mode specgfm` is the paper configuration (same stack as historical `f2p4_r4_gee`).

### RQ2 ablations

```bash
python run_specgfm.py --mode rq2_wo_band --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode rq2_wo_gee  --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode rq2_wo_he   --dataset Cornell --seeds 1024 --shot_num 1
python run_specgfm.py --mode rq2_wo_ho   --dataset Cornell --seeds 1024 --shot_num 1
```

## Layout

- `run_specgfm.py` — paper launcher  
- `MDGFM.py` — train / evaluate  
- `preprompt.py` / `downprompt.py` — pre-train & adaptation  
- `layers/` `models/` `utils/` — backbone helpers  

Checkpoints and raw datasets are not included.
