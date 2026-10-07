# SpecGFM

Code for **Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation** (IEEE TKDE submission).

Paper setting: **leave-one-out cross-domain few-shot node classification** — one dataset is the unseen target; the other five are source domains for pre-training.

## Public datasets (not shipped in this repo)

Place graphs under `data/` (PyTorch Geometric layout). The six public benchmarks used in the paper are:

| Dataset | Type | Loader (in `MDGFM.py`) |
|---------|------|-------------------------|
| **Cora** | homophilic citation | `Planetoid(root='data', name='Cora')` |
| **Citeseer** | homophilic citation | `Planetoid(root='data', name='Citeseer')` |
| **Pubmed** | homophilic citation | `Planetoid(root='data', name='Pubmed')` |
| **Cornell** | heterophilic WebKB | `WebKB(root='data', name='Cornell')` |
| **Chameleon** | heterophilic Wikipedia | `WikipediaNetwork(root='data', name='Chameleon')` |
| **Squirrel** | heterophilic Wikipedia | `WikipediaNetwork(root='data', name='Squirrel')` |

Also needed for few-shot evaluation (generate once if missing):

- `data/fewshot_<dataset>/<k>-shot_<dataset>/<episode>/{idx,labels}.pt`  
  Use `generate_idx.py` to build splits if you do not already have them.

**Do not** commit raw `data/` into git; download / process locally.

## Environment

- Python 3.8+ with GPU recommended  
- PyTorch, PyTorch Geometric, DGL  
- numpy, scipy, scikit-learn, tqdm  

## SpecGFM main runs (paper mode)

Paper SpecGFM configuration corresponds to mode **`f2p4_r4_gee`** (Band / dual-branch stack + support-GEE on the homophilic branch) via `run_cora_seeds.py`.

### Single target, 1-shot (example: Cora)

```bash
python run_cora_seeds.py --mode f2p4_r4_gee --dataset Cora --seeds 1024 --shot_num 1
```

### Single target, 5-shot

```bash
python run_cora_seeds.py --mode f2p4_r4_gee --dataset Cora --seeds 1024 --shot_num 5
```

### All six targets (loop)

```bash
for ds in Cora Citeseer Pubmed Cornell Chameleon Squirrel; do
  python run_cora_seeds.py --mode f2p4_r4_gee --dataset "$ds" --seeds 1024 --shot_num 1
done
```

Replace `--shot_num 1` with `5` for 5-shot. For multi-seed paper tables, pass several seeds, e.g. `--seeds 512 1024 2048 4096 8192`.

### Optional dry-run (print commands only)

```bash
python run_cora_seeds.py --mode f2p4_r4_gee --dataset Cora --seeds 1024 --shot_num 5 --dry_run
```

### Lower-level entry

`run_cora_seeds.py` forwards flags to `MDGFM.py` / `runexp.py`. Direct use:

```bash
python MDGFM.py --help
python runexp.py --help
```

## Code layout

| Path | Role |
|------|------|
| `MDGFM.py` | Main train / few-shot eval entry |
| `run_cora_seeds.py` | Paper experiment launcher (modes / seeds / datasets) |
| `runexp.py` | Thin wrapper that calls `MDGFM.py` |
| `preprompt.py` | Multi-domain spectral–structural pre-training (BandGSL, …) |
| `downprompt.py` | Homophily-guided dual-branch adaptation |
| `layers/`, `models/`, `utils/` | Backbone and helpers |

## Note

This repository contains **main experiment model code** needed to run SpecGFM.  
It does not include checkpoints, full one-off shell suites, or dataset files.
