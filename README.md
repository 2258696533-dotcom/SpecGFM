# SpecGFM

Code for **Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation** (TKDE submission).

## Main entry

- `MDGFM.py` — pre-training + few-shot downstream evaluation
- `preprompt.py` — multi-domain spectral–structural pre-training (BandGSL, etc.)
- `downprompt.py` — homophily-guided dual-branch adaptation

## Layout

```
MDGFM.py / preprompt.py / downprompt.py
layers/    # GCN and related layers
models/    # encoders and helpers used by the main pipeline
utils/     # data processing and contrastive bound
```

## Environment (typical)

- Python 3.8+
- PyTorch, PyTorch Geometric, DGL
- numpy, scipy, scikit-learn, tqdm

## Data

Public benchmarks (Cora, Citeseer, Pubmed, Cornell, Chameleon, Squirrel) are **not** shipped here.
Put processed graphs under `data/` (Planetoid / WebKB layout as used by `MDGFM.py`).

## Run

```bash
python MDGFM.py --help
# or
python runexp.py --help
```

## Note

This repo keeps **main experiment model code** (entry + modules it imports).  
It does not include local experiment shell suites, checkpoints, or datasets.
