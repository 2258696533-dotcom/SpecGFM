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

## Data

Public graph benchmarks (Cora, Citeseer, Pubmed, Cornell, Chameleon, Squirrel) are **not** included.
Place processed data under `data/` locally (or follow your existing Planetoid/WebKB paths).

## Note

This repository contains the **main experiment model code only** (not full experiment logs, paper drafts, or one-off run scripts).
