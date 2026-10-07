"""ALL-IN feature adapter utilities.

This module provides a lightweight ALL-IN-inspired preprocessing path that can be
plugged into MDGFM without changing the original default behavior.

Pipeline (ALL-IN branch):
1) Shared random projection: R = X @ C
2) Center projected node features: H = R - mean(R)
3) Compress H to fixed `out_dim` (PCA), keeping node-feature shape [n, out_dim]
   so downstream MDGFM code remains unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _to_numpy(features) -> np.ndarray:
    """Convert tensor/array features to float64 numpy array."""
    if hasattr(features, "detach"):
        arr = features.detach().cpu().numpy()
    else:
        arr = np.asarray(features)
    return arr.astype(np.float64, copy=False)


def allin_project_and_compress(
    features,
    out_dim: int,
    rp_dim: int = 512,
    seed: int = 1024,
    post_norm: str = "none",
    eps: float = 1e-8,
) -> np.ndarray:
    """ALL-IN-inspired feature adaptation.

    Args:
        features: Node feature matrix [num_nodes, feat_dim].
        out_dim: Final feature dimension expected by MDGFM (e.g., 50).
        rp_dim: Random projection width (h in ALL-IN style pipeline).
        seed: Random seed for deterministic projection.
        post_norm: Optional post-normalization on compressed features.
            - "none": keep raw PCA output.
            - "zscore": per-dimension standardization.
        eps: Numeric stability term for normalization.

    Returns:
        A numpy array with shape [num_nodes, out_dim].
    """
    x = _to_numpy(features)
    n, d = x.shape

    # [ALL-IN] Step 1: random projection to a shared feature space.
    rng = np.random.RandomState(seed)
    c = rng.normal(loc=0.0, scale=1.0 / np.sqrt(max(d, 1)), size=(d, rp_dim))
    r0 = x @ c  # [n, rp_dim]

    # [ALL-IN] Step 2: center projected node features.
    h = r0 - r0.mean(axis=0, keepdims=True)  # [n, rp_dim]

    # Step 3: compress projected node features (not n x n node covariance).
    # This keeps representation semantics aligned with MDGFM's node feature input.
    final_dim = min(out_dim, n, h.shape[1])
    pca = PCA(n_components=final_dim)
    z = pca.fit_transform(h)

    # If graph is very small and final_dim < out_dim, right-pad with zeros.
    if final_dim < out_dim:
        pad = np.zeros((n, out_dim - final_dim), dtype=z.dtype)
        z = np.concatenate([z, pad], axis=1)
    if post_norm == "zscore":
        mean = z.mean(axis=0, keepdims=True)
        std = z.std(axis=0, keepdims=True)
        z = (z - mean) / (std + eps)
    return z

