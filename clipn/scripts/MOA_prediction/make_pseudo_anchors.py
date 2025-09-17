#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create pseudo-anchors (unsupervised clusters) for prototype MOA scoring.
-------------------------------------------------------------------------

Clusters aggregated compound embeddings and writes a 2-column TSV:
[id_col, moa], where moa is 'Cluster_0001', 'Cluster_0002', ...

- Input can be well-level; we aggregate to compound using a robust method.
- Clustering:
    * 'auto' (default): HDBSCAN-first; if unavailable or degenerate, fall back to KMeans with auto-k.
    * 'hdbscan': force HDBSCAN (error if not installed).
    * 'kmeans': force KMeans (supports fixed k or auto-k).

Outputs (TSV; never comma-separated):
- anchors_pseudo.tsv : columns [id_col, moa]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ------------------------------- maths helpers ------------------------------- #

def l2_normalise(*, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalisation.

    Parameters
    ----------
    X : np.ndarray
        Matrix (n_samples × n_features).
    eps : float, optional
        Numerical stabiliser added to norms, by default 1e-12.

    Returns
    -------
    np.ndarray
        Row-normalised matrix.
    """
    n = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    return X / n


def geometric_median(*, X: np.ndarray, max_iter: int = 256, tol: float = 1e-6) -> np.ndarray:
    """
    Compute the geometric median of points using Weiszfeld's algorithm.

    Parameters
    ----------
    X : np.ndarray
        Points of shape (n_points, n_features).
    max_iter : int, optional
        Maximum iterations, by default 256.
    tol : float, optional
        Convergence tolerance, by default 1e-6.

    Returns
    -------
    np.ndarray
        1D array of length n_features representing the geometric median.
    """
    if X.shape[0] == 1:
        return X[0].copy()
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        if np.any(d < 1e-12):
            return X[np.argmin(d)].copy()
        w = 1.0 / d
        y_new = np.average(X, axis=0, weights=w)
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


def trimmed_mean(*, X: np.ndarray, trim_frac: float = 0.1) -> np.ndarray:
    """
    Compute a per-feature trimmed mean.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n_points, n_features).
    trim_frac : float, optional
        Fraction to trim at each tail, by default 0.1.

    Returns
    -------
    np.ndarray
        1D trimmed-mean vector.
    """
    if X.shape[0] == 1 or trim_frac <= 0:
        return X.mean(axis=0)
    lo = int(np.floor(trim_frac * X.shape[0]))
    hi = int(np.ceil((1 - trim_frac) * X.shape[0]))
    Xs = np.sort(X, axis=0)
    Xc = Xs[lo:hi, :]
    return Xc.mean(axis=0)


# -------------------------------- I/O helpers -------------------------------- #

def detect_id_column(*, df: pd.DataFrame, id_col: Optional[str]) -> str:
    """
    Detect or validate the identifier column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_col : Optional[str]
        Desired id column name, if provided.

    Returns
    -------
    str
        The resolved identifier column name.

    Raises
    ------
    ValueError
        If no suitable id column is found.
    """
    if id_col is not None:
        if id_col in df.columns:
            return id_col
        raise ValueError(f"Identifier column '{id_col}' not found.")
    for c in ["cpd_id", "compound_id", "Compound", "compound", "QueryID", "id"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect an identifier column (tried common candidates).")


# ------------------------ replicate aggregation (to compounds) ---------------- #

def aggregate_compounds(
    *,
    df: pd.DataFrame,
    id_col: str,
    method: str = "median",
    trimmed_frac: float = 0.1,
) -> pd.DataFrame:
    """
    Aggregate replicate rows per compound into a single embedding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least id_col and numeric embedding columns.
    id_col : str
        Identifier column for grouping (e.g., 'cpd_id').
    method : str, optional
        Aggregation method: 'median', 'mean', 'trimmed_mean', 'geometric_median',
        by default 'median'.
    trimmed_frac : float, optional
        Trimming fraction for 'trimmed_mean', by default 0.1.

    Returns
    -------
    pd.DataFrame
        One row per compound with aggregated numeric columns and the id_col.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows: List[Dict[str, float]] = []
    for cid, sub in df.groupby(id_col, sort=False):
        X = sub[num_cols].to_numpy()
        if method == "median":
            vec = np.median(X, axis=0)
        elif method == "mean":
            vec = np.mean(X, axis=0)
        elif method == "trimmed_mean":
            vec = trimmed_mean(X=X, trim_frac=trimmed_frac)
        elif method == "geometric_median":
            vec = geometric_median(X=X)
        else:
            raise ValueError(f"Unknown aggregate_method '{method}'")
        rows.append({"__id__": cid, **{c: float(v) for c, v in zip(num_cols, vec)}})
    out = pd.DataFrame(rows).rename(columns={"__id__": id_col})
    return out[[id_col] + num_cols]


# --------------------------------- clustering -------------------------------- #

def try_hdbscan(
    *,
    X: np.ndarray,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
) -> Optional[np.ndarray]:
    """
    Attempt HDBSCAN clustering. Returns labels or None if unavailable/fails.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    min_cluster_size : Optional[int]
        HDBSCAN min_cluster_size. If None, a heuristic is used.
    min_samples : Optional[int]
        HDBSCAN min_samples. If None, HDBSCAN's default is used.

    Returns
    -------
    Optional[np.ndarray]
        Labels array of length n_samples (with -1 for noise) or None on failure.
    """
    try:
        import hdbscan  # type: ignore
    except Exception:
        return None

    n = X.shape[0]
    mcs = int(max(5, min_cluster_size if min_cluster_size is not None else np.ceil(0.01 * n)))
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, prediction_data=False)
        labels = clusterer.fit_predict(X)
        return labels.astype(int)
    except Exception:
        return None


def handle_noise(
    *,
    X: np.ndarray,
    labels: np.ndarray,
    strategy: str = "own_cluster",
) -> np.ndarray:
    """
    Handle HDBSCAN noise labels (-1) according to a chosen strategy.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    labels : np.ndarray
        Labels with possible -1 values for noise.
    strategy : str, optional
        One of {"own_cluster", "drop", "nearest"}, by default "own_cluster".

    Returns
    -------
    np.ndarray
        Updated label array. For "drop", returned labels exclude dropped rows.

    Notes
    -----
    - "own_cluster": every noise point becomes its own singleton cluster.
    - "drop": noise points are removed entirely (caller must filter rows).
    - "nearest": assign noise to the nearest (cosine) non-noise centroid.
    """
    assert strategy in {"own_cluster", "drop", "nearest"}
    if (-1 not in labels):
        return labels

    noise_mask = labels == -1
    if strategy == "drop":
        # Indicate drop by setting to a special large negative code.
        # Caller will filter them out.
        lbls = labels.copy()
        lbls[noise_mask] = -999_999
        return lbls

    if strategy == "own_cluster":
        lbls = labels.copy()
        non_noise = lbls != -1
        start = (lbls[non_noise].max(initial=-1) + 1) if non_noise.any() else 0
        lbls[noise_mask] = np.arange(start, start + noise_mask.sum(), dtype=int)
        return lbls

    # nearest: assign to nearest non-noise centroid (cosine similarity)
    lbls = labels.copy()
    non_noise_mask = lbls != -1
    if not non_noise_mask.any():
        # All noise: make each its own cluster
        start = 0
        lbls[noise_mask] = np.arange(start, start + noise_mask.sum(), dtype=int)
        return lbls

    uniq = np.unique(lbls[non_noise_mask])
    centroids = []
    for u in uniq:
        idx = np.where(lbls == u)[0]
        c = X[idx, :].mean(axis=0)
        nrm = np.linalg.norm(c)
        centroids.append(c / nrm if nrm > 0 else c)
    C = np.vstack(centroids)
    # assign each noise point to argmax cosine sim
    Ns = X[noise_mask, :]
    sims = Ns @ C.T
    assign = np.argmax(sims, axis=1)
    lbls[noise_mask] = uniq[assign]
    return lbls


def auto_kmeans(
    *,
    X: np.ndarray,
    random_seed: int,
    min_k: int,
    max_k: int,
    sample_size: int = 2000,
) -> Tuple[np.ndarray, int]:
    """
    Run KMeans with auto selection of k via cosine-silhouette on a subsample.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    random_seed : int
        Random seed.
    min_k : int
        Minimum k to consider (≥2).
    max_k : int
        Maximum k to consider (≥ min_k + 1).
    sample_size : int, optional
        Subsample size for silhouette evaluation, by default 2000.

    Returns
    -------
    Tuple[np.ndarray, int]
        (labels, chosen_k)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int), 1

    # Build a compact grid of candidate ks
    ks: List[int] = sorted(set([
        max(2, min_k),
        int(round(np.sqrt(n))),
        int(round(1.5 * np.sqrt(n))),
        int(round(2.0 * np.sqrt(n))),
        max_k,
    ]))
    ks = [k for k in ks if 2 <= k <= min(max_k, n - 1)]
    if not ks:
        ks = [min(2, n - 1)]

    # Subsample indices (without replacement) for silhouette
    rng = np.random.default_rng(random_seed)
    if n > sample_size:
        sample_idx = rng.choice(n, size=sample_size, replace=False)
    else:
        sample_idx = np.arange(n)

    best_k = None
    best_score = -np.inf
    best_labels = None

    for k in ks:
        try:
            km = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
            labels = km.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X[sample_idx, :], labels[sample_idx], metric="cosine")
            if (score > best_score) or (np.isclose(score, best_score) and (best_k is None or k < best_k)):
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        # Fallback: a reasonable k
        k = min(max(2, int(round(np.sqrt(n)))), n - 1)
        km = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        best_labels = km.fit_predict(X)
        best_k = k

    return best_labels.astype(int), int(best_k)


def force_kmeans(*, X: np.ndarray, k: int, random_seed: int) -> np.ndarray:
    """
    Run KMeans with a fixed k.

    Parameters
    ----------
    X : np.ndarray
        Normalised data.
    k : int
        Number of clusters.
    random_seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Labels array (length n_samples).
    """
    from sklearn.cluster import KMeans

    k_eff = max(2, min(k, X.shape[0]))
    km = KMeans(n_clusters=k_eff, random_state=random_seed, n_init="auto")
    return km.fit_predict(X).astype(int)


# ---------------------------------- main ------------------------------------ #

def main() -> None:
    """
    Parse arguments, cluster embeddings into pseudo-anchors, and write TSV.

    Notes
    -----
    - Outputs are always TSV (tab-separated) to comply with “never comma-separated”.
    - 'auto' mode: HDBSCAN-first (if installed), then KMeans with auto-k if needed.
    """
    parser = argparse.ArgumentParser(description="Make pseudo-anchors by clustering embeddings (TSV in/out, UK English).")
    parser.add_argument("--embeddings_tsv", type=str, required=True, help="TSV with embeddings (well- or compound-level).")
    parser.add_argument("--out_anchors_tsv", type=str, required=True, help="Output TSV path for pseudo-anchors.")
    parser.add_argument("--id_col", type=str, default="cpd_id", help="Identifier column name (default: cpd_id).")

    parser.add_argument(
        "--aggregate_method",
        type=str,
        default="median",
        choices=["median", "mean", "trimmed_mean", "geometric_median"],
        help="Replicate aggregation method.",
    )
    parser.add_argument("--trimmed_frac", type=float, default=0.1, help="Trim fraction for trimmed_mean (default: 0.1).")

    parser.add_argument(
        "--clusterer",
        type=str,
        default="auto",
        choices=["auto", "kmeans", "hdbscan"],
        help="Clustering algorithm: 'auto' (HDBSCAN-first, KMeans fallback), 'kmeans', or 'hdbscan'.",
    )
    parser.add_argument("--n_clusters", type=int, default=-1, help="KMeans clusters (use -1 for auto-k).")

    parser.add_argument("--auto_min_clusters", type=int, default=8, help="Lower bound for auto-k search (default: 8).")
    parser.add_argument("--auto_max_clusters", type=int, default=64, help="Upper bound for auto-k search (default: 64).")
    parser.add_argument("--silhouette_sample_size", type=int, default=2000, help="Sample size for silhouette (default: 2000).")

    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=-1, help="HDBSCAN min_cluster_size (<=0 for auto).")
    parser.add_argument("--hdbscan_min_samples", type=int, default=-1, help="HDBSCAN min_samples (<=0 for default).")
    parser.add_argument(
        "--hdbscan_noise",
        type=str,
        default="own_cluster",
        choices=["own_cluster", "drop", "nearest"],
        help="What to do with HDBSCAN noise (-1): keep as singletons, drop, or assign to nearest cluster.",
    )

    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Load and aggregate
    df = pd.read_csv(args.embeddings_tsv, sep="\t")
    id_col = detect_id_column(df=df, id_col=args.id_col)
    agg = aggregate_compounds(
        df=df,
        id_col=id_col,
        method=args.aggregate_method,
        trimmed_frac=args.trimmed_frac,
    )
    num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    X = l2_normalise(X=agg[num_cols].to_numpy())
    ids = agg[id_col].astype(str).to_numpy()
    n = X.shape[0]

    # Decide clustering path
    labels: Optional[np.ndarray] = None
    chosen_algo = None
    chosen_k = None

    if args.clusterer in {"auto", "hdbscan"}:
        labels_hdb = try_hdbscan(
            X=X,
            min_cluster_size=(args.hdbscan_min_cluster_size if args.hdbscan_min_cluster_size > 0 else None),
            min_samples=(args.hdbscan_min_samples if args.hdbscan_min_samples > 0 else None),
        )
        if labels_hdb is not None:
            lbls = handle_noise(X=X, labels=labels_hdb, strategy=args.hdbscan_noise)

            if args.hdbscan_noise == "drop":
                keep_mask = lbls != -999_999
                X_used = X[keep_mask, :]
                ids_used = ids[keep_mask]
                lbls = lbls[keep_mask]
            else:
                X_used = X
                ids_used = ids

            # Check degeneracy: at least 2 clusters
            if len(np.unique(lbls)) >= 2:
                labels = lbls
                chosen_algo = "hdbscan"
                # forward X/ids for naming
                X = X_used
                ids = ids_used

    if labels is None:
        # KMeans path
        chosen_algo = "kmeans"
        if args.n_clusters is not None and args.n_clusters > 1 and args.clusterer != "auto":
            labels = force_kmeans(X=X, k=args.n_clusters, random_seed=args.random_seed)
            chosen_k = int(args.n_clusters)
        else:
            # Auto-k range
            min_k = max(2, min(args.auto_min_clusters, n - 1))
            # Allow auto_max up to n-1, but cap to something reasonable
            max_k = min(max(args.auto_max_clusters, min_k + 1), n - 1)
            labels, chosen_k = auto_kmeans(
                X=X,
                random_seed=args.random_seed,
                min_k=min_k,
                max_k=max_k,
                sample_size=max(100, args.silhouette_sample_size),
            )

    # Build names Cluster_0001, Cluster_0002, ...
    uniq = np.unique(labels)
    mapping = {lab: f"Cluster_{i + 1:04d}" for i, lab in enumerate(sorted(uniq))}
    moa = [mapping[int(lab)] for lab in labels]

    out = pd.DataFrame({id_col: ids, "moa": moa})
    Path(args.out_anchors_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_anchors_tsv, sep="\t", index=False)


if __name__ == "__main__":
    main()
