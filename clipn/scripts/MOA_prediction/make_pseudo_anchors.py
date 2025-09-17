#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create pseudo-anchors (unsupervised clusters) for prototype MOA scoring.
-------------------------------------------------------------------------

This script clusters aggregated compound embeddings and writes:
1) anchors TSV mapping each compound to a pseudo-MOA cluster;
2) a one-row run summary TSV with algorithm/quality metrics;
3) a per-cluster stats TSV.

- Input can be well-level; we aggregate to compound using a robust method.
- Clustering:
    * 'auto' (default): HDBSCAN-first; if unavailable/degenerate, fall back to KMeans with auto-k.
    * 'hdbscan': force HDBSCAN (error if not installed).
    * 'kmeans': force KMeans (supports fixed k or auto-k).

Outputs (TSV; never comma-separated)
------------------------------------
- anchors_pseudo.tsv          : columns [id_col, moa, cluster_id]
- anchors_pseudo_summary.tsv  : one row of run metadata and QC
- anchors_pseudo_clusters.tsv : cluster-level counts
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
    nrm = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    return X / nrm


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
    for cid, sub in df.groupby(by=id_col, sort=False):
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
    out = pd.DataFrame(data=rows).rename(columns={"__id__": id_col})
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
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
    Tuple[np.ndarray, Optional[np.ndarray]]
        (labels_out, keep_mask)
        - labels_out: updated labels; for "drop" the returned labels have no
          sentinel codes; for others length equals X.shape[0].
        - keep_mask: for "drop", a boolean mask of kept rows; otherwise None.
    """
    assert strategy in {"own_cluster", "drop", "nearest"}
    if (-1 not in labels):
        return labels, None

    noise_mask = labels == -1
    if strategy == "drop":
        keep_mask = ~noise_mask
        return labels[keep_mask], keep_mask

    if strategy == "own_cluster":
        lbls = labels.copy()
        non_noise = lbls != -1
        start = (lbls[non_noise].max(initial=-1) + 1) if non_noise.any() else 0
        lbls[noise_mask] = np.arange(start, start + noise_mask.sum(), dtype=int)
        return lbls, None

    # "nearest": assign noise to nearest non-noise centroid by cosine
    lbls = labels.copy()
    non_noise_mask = lbls != -1
    if not non_noise_mask.any():
        # All noise: make each its own cluster
        start = 0
        lbls[noise_mask] = np.arange(start, start + noise_mask.sum(), dtype=int)
        return lbls, None

    uniq = np.unique(lbls[non_noise_mask])
    centroids = []
    for u in uniq:
        idx = np.where(lbls == u)[0]
        c = X[idx, :].mean(axis=0)
        nrm = np.linalg.norm(c)
        centroids.append(c / nrm if nrm > 0 else c)
    C = np.vstack(centroids)
    sims = X[noise_mask, :] @ C.T
    assign = np.argmax(sims, axis=1)
    lbls[noise_mask] = uniq[assign]
    return lbls, None


def auto_kmeans(
    *,
    X: np.ndarray,
    random_seed: int,
    min_k: int,
    max_k: int,
    sample_size: int = 2000,
) -> Tuple[np.ndarray, int]:
    """
    Run KMeans with auto selection of k via cosine silhouette on a subsample.

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
        return np.zeros(shape=(n,), dtype=int), 1

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
        from sklearn.cluster import KMeans  # re-import inside function for clarity
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


def cosine_silhouette(
    *,
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> float:
    """
    Compute cosine silhouette (optionally on a subsample). NaN if undefined.

    Parameters
    ----------
    X : np.ndarray
        Normalised data.
    labels : np.ndarray
        Cluster labels.
    sample_size : int
        Subsample size for efficiency.
    random_seed : int
        Random seed.

    Returns
    -------
    float
        Silhouette score in [-1, 1], or NaN if <2 clusters.
    """
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2 or X.shape[0] < 2:
        return float("nan")
    rng = np.random.default_rng(random_seed)
    if X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        return float(silhouette_score(X[idx, :], labels[idx], metric="cosine"))
    return float(silhouette_score(X, labels, metric="cosine"))


# ---------------------------------- main ------------------------------------ #

def main() -> None:
    """
    Parse arguments, cluster embeddings into pseudo-anchors, and write TSVs.

    Notes
    -----
    - Outputs are always TSV (tab-separated) to comply with “never comma-separated”.
    - 'auto' mode: HDBSCAN-first (if installed), then KMeans with auto-k if needed.
    """
    parser = argparse.ArgumentParser(description="Make pseudo-anchors by clustering embeddings (TSV in/out, UK English).")
    parser.add_argument("--embeddings_tsv", type=str, required=True, help="TSV with embeddings (well- or compound-level).")
    parser.add_argument("--out_anchors_tsv", type=str, required=True, help="Output TSV path for pseudo-anchors.")
    parser.add_argument("--out_summary_tsv", type=str, default="", help="Optional TSV path for run summary. Defaults next to anchors.")
    parser.add_argument("--out_clusters_tsv", type=str, default="", help="Optional TSV path for per-cluster stats. Defaults next to anchors.")
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

    # Resolve paths
    anchors_path = Path(args.out_anchors_tsv)
    default_summary = anchors_path.with_name("anchors_pseudo_summary.tsv")
    default_clusters = anchors_path.with_name("anchors_pseudo_clusters.tsv")
    summary_path = Path(args.out_summary_tsv) if args.out_summary_tsv else default_summary
    clusters_path = Path(args.out_clusters_tsv) if args.out_clusters_tsv else default_clusters
    anchors_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and aggregate
    df = pd.read_csv(filepath_or_buffer=args.embeddings_tsv, sep="\t")
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
    n_in = int(X.shape[0])

    # Decide clustering path
    labels: Optional[np.ndarray] = None
    chosen_algo: str = ""
    chosen_k: Optional[int] = None
    noise_handling_used: Optional[str] = None
    dropped_count = 0

    if args.clusterer in {"auto", "hdbscan"}:
        labels_hdb = try_hdbscan(
            X=X,
            min_cluster_size=(args.hdbscan_min_cluster_size if args.hdbscan_min_cluster_size > 0 else None),
            min_samples=(args.hdbscan_min_samples if args.hdbscan_min_samples > 0 else None),
        )
        if labels_hdb is not None:
            lbls, keep_mask = handle_noise(X=X, labels=labels_hdb, strategy=args.hdbscan_noise)
            noise_handling_used = args.hdbscan_noise
            if keep_mask is not None:
                X = X[keep_mask, :]
                ids = ids[keep_mask]
                dropped_count = int((~keep_mask).sum())
            if len(np.unique(lbls)) >= 2:
                labels = lbls
                chosen_algo = "hdbscan"

    if labels is None:
        # KMeans path (forced or fallback)
        chosen_algo = "kmeans"
        noise_handling_used = "n/a"
        if args.clusterer == "kmeans" and args.n_clusters and args.n_clusters > 1:
            labels = force_kmeans(X=X, k=args.n_clusters, random_seed=args.random_seed)
            chosen_k = int(args.n_clusters)
        else:
            # Auto-k range bounded and sane
            n_used = X.shape[0]
            min_k = max(2, min(args.auto_min_clusters, n_used - 1))
            max_k = min(max(args.auto_max_clusters, min_k + 1), n_used - 1)
            labels, chosen_k = auto_kmeans(
                X=X,
                random_seed=args.random_seed,
                min_k=min_k,
                max_k=max_k,
                sample_size=max(100, args.silhouette_sample_size),
            )

    # Build sequential cluster IDs and names
    uniq = sorted(np.unique(labels).tolist())
    remap = {lab: i for i, lab in enumerate(uniq)}
    cluster_id = np.array([remap[int(lab)] for lab in labels], dtype=int)
    moa = [f"Cluster_{i + 1:04d}" for i in cluster_id]

    # Compute QC metrics
    n_used = int(X.shape[0])
    num_clusters = int(len(uniq))
    sil = cosine_silhouette(
        X=X,
        labels=cluster_id,
        sample_size=args.silhouette_sample_size,
        random_seed=args.random_seed,
    )

    # Write anchors: cpd_id → cluster
    anchors_df = pd.DataFrame(data={id_col: ids, "moa": moa, "cluster_id": cluster_id})
    anchors_df.to_csv(path_or_buf=anchors_path, sep="\t", index=False)

    # Per-cluster stats
    counts = pd.Series(data=cluster_id, dtype=int).value_counts().sort_index()
    clust_df = pd.DataFrame(
        data={
            "cluster_id": counts.index.astype(int),
            "moa": [f"Cluster_{i + 1:04d}" for i in counts.index.astype(int)],
            "n_members": counts.values.astype(int),
        }
    )
    clust_df.to_csv(path_or_buf=clusters_path, sep="\t", index=False)

    # One-row summary
    summary = pd.DataFrame(
        data=[{
            "algorithm": chosen_algo,
            "clusterer_requested": args.clusterer,
            "k_if_kmeans": (chosen_k if chosen_algo == "kmeans" else np.nan),
            "num_clusters": num_clusters,
            "n_samples_input": n_in,
            "n_samples_used": n_used,
            "dropped_count": dropped_count,
            "noise_handling": (noise_handling_used if noise_handling_used is not None else "n/a"),
            "silhouette_cosine": sil,
            "aggregate_method": args.aggregate_method,
            "trimmed_frac": args.trimmed_frac if args.aggregate_method == "trimmed_mean" else np.nan,
            "auto_min_clusters": args.auto_min_clusters,
            "auto_max_clusters": args.auto_max_clusters,
            "n_clusters_requested": args.n_clusters,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "random_seed": args.random_seed,
        }]
    )
    summary.to_csv(path_or_buf=summary_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
