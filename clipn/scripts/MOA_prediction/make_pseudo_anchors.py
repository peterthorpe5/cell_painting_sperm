#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create pseudo-anchors (unsupervised clusters) for prototype MOA scoring.
-------------------------------------------------------------------------

This script clusters aggregated compound embeddings and writes:
1) anchors TSV mapping each compound to a pseudo-MOA cluster;
2) a one-row run summary TSV with algorithm/quality metrics;
3) a per-cluster stats TSV including per-cluster cosine silhouette and
   nearest-centroid cosine (and nearest cluster id).

- Input can be well-level; we aggregate to compound using a robust method.
- Clustering:
    * 'kmeans' (default): supports fixed k or auto-k via cosine silhouette.
    * 'auto': HDBSCAN-first; if unavailable/degenerate, fall back to KMeans auto-k.
    * 'hdbscan': force HDBSCAN (error if not installed).

Outputs (TSV; never comma-separated)
------------------------------------
- anchors_pseudo.tsv          : columns [id_col, moa, cluster_id]
- anchors_pseudo_summary.tsv  : one row of run metadata and QC
- anchors_pseudo_clusters.tsv : per-cluster stats:
    [cluster_id, moa, n_members, silhouette_cosine,
     nearest_cluster_id, nearest_centroid_cosine]

In this script, silhouette is a quality score for clustering computed with cosine distance on the L2-normalised embeddings.
     
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
import logging
import sys



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
    num_cols = numeric_feature_columns(df)
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


# ------------------------------- clustering utils ---------------------------- #

def cosine_silhouette_scores(*, X: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute overall and per-sample cosine silhouette scores.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    labels : np.ndarray
        Cluster labels (length n_samples).

    Returns
    -------
    Tuple[float, np.ndarray]
        (overall_silhouette, per_sample_silhouette) where the first is NaN if
        there are <2 clusters with at least 2 members.
    """
    try:
        from sklearn.metrics import silhouette_score, silhouette_samples
    except Exception as exc:
        raise SystemExit("scikit-learn is required for silhouette computation.") from exc

    uniq = np.unique(labels)
    # Need at least two clusters with >=2 members
    valid_clusters = [u for u in uniq if (labels == u).sum() >= 2]
    if len(valid_clusters) < 2:
        return float("nan"), np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=float)

    overall = float(silhouette_score(X=X, labels=labels, metric="cosine"))
    per_sample = silhouette_samples(X=X, labels=labels, metric="cosine")
    return overall, per_sample


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
    strategy: str = "nearest",
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
        One of {"own_cluster", "drop", "nearest"}, by default "nearest".

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

def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the columns that are actual embedding features.

    Priority:
    1) Columns named as plain integers: "0","1","2",...
    2) Else, columns starting with "feat_" followed by digits.
    3) Else, fallback to numeric dtype but EXCLUDE common metadata names.

    This prevents leaking columns like 'Sample' into the embedding.
    """
    cols = [str(c) for c in df.columns]

    # 1) strictly-integer column names
    int_like = [c for c in cols if c.isdigit()]
    if int_like:
        return sorted(int_like, key=lambda s: int(s))

    # 2) feat_### pattern
    feat_like = [c for c in cols if re.fullmatch(r"feat_\d+", c)]
    if feat_like:
        # keep order as they appear
        return feat_like

    # 3) fallback: numeric dtype minus known metadata
    meta_blocklist = {
        "sample", "dataset", "plate", "well", "row", "col",
        "time", "replicate", "dose", "concentration",
        "cpd_id", "compound_id", "id"
    }
    num_cols = df.select_dtypes(include=[np.number]).columns
    keep = [c for c in num_cols if c.lower() not in meta_blocklist]
    return keep



def overlay_given_labels(
    *,
    clusters_df: Optional[pd.DataFrame] = None,
    anchors: Optional[pd.DataFrame] = None,
    labels_tsv: str,
    id_col: str,
    labels_id_col: Optional[str] = None,
    labels_label_col: str = "label",
) -> pd.DataFrame:
    """
    Merge user-provided labels onto cluster assignments and derive a final MOA.

    Backwards compatible with previous usage:
    `_overlay_given_labels(anchors=..., labels_tsv=..., id_col=..., labels_id_col=..., labels_label_col=...)`
    or the newer `clusters_df=...`.

    Parameters
    ----------
    clusters_df : Optional[pd.DataFrame]
        DataFrame with per-compound cluster assignment. Must contain at least
        [id_col, "moa"]. If available, "cluster_id" will also be preserved.
    anchors : Optional[pd.DataFrame]
        Alias for `clusters_df` (for backwards compatibility).
    labels_tsv : str
        Path to a TSV with at least [id_col (or labels_id_col), labels_label_col].
        May contain multiple rows per compound; these will be collapsed.
    id_col : str
        Identifier column name used in `clusters_df` / embeddings (e.g., 'cpd_id').
    labels_id_col : Optional[str]
        Column name in labels_tsv that corresponds to the identifier. If None,
        defaults to `id_col`.
    labels_label_col : str
        Column in labels_tsv containing the human-provided label (default 'label').

    Returns
    -------
    pd.DataFrame
        Columns: [id_col, "moa", "cluster_id", "given_label", "is_labelled", "moa_final"].
    """
    # Accept both parameter names for the clustered table
    if clusters_df is None and anchors is None:
        raise ValueError("Provide either clusters_df or anchors.")
    df = clusters_df if clusters_df is not None else anchors
    df = df.copy()

    # Ensure id is a proper column (not only an index)
    if df.index.name == id_col:
        df = df.reset_index()
    if id_col not in df.columns:
        if df.index.name is None:
            df = df.reset_index().rename(columns={"index": id_col})
        else:
            raise KeyError(f"'{id_col}' is neither a column nor the index in the cluster table.")

    # Normalise id and guarantee 'cluster_id'
    df[id_col] = df[id_col].astype(str).str.strip()
    if "cluster_id" not in df.columns:
        df["cluster_id"] = pd.NA

    # Load labels
    lab = pd.read_csv(labels_tsv, sep="\t", dtype=str, keep_default_na=False, na_values=[""], 
                        low_memory=False)
    src_id = labels_id_col if labels_id_col is not None else id_col
    if src_id not in lab.columns:
        raise ValueError(
            f"Labels file '{labels_tsv}' must contain an id column "
            f"('{src_id}' or '{id_col}'). Found columns: {list(lab.columns)}"
        )
    if labels_label_col not in lab.columns:
        raise ValueError(
            f"Labels file '{labels_tsv}' must contain label column '{labels_label_col}'. "
            f"Found columns: {list(lab.columns)}"
        )

    if src_id != id_col:
        lab = lab.rename(columns={src_id: id_col})
    lab[id_col] = lab[id_col].astype(str).str.strip()
    lab[labels_label_col] = lab[labels_label_col].astype(str).str.strip()

    # Collapse multiple labels per compound (keep unique; stable order)
    lab_agg = (
        lab.groupby(id_col, sort=False)[labels_label_col]
           .apply(lambda s: "; ".join(pd.unique([x for x in s.values if x != ""])))
           .reset_index()
           .rename(columns={labels_label_col: "given_label"})
    )

    # Note any labels that do not appear in clustered set
    missing = set(lab_agg[id_col]) - set(df[id_col])
    if missing:
        print(f"NOTE: {len(missing)} labelled compounds not in clustered set; ignoring.", flush=True)

    # Join
    out = df.merge(lab_agg, on=id_col, how="left", validate="one_to_one")

    # Flags
    out["given_label"] = out["given_label"].replace("", np.nan)
    out["is_labelled"] = out["given_label"].notna()

    # Final MOA: keep cluster MOA and append label(s) if present and different.
    base = out["moa"].astype(str).str.strip().fillna("")
    labv = out["given_label"].astype(str).str.strip().fillna("")
    same = base.str.lower() == labv.str.lower()
    out["moa_final"] = np.where(
        (labv.eq("")) | same,
        base,
        base + " | " + labv,
    )

    return out[[id_col, "moa", "cluster_id", "given_label", "is_labelled", "moa_final"]]



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

    # Candidate ks
    ks: List[int] = sorted(set([
        max(2, min_k),
        int(round(np.sqrt(n))),
        int(round(1.5 * np.sqrt(n))),
        int(round(2.0 * np.sqrt(n))),
        min(max_k, max(2, n - 1)),
    ]))
    ks = [k for k in ks if 2 <= k <= min(max_k, n - 1)]
    if not ks:
        ks = [min(2, n - 1)]

    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(n, size=min(n, sample_size), replace=False)

    best_k: Optional[int] = None
    best_score = -np.inf
    best_labels: Optional[np.ndarray] = None

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X=X[sample_idx, :], labels=labels[sample_idx], metric="cosine")
        if (score > best_score) or (np.isclose(score, best_score) and (best_k is None or k < best_k)):
            best_score = float(score)
            best_k = int(k)
            best_labels = labels

    if best_labels is None:
        # Fallback: k=2
        from sklearn.cluster import KMeans as _KMeans
        km = _KMeans(n_clusters=2, random_state=random_seed, n_init="auto")
        best_labels = km.fit_predict(X)
        best_k = 2

    return best_labels, int(best_k)


def _parse_k_candidates(*, text: str, n_max: int) -> list[int]:
    """
    Parse a comma-separated candidate-k string into a validated, unique, sorted list.

    Parameters
    ----------
    text : str
        Comma-separated ks (e.g., "8,12,16,24").
    n_max : int
        Upper bound (exclusive) for k (must be < n_max).

    Returns
    -------
    list[int]
        Valid candidate ks (2 <= k <= n_max-1), sorted and unique.
    """
    ks = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
        except ValueError:
            continue
        if 2 <= k <= max(2, n_max - 1):
            ks.append(k)
    ks = sorted(set(ks))
    if not ks:
        ks = [2] if n_max >= 3 else [1]
    return ks


def _consensus_matrix_from_labels(*, labels_runs: list[np.ndarray]) -> np.ndarray:
    """
    Build a co-association (consensus) matrix from repeated label assignments.

    Each element (i,j) is the fraction of runs where sample i and j were assigned
    the same cluster. Assumes each labels array covers the same N samples (we
    predict labels for all rows even if a run was fit on a subsample).

    Parameters
    ----------
    labels_runs : list[np.ndarray]
        List of length R runs; each array has shape (N,) of integer labels.

    Returns
    -------
    np.ndarray
        Consensus matrix of shape (N, N) with values in [0, 1].
    """
    if not labels_runs:
        raise ValueError("labels_runs is empty.")
    N = int(labels_runs[0].shape[0])
    R = len(labels_runs)
    M = np.zeros((N, N), dtype=np.float32)
    for lab in labels_runs:
        if lab.shape[0] != N:
            raise ValueError("All label arrays must have the same length.")
        # Accumulate agreement: I[label_i == label_j]
        eq = (lab[:, None] == lab[None, :])
        M += eq.astype(np.float32)
    M /= float(R)
    np.fill_diagonal(M, 1.0)
    return M


def _consensus_silhouette(*, consensus: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette on the consensus-derived distance matrix (1 - consensus).

    We first convert the consensus matrix to a dissimilarity matrix D = 1 - C,
    then compute the standard silhouette score with metric='precomputed'.

    Parameters
    ----------
    consensus : np.ndarray
        Consensus matrix (N x N), values in [0, 1].
    labels : np.ndarray
        Integer labels for the clustering used for scoring (length N).

    Returns
    -------
    float
        Mean silhouette score on the consensus distances (NaN if undefined).
    """
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return float("nan")
    if consensus.ndim != 2 or consensus.shape[0] != consensus.shape[1]:
        return float("nan")
    N = consensus.shape[0]
    if labels.shape[0] != N:
        return float("nan")
    # Need at least two clusters with >= 2 members
    uniq, counts = np.unique(labels, return_counts=True)
    valid = (counts >= 2).sum()
    if valid < 2:
        return float("nan")
    D = 1.0 - np.clip(consensus, 0.0, 1.0)
    # Guard: metric='precomputed' expects distances; diagonal must be 0
    np.fill_diagonal(D, 0.0)
    try:
        return float(silhouette_score(X=D, labels=labels, metric="precomputed"))
    except Exception:
        return float("nan")


def _consensus_partition(*, consensus: np.ndarray, n_clusters: int, linkage: str = "average",
                         random_seed: int = 0) -> np.ndarray:
    """
    Derive a consensus partition by clustering the consensus matrix.

    Parameters
    ----------
    consensus : np.ndarray
        Consensus similarity matrix (N x N), values in [0, 1].
    n_clusters : int
        Number of clusters to extract.
    linkage : {"average","complete","single","ward"}
        Linkage for AgglomerativeClustering (default: "average").
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Integer labels of length N.
    """
    from sklearn.cluster import AgglomerativeClustering
    # Convert to distance for clustering where required
    D = 1.0 - np.clip(consensus, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    # 'precomputed' affinity uses distances
    if linkage == "ward":
        # Ward does not support precomputed distances; use features instead.
        # Map similarities to features via spectral-style embedding (top components).
        # Fast fallback: use (N x N) consensus directly as features.
        X_feat = consensus
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        lab = model.fit_predict(X_feat)
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage=linkage
        )
        lab = model.fit_predict(D)
    return lab.astype(int)


def _mean_ari_between_partitions(*, labels_runs: list[np.ndarray]) -> float:
    """
    Compute the mean pairwise Adjusted Rand Index across all runs.

    Parameters
    ----------
    labels_runs : list[np.ndarray]
        List of partitions (arrays of shape (N,)).

    Returns
    -------
    float
        Mean ARI across all unique pairs (NaN if < 2 runs).
    """
    if len(labels_runs) < 2:
        return float("nan")
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception:
        return float("nan")
    s = 0.0
    m = 0
    for i in range(len(labels_runs)):
        for j in range(i + 1, len(labels_runs)):
            s += float(adjusted_rand_score(labels_runs[i], labels_runs[j]))
            m += 1
    return (s / m) if m > 0 else float("nan")


def _pac_score(*, consensus: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    """
    Proportion of Ambiguous Clustering (PAC): fraction of consensus entries in (low, high).

    Lower PAC means more stable clustering.

    Parameters
    ----------
    consensus : np.ndarray
        Consensus matrix (N x N), values in [0, 1].
    low : float
        Lower threshold (default 0.1).
    high : float
        Upper threshold (default 0.9).

    Returns
    -------
    float
        PAC score in [0, 1]; lower is better.
    """
    if consensus.size == 0:
        return float("nan")
    C = consensus.copy()
    # Ignore diagonal
    np.fill_diagonal(C, np.nan)
    amb = np.logical_and(C > low, C < high)
    denom = np.isfinite(C).sum()
    if denom == 0:
        return float("nan")
    return float(np.nan_to_num(amb, nan=0.0).sum() / denom)


def _cap_k_for_duplicates(X: np.ndarray, k: int, logger: "logging.Logger") -> int:
    """
    Ensure k does not exceed the number of distinct samples in X.
    """
    try:
        # For large X, consider hashing rows for speed, but np.unique is okay for modest sizes.
        n_distinct = np.unique(X, axis=0).shape[0]
    except Exception:
        logger.exception("Could not compute distinct rows; leaving k unchanged.")
        return k
    k_eff = max(2, min(k, max(2, n_distinct - 1)))
    if k_eff < k:
        logger.info("Clamping k from %d to %d due to %d distinct rows.", k, k_eff, n_distinct)
    return k_eff


def bootstrap_k_selection_kmeans(
    *,
    X: np.ndarray,
    k_list: list[int],
    n_bootstrap: int,
    subsample_frac: float,
    stability_metric: str,
    consensus_linkage: str,
    random_seed: int,
    pac_low: float = 0.1,
    pac_high: float = 0.9,
) -> tuple[int, pd.DataFrame, dict[int, np.ndarray]]:
    """
    Choose k for KMeans via bootstrap/consensus stability.

    For each k, repeat:
      1) Subsample indices (without replacement) at 'subsample_frac';
      2) Fit KMeans(k) on the subsample (random_state varies per run);
      3) Predict labels for ALL N samples using km.predict(X) to align sizes.

    Build a consensus matrix across runs for the same k and compute a stability score:
      - 'consensus_silhouette': silhouette on (1 - consensus) using a consensus partition;
      - 'mean_ari'           : mean ARI across all run partitions;
      - 'pac'                : lower is better (proportion of ambiguous entries).

    Parameters
    ----------
    X : np.ndarray
        Normalised data (N x D).
    k_list : list[int]
        Candidate k values (2..N-1).
    n_bootstrap : int
        Number of bootstrap replicates per k.
    subsample_frac : float
        Fraction in (0,1]; used to fit KMeans on a subset before predicting all.
    stability_metric : {"consensus_silhouette","mean_ari","pac"}
        Stability metric to optimise (higher is better except PAC).
    consensus_linkage : {"average","complete","single","ward"}
        Linkage for consensus partition.
    random_seed : int
        Random seed.

    pac_low : float
        Lower threshold for PAC (default 0.1).
    pac_high : float
        Upper threshold for PAC (default 0.9).


    Returns
    -------
    tuple[int, pd.DataFrame, dict[int, np.ndarray]]
        (best_k, table, consensus_by_k)
        - best_k: chosen k under the metric (ties → higher silhouette on full data, then smaller k)
        - table : per-k DataFrame with stability metrics and tie-breakers
        - consensus_by_k: optional consensus matrices per k (may be large)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(random_seed)
    N = X.shape[0]
    # Safety
    k_list = [k for k in k_list if 2 <= k <= max(2, N - 1)]
    results = []
    consensus_by_k: dict[int, np.ndarray] = {}
    repr_labels_for_sil: dict[int, np.ndarray] = {}

    for k in k_list:
        labels_runs: list[np.ndarray] = []
        for b in range(n_bootstrap):
            # Subsample to FIT only; PREDICT on all to align size
            m = max(2 * k, int(np.round(subsample_frac * N)))  # ensure enough points
            m = int(np.clip(m, 2 * k, N))
            idx_fit = rng.choice(N, size=m, replace=False)

            seed_b = int(rng.integers(0, 2**32 - 1))
            k_eff = max(2, min(k, X.shape[0] - 1))
            # Optional: respect distinct rows to avoid “distinct clusters < k” warnings
            k_eff = _cap_k_for_duplicates(X[idx_fit, :], k_eff, logging.getLogger("clipn_logger"))
            km = KMeans(n_clusters=k_eff, random_state=seed_b, n_init="auto")

            km.fit(X[idx_fit, :])
            lab_all = km.predict(X)  # length N
            labels_runs.append(lab_all.astype(int))

        # Consensus and stability
        C = _consensus_matrix_from_labels(labels_runs=labels_runs)
        consensus_by_k[k] = C

        # Consensus partition for silhouette metric
        try:
            lab_cons = _consensus_partition(consensus=C, n_clusters=k, linkage=consensus_linkage,
                                            random_seed=random_seed)
        except Exception:
            lab_cons = labels_runs[0]  # fallback to one run
        repr_labels_for_sil[k] = lab_cons

        c_sil = _consensus_silhouette(consensus=C, labels=lab_cons)
        mean_ari = _mean_ari_between_partitions(labels_runs=labels_runs)

        pac = _pac_score(consensus=C, low=pac_low, high=pac_high)


        # Also compute classical silhouette on features (tie-breaker)
        try:
            sil_feat = float(silhouette_score(X=X, labels=lab_cons, metric="cosine"))
        except Exception:
            sil_feat = float("nan")

        results.append({
            "k": int(k),
            "stability_consensus_silhouette": float(c_sil),
            "stability_mean_ari": float(mean_ari),
            "stability_pac": float(pac),
            "silhouette_feature_cosine": sil_feat,
        })

    tab = pd.DataFrame(results).sort_values("k").reset_index(drop=True)

    # Choose best k
    metric = stability_metric
    if metric == "consensus_silhouette":
        key = "stability_consensus_silhouette"
        # Higher is better
        # tab["_rank"] = (-tab[key], -tab["silhouette_feature_cosine"].fillna(-np.inf), tab["k"])
        tab["_s2"] = tab["silhouette_feature_cosine"].fillna(-np.inf)
        tab = tab.sort_values(
            by=[key, "_s2", "k"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        best_k = int(tab["k"].iloc[0])
        tab.drop(columns=["_s2"], inplace=True)

    elif metric == "mean_ari":
        key = "stability_mean_ari"
        # tab["_rank"] = (-tab[key], -tab["silhouette_feature_cosine"].fillna(-np.inf), tab["k"])
        tab["_s2"] = tab["silhouette_feature_cosine"].fillna(-np.inf)
        tab = tab.sort_values(
            by=[key, "_s2", "k"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        best_k = int(tab["k"].iloc[0])
        tab.drop(columns=["_s2"], inplace=True)
    else:  # pac → lower is better
        key = "stability_pac"
        # tab["_rank"] = (tab[key], -tab["silhouette_feature_cosine"].fillna(-np.inf), tab["k"])
        tab["_s2"] = tab["silhouette_feature_cosine"].fillna(-np.inf)
        tab = tab.sort_values(
            by=[key, "_s2", "k"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        best_k = int(tab["k"].iloc[0])
        tab.drop(columns=["_s2"], inplace=True)

    return best_k, tab.reset_index(drop=True), consensus_by_k




def setup_logging(out_dir: str | Path, experiment: str) -> logging.Logger:
    """
    Configure logging with stream (stderr) and file handlers.

    Parameters
    ----------
    out_dir : str | Path
        Output directory for logs.
    experiment : str
        Experiment name; used for the log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{experiment}.log"

    logger = logging.getLogger("clipn_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_filename, mode="w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("Python Version: %s", sys.version_info)
    logger.info("Command-line Arguments: %s", " ".join(sys.argv))


    return logger



# ----------------------------------- main ------------------------------------ #

def main() -> None:
    """
    Parse arguments and write pseudo-anchors TSVs (anchors, summary, clusters).
    """
    parser = argparse.ArgumentParser(description="Make pseudo-anchors by clustering embeddings (TSV in/out, UK English).")
    parser.add_argument("--embeddings_tsv", type=str, required=True, help="TSV with embeddings (well- or compound-level).")
    parser.add_argument("--out_anchors_tsv", type=str, required=True, help="Output TSV path for pseudo-anchors.")
    parser.add_argument("--out_summary_tsv", type=str, required=True, help="Output TSV path for run summary.")
    parser.add_argument("--out_clusters_tsv", type=str, required=True, help="Output TSV path for per-cluster stats.")
    parser.add_argument("--id_col", type=str, default="cpd_id", help="Identifier column name (default: cpd_id).")

    parser.add_argument("--labels_tsv", type=str, default=None,
                        help="Optional TSV mapping IDs to labels for overlay (e.g., cpd_id\\tlabel).")
    parser.add_argument("--labels_id_col", type=str, default="cpd_id",
                        help="ID column name in labels TSV (default: cpd_id).")
    parser.add_argument("--labels_label_col", type=str, default="label",
                        help="Label column name in labels TSV (default: label).")

    parser.add_argument("--aggregate_method", type=str, default="median",
                        choices=["median", "mean", "trimmed_mean", "geometric_median"],
                        help="Replicate aggregation method (default: median).")
    parser.add_argument("--trimmed_frac", type=float, default=0.1, help="Trim fraction for trimmed_mean (default: 0.1).")

    parser.add_argument("--clusterer", type=str, default="kmeans",
                        choices=["kmeans", "auto", "hdbscan"],
                        help="Clustering algorithm: 'kmeans' (default), 'auto' (HDBSCAN-first), or 'hdbscan'.")
    parser.add_argument("--n_clusters", type=int, default=-1, help="KMeans clusters (-1 for auto-k).")
    parser.add_argument("--auto_min_clusters", type=int, default=8, help="Minimum k considered in auto-k (default: 8).")
    parser.add_argument("--auto_max_clusters", type=int, default=64, help="Maximum k considered in auto-k (default: 64).")

    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=-1, help="HDBSCAN min_cluster_size (<=0 → heuristic).")
    parser.add_argument("--hdbscan_min_samples", type=int, default=-1, help="HDBSCAN min_samples (<=0 → default).")
    parser.add_argument("--hdbscan_noise", type=str, default="nearest",
                        choices=["own_cluster", "drop", "nearest"],
                        help="How to handle HDBSCAN noise (default: nearest).")

    parser.add_argument("--hdbscan_min_silhouette", type=float, default=0.02,
                        help="If HDBSCAN cosine silhouette < this, fall back to KMeans.")
    parser.add_argument("--max_singleton_frac", type=float, default=0.15,
                        help="If fraction of singleton clusters > this, fall back to KMeans.")
    parser.add_argument("--silhouette_sample_size", type=int, default=2000,
                        help="Subsample size for auto-k silhouette scoring (default: 2000).")
    
    # --- Bootstrap/consensus k-selection for main clusters ---
    parser.add_argument("--bootstrap_k_main", action="store_true",
                        help="Enable bootstrap/consensus selection of k for main clustering (KMeans path).")
    parser.add_argument("--k_candidates_main", type=str, default="8,12,16,24,32",
                        help="Comma-separated list of k candidates for bootstrap selection (default: 8,12,16,24,32).")
    parser.add_argument("--n_bootstrap_main", type=int, default=100,
                        help="Number of bootstrap replicates per k (default: 100).")
    parser.add_argument("--subsample_main", type=float, default=0.8,
                        help="Fraction of rows to fit KMeans on per bootstrap (then predict all rows; default: 0.8).")
    parser.add_argument("--stability_metric_main", type=str,
                        choices=["consensus_silhouette", "mean_ari", "pac"],
                        default="consensus_silhouette",
                        help="Stability metric to pick k (default: consensus_silhouette).")
    parser.add_argument("--consensus_linkage_main", type=str,
                        choices=["average", "complete", "single", "ward"],
                        default="average",
                        help="Linkage for consensus clustering used in stability scoring (default: average).")
    parser.add_argument("--consensus_pac_limits", type=str, default="0.1,0.9",
                        help="PAC lower,upper thresholds for 'pac' metric (default: 0.1,0.9).")
    parser.add_argument("--out_k_selection_tsv", type=str, default=None,
                        help="Optional path to write per-k stability table (TSV). "
                             "Default: alongside summary as anchors_pseudo_k_selection.tsv.")


    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Setup logging
    for path in [args.out_anchors_tsv, args.out_summary_tsv, args.out_clusters_tsv]:
        p = Path(path)
        if p.exists() and p.is_dir():
            raise SystemExit(f"Output path is a directory (expected file): {p}")
    p.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir=Path(args.out_anchors_tsv).parent, experiment="make_pseudo_anchors")
    logger.info("Starting pseudo-anchor generation.")


    # Load embeddings and aggregate to compounds
    df = pd.read_csv(args.embeddings_tsv, sep="\t", low_memory=False)
    logger.info("Loaded embeddings TSV with %d rows and %d columns.", df.shape[0], df.shape[1])
    id_col = detect_id_column(df=df, id_col=args.id_col)

    agg = aggregate_compounds(
        df=df,
        id_col=id_col,
        method=args.aggregate_method,
        trimmed_frac=args.trimmed_frac,
    )
    logger.info("Aggregated to %d unique compounds using method '%s'.", agg.shape[0], args.aggregate_method)
    # num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = numeric_feature_columns(agg)

    X_all = l2_normalise(X=agg[num_cols].to_numpy())
    ids_all = agg[id_col].astype(str).tolist()

    n_input = int(len(ids_all))

    # Choose clustering path
    chosen_algo = None
    labels: Optional[np.ndarray] = None
    keep_mask: Optional[np.ndarray] = None
    dropped_count = 0
    k_if_kmeans = -1
    noise_handling = "n/a"
    silhouette_overall = float("nan")

    if args.clusterer in {"auto", "hdbscan"}:
        logger.info("Attempting HDBSCAN clustering...")
        # Try HDBSCAN
        labels_h = try_hdbscan(
            X=X_all,
            min_cluster_size=(None if args.hdbscan_min_cluster_size <= 0 else args.hdbscan_min_cluster_size),
            min_samples=(None if args.hdbscan_min_samples <= 0 else args.hdbscan_min_samples),
        )
        if labels_h is not None:
            lbls, keep_mask = handle_noise(X=X_all, labels=labels_h, strategy=args.hdbscan_noise)
            # Quality checks
            uniq, counts = np.unique(lbls, return_counts=True)
            singleton_frac = float((counts == 1).sum()) / float(counts.sum())

            try:
                sil_overall, sil_samples = cosine_silhouette_scores(X=(X_all if keep_mask is None else X_all[keep_mask, :]),
                                                                    labels=lbls)
            except SystemExit:
                sil_overall, sil_samples = float("nan"), np.full(lbls.shape, np.nan)

            ok = True
            if np.isnan(sil_overall) or (sil_overall < float(args.hdbscan_min_silhouette)):
                ok = False
            if singleton_frac > float(args.max_singleton_frac):
                ok = False

            if ok:
                labels = lbls
                chosen_algo = "hdbscan"
                silhouette_overall = sil_overall
                noise_handling = args.hdbscan_noise
                if keep_mask is not None:
                    dropped_count = int((~keep_mask).sum())
            # else: fall through to kmeans
        # elif: fall through to kmeans

    if (labels is None) and (args.clusterer in {"auto", "kmeans"}):
        logger.info("Using KMeans clustering...")

        # Decide k: bootstrap/consensus if requested, else existing auto/fixed
        chosen_k_for_kmeans: Optional[int] = None
        per_k_table: Optional[pd.DataFrame] = None
        out_k_selection_tsv_summary: str = ""

        if bool(args.bootstrap_k_main) and int(args.n_clusters) == -1:
            # Prepare candidates
            k_list = _parse_k_candidates(text=args.k_candidates_main, n_max=X_all.shape[0])
            if not k_list:
                logger.warning("No valid k candidates parsed; falling back to auto silhouette.")
            else:
                # Parse PAC limits
                try:
                    pac_l, pac_u = map(float, str(args.consensus_pac_limits).split(","))
                except Exception:
                    pac_l, pac_u = 0.1, 0.9

                # Run bootstrap/consensus selection
                best_k, k_table, _cons_by_k = bootstrap_k_selection_kmeans(
                                                    X=X_all,
                                                    k_list=k_list,
                                                    n_bootstrap=int(args.n_bootstrap_main),
                                                    subsample_frac=float(args.subsample_main),
                                                    stability_metric=str(args.stability_metric_main),
                                                    consensus_linkage=str(args.consensus_linkage_main),
                                                    random_seed=int(args.random_seed),
                                                    pac_low=pac_l,
                                                    pac_high=pac_u,
                                                )

                per_k_table = k_table.copy()
                chosen_k_for_kmeans = int(best_k)
                logger.info("Bootstrap/consensus selection chose k=%d (metric=%s).",
                            chosen_k_for_kmeans, args.stability_metric_main)

                # Write per-k table if requested
                out_k_tsv = args.out_k_selection_tsv
                if out_k_tsv is None:
                    out_k_tsv = str(Path(args.out_summary_tsv).with_name("anchors_pseudo_k_selection.tsv"))
                    Path(out_k_tsv).parent.mkdir(parents=True, exist_ok=True)
                    per_k_table.to_csv(out_k_tsv, sep="\t", index=False)
                    logger.info("Wrote per-k stability table -> %s", out_k_tsv)

                # Record for summary
                out_k_selection_tsv_summary = out_k_tsv

        from sklearn.cluster import KMeans

        if chosen_k_for_kmeans is not None:
            k_fixed = int(chosen_k_for_kmeans)
            km = KMeans(n_clusters=k_fixed, random_state=args.random_seed, n_init="auto")
            labels = km.fit_predict(X=X_all)
            k_if_kmeans = int(k_fixed)
            chosen_algo = "kmeans_bootstrap"
        else:
            if int(args.n_clusters) == -1:
                labels_k, k_chosen = auto_kmeans(
                    X=X_all,
                    random_seed=args.random_seed,
                    min_k=args.auto_min_clusters,
                    max_k=args.auto_max_clusters,
                    sample_size=args.silhouette_sample_size,
                )
                labels = labels_k
                k_if_kmeans = int(k_chosen)
            else:
                k_fixed = max(2, min(int(args.n_clusters), X_all.shape[0]))
                k_eff = _cap_k_for_duplicates(X_all, k_fixed, logger)
                km = KMeans(n_clusters=k_eff, random_state=args.random_seed, n_init="auto")
                labels = km.fit_predict(X=X_all)
                k_if_kmeans = int(k_eff)
            chosen_algo = "kmeans"


        keep_mask = None  # we used all rows
        noise_handling = "n/a"
        # Overall silhouette on full data
        try:
            sil_overall, _ = cosine_silhouette_scores(X=X_all, labels=labels)
        except SystemExit:
            sil_overall = float("nan")
        silhouette_overall = sil_overall


    if labels is None:
        logger.error("Clustering failed in all modes.")
        raise SystemExit("Clustering failed in all modes.")

    # If 'drop' strategy was used, subset data and ids
    if keep_mask is not None:
        X = X_all[keep_mask, :]
        ids = [cid for cid, keep in zip(ids_all, keep_mask) if bool(keep)]
    else:
        X = X_all
        ids = ids_all

    # Reindex cluster ids to 0..C-1 and build Cluster_#### names
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(sorted(uniq))}
    cl_ids = np.array([remap[int(l)] for l in labels], dtype=int)
    cluster_names = {new: f"C_{new + 1:02d}" for new in sorted(remap.values())}
    moa = [cluster_names[int(c)] for c in cl_ids]

    # Per-sample silhouette (for per-cluster mean)
    try:
        _, sil_samples = cosine_silhouette_scores(X=X, labels=cl_ids)
    except SystemExit:
        sil_samples = np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=float)

    # Centroids for nearest-centroid cosine
    centroids = []
    for k in sorted(cluster_names.keys()):
        idx = np.where(cl_ids == k)[0]
        if idx.size == 0:
            centroids.append(np.zeros(X.shape[1], dtype=float))
            continue
        c = X[idx, :].mean(axis=0)
        nrm = np.linalg.norm(c)
        centroids.append((c / nrm) if nrm > 0 else c)
    C = np.vstack(centroids) if centroids else np.zeros((0, X.shape[1]), dtype=float)
    logger.info("Formed %d clusters.", C.shape[0])
    # Pairwise centroid cosine (avoid self)
    if C.shape[0] >= 2:
        M = C @ C.T
        np.fill_diagonal(M, -np.inf)
        nearest_idx = np.argmax(M, axis=1)
        nearest_cos = M[np.arange(M.shape[0]), nearest_idx]
    else:
        nearest_idx = np.full(shape=(C.shape[0],), fill_value=-1, dtype=int)
        nearest_cos = np.full(shape=(C.shape[0],), fill_value=np.nan, dtype=float)

    # Build cluster stats table
    cluster_rows: List[Dict[str, object]] = []
    for k in sorted(cluster_names.keys()):
        idx = np.where(cl_ids == k)[0]
        n_members = int(idx.size)
        sil = float(np.nan) if (sil_samples is None or n_members == 0) else float(np.nanmean(sil_samples[idx]))
        near_id = int(nearest_idx[k]) if (C.shape[0] >= 2) else -1
        near_cos = float(nearest_cos[k]) if (C.shape[0] >= 2) else float("nan")
        cluster_rows.append({
            "cluster_id": k,
            "moa": cluster_names[k],
            "n_members": n_members,
            "silhouette_cosine": sil,
            "nearest_cluster_id": near_id,
            "nearest_centroid_cosine": near_cos,
        })
    clusters_df = pd.DataFrame(cluster_rows)
    logger.info("Wrote per-cluster stats for %d clusters.", clusters_df.shape[0])

    # Anchors table
    anchors_df = pd.DataFrame({id_col: ids, "moa": moa, "cluster_id": cl_ids})
    logger.info("Wrote anchors table with %d rows.", anchors_df.shape[0])
    anchors_df = anchors_df[[id_col, "moa", "cluster_id"]]
    if args.labels_tsv:
        anchors_df = overlay_given_labels(
            anchors=anchors_df,
            labels_tsv=args.labels_tsv,
            id_col=id_col,
            labels_id_col=args.labels_id_col,
            labels_label_col=args.labels_label_col,
        )
        logger.info("Overlayed user labels; %d compounds labelled.", anchors_df.get("is_labelled", pd.Series(dtype=bool)).sum())
    else:
        logger.info("No labels_tsv provided; skipping label overlay.")


    # Summary row
    summary = pd.DataFrame([{
        "algorithm": chosen_algo,
        "clusterer_requested": args.clusterer,
        "k_if_kmeans": k_if_kmeans,
        "num_clusters": int(len(cluster_names)),
        "n_samples_input": n_input,
        "n_samples_used": int(X.shape[0]),
        "dropped_count": int(dropped_count),
        "noise_handling": noise_handling,
        "silhouette_cosine": float(silhouette_overall),
        "aggregate_method": args.aggregate_method,
        "trimmed_frac": float(args.trimmed_frac),
        "auto_min_clusters": int(args.auto_min_clusters),
        "auto_max_clusters": int(args.auto_max_clusters),
        "n_clusters_requested": int(args.n_clusters),
        "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
        "hdbscan_min_samples": int(args.hdbscan_min_samples),
        "hdbscan_min_silhouette": float(args.hdbscan_min_silhouette),
        "max_singleton_frac": float(args.max_singleton_frac),
        "random_seed": int(args.random_seed),
        "bootstrap_k_main": bool(args.bootstrap_k_main),
        "k_candidates_main": str(args.k_candidates_main),
        "n_bootstrap_main": int(args.n_bootstrap_main),
        "subsample_main": float(args.subsample_main),
        "stability_metric_main": str(args.stability_metric_main),
        "consensus_linkage_main": str(args.consensus_linkage_main),
        "out_k_selection_tsv": out_k_selection_tsv_summary,
        "consensus_pac_limits": str(args.consensus_pac_limits),
    }])
    logger.info("Run summary: %s", summary.iloc[0].to_dict())

    # Write outputs (TSV only)
    Path(args.out_anchors_tsv).parent.mkdir(parents=True, exist_ok=True)
    anchors_df.to_csv(args.out_anchors_tsv, sep="\t", index=False)
    clusters_df.to_csv(args.out_clusters_tsv, sep="\t", index=False)
    summary.to_csv(args.out_summary_tsv, sep="\t", index=False)
    logger.info("Wrote output TSVs to '%s', '%s', and '%s'.",
                args.out_anchors_tsv, args.out_clusters_tsv, args.out_summary_tsv)
    logger.info("Finished pseudo-anchor generation.")


if __name__ == "__main__":
    main()
