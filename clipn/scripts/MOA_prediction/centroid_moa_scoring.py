#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centroid-based MOA Scoring for CLIPn or Feature-Space Embeddings
----------------------------------------------------------------

This script infers mode-of-action (MOA) by matching compound embeddings to
MOA *centroids* (class representatives). It is robust to replicate wells and
does not rely on single nearest neighbours.

Workflow
--------
1) Load an embeddings TSV containing at least one identifier column (e.g., 'cpd_id')
   and many numeric columns (the embedding dimensions).
2) Aggregate replicate rows per compound using a robust estimator
   (median / trimmed-mean / geometric-median).
3) Load an anchors TSV mapping a subset of compounds to MOA labels.
4) Build one or more centroids per MOA (median/mean or k-means subclusters),
   optionally with a minimum-members gate and size-aware shrinkage.
5) Score *all* compounds against centroids with cosine (and optionally CSLS).
6) Aggregate centroid scores per MOA (max or mean), compute top prediction,
   margins, and optional permutation-based FDR (Benjamini–Hochberg across compounds).

Outputs (TSV; never comma-separated)
------------------------------------
- <out_dir>/compound_embeddings.tsv
    One row per compound after aggregation (id + numeric dims).
- <out_dir>/centroids_summary.tsv
    One row per centroid: MOA, centroid_index, n_members, method, shrinkage_effective.
- <out_dir>/compound_moa_scores.tsv
    Long-form scores: compound × MOA (cosine; CSLS if requested). Includes anchor flags.
- <out_dir>/compound_predictions.tsv
    One row per compound: top MOA, chosen score, margin; diagnostics
    (cosine/CSLS tops); optional p-value and q-value (BH) from permutations.
    Includes anchor flags to indicate potential self-inflation.

Notes
-----
- “Neighbourhood” and “normalise” follow UK English.
- CSLS is computed by default; the decision rule defaults to 'auto'.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------- maths / array utilities -------------------------- #

def l2_normalise(*, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalise rows of a matrix.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n_samples, n_features).
    eps : float, optional
        Numerical stabiliser added to norms, by default 1e-12.

    Returns
    -------
    np.ndarray
        Row-normalised matrix.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


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
        Convergence tolerance on movement, by default 1e-6.

    Returns
    -------
    np.ndarray
        1D array representing the geometric median.
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


# ------------------------------- I/O helpers -------------------------------- #

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
    candidates = ["cpd_id", "compound_id", "Compound", "compound", "QueryID", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Could not detect an identifier column (tried common candidates).")


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
    import re
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




# ------------------------ aggregation & centroid build ----------------------- #

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
    groups = df.groupby(id_col, sort=False)
    rows = []
    for cid, sub in groups:
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
        rows.append((cid, vec))
    out = pd.DataFrame([{"__id__": cid, **{f: v for f, v in zip(num_cols, vec)}} for cid, vec in rows])
    out = out.rename(columns={"__id__": id_col})
    return out[[id_col] + num_cols]


def build_moa_centroids(
    *,
    embeddings: pd.DataFrame,
    anchors: pd.DataFrame,
    id_col: str,
    moa_col: str,
    n_centroids_per_moa: int = 1,
    centroid_method: str = "median",
    centroid_shrinkage: float = 0.0,
    min_members_per_moa: int = 1,
    skip_tiny_moas: bool = False,
    adaptive_shrinkage: bool = False,
    adaptive_shrinkage_c: float = 0.5,
    adaptive_shrinkage_max: float = 0.3,
    random_seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build one or more centroids per MOA, with optional minimum-members gate
    and size-aware (adaptive) shrinkage for small-n centroids.

    Parameters
    ----------
    embeddings : pd.DataFrame
        Aggregated compound embeddings (one row per id).
    anchors : pd.DataFrame
        Table with columns [id_col, moa_col] giving labelled anchor compounds.
    id_col : str
        Identifier column name.
    moa_col : str
        MOA label column name.
    n_centroids_per_moa : int, optional
        Number of sub-centroids per MOA (k-means within each MOA if >1),
        by default 1.
    centroid_method : str, optional
        'median' or 'mean' when n_centroids_per_moa == 1, by default 'median'.
    centroid_shrinkage : float, optional
        Baseline shrinkage towards the global mean (0..1), by default 0.0.
    min_members_per_moa : int, optional
        Minimum labelled members to form a centroid, by default 1.
    skip_tiny_moas : bool, optional
        If True, MOAs with < min_members_per_moa are skipped; otherwise they
        are kept, but may be stabilised via adaptive shrinkage if enabled.
    adaptive_shrinkage : bool, optional
        If True, add a size-aware term alpha_add = min(adaptive_shrinkage_max,
        adaptive_shrinkage_c / n_members). Effective alpha is clamped to [0, 1].
    adaptive_shrinkage_c : float, optional
        C constant for the size-aware term (default 0.5).
    adaptive_shrinkage_max : float, optional
        Maximum size-aware addition (default 0.3).
    random_seed : int, optional
        Random seed for subclustering, by default 0.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, List[str]]
        (centroids_summary_df, P_matrix, centroid_moas)
        - centroids_summary_df: columns [moa, centroid_index, n_members,
          method, shrinkage_effective]
        - P_matrix: centroid matrix (n_centroids x d), L2-normalised
        - centroid_moas: per-centroid MOA labels (length n_centroids)
    """
    rng = np.random.RandomState(random_seed)
    id_idx = {cid: i for i, cid in enumerate(embeddings[id_col].tolist())}
    # num_cols = embeddings.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = numeric_feature_columns(embeddings)

    X_all = l2_normalise(X=embeddings[num_cols].to_numpy())

    # Global mean direction for optional shrinkage
    gmean = X_all.mean(axis=0)
    gmean = gmean / np.linalg.norm(gmean) if np.linalg.norm(gmean) > 0 else gmean

    # Collect labelled members per MOA
    labelled = anchors[[id_col, moa_col]].dropna().copy()
    labelled = labelled[labelled[id_col].isin(id_idx.keys())]
    moa_groups = labelled.groupby(moa_col)

    def effective_alpha(n_members: int) -> float:
        """Combine baseline and optional size-aware shrinkage; clamp to [0, 1]."""
        alpha = float(centroid_shrinkage)
        if adaptive_shrinkage and n_members > 0:
            alpha += min(float(adaptive_shrinkage_max), float(adaptive_shrinkage_c) / float(n_members))
        return float(min(1.0, max(0.0, alpha)))

    P_list: List[np.ndarray] = []
    centroid_moas: List[str] = []
    summary_rows: List[Dict[str, object]] = []

    for moa, sub in moa_groups:
        idxs = [id_idx[c] for c in sub[id_col].tolist()]
        X_m = X_all[idxs, :]
        n_members_moa = int(X_m.shape[0])

        # Tiny MOA gate
        if n_members_moa < int(min_members_per_moa) and bool(skip_tiny_moas):
            summary_rows.append(
                {"moa": moa, "centroid_index": -1, "n_members": n_members_moa,
                 "method": "skipped_tiny", "shrinkage_effective": 0.0}
            )
            continue

        if n_centroids_per_moa <= 1 or X_m.shape[0] <= 2:
            if centroid_method == "median":
                proto = np.median(X_m, axis=0)
            elif centroid_method == "mean":
                proto = np.mean(X_m, axis=0)
            else:
                raise ValueError(f"Unknown centroid_method '{centroid_method}'")

            alpha = effective_alpha(n_members=n_members_moa)
            if alpha > 0:
                proto = (1 - alpha) * proto + alpha * gmean
            proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto

            P_list.append(proto)
            centroid_moas.append(str(moa))
            summary_rows.append(
                {"moa": moa, "centroid_index": 0, "n_members": n_members_moa,
                 "method": f"{centroid_method}", "shrinkage_effective": float(alpha)}
            )
        else:
            # k-means within this MOA to create sub-centroids
            try:
                from sklearn.cluster import KMeans
                n_k = min(n_centroids_per_moa, X_m.shape[0])
                km = KMeans(n_clusters=n_k, random_state=rng.randint(0, 10**6), n_init="auto")
                labels = km.fit_predict(X_m)
                for j in range(n_k):
                    sel = X_m[labels == j, :]
                    n_sub = int(sel.shape[0])
                    if n_sub == 0:
                        continue
                    if n_sub < int(min_members_per_moa) and bool(skip_tiny_moas):
                        summary_rows.append(
                            {"moa": moa, "centroid_index": j, "n_members": n_sub,
                             "method": "skipped_tiny_subcluster", "shrinkage_effective": 0.0}
                        )
                        continue

                    proto = np.median(sel, axis=0) if centroid_method == "median" else np.mean(sel, axis=0)
                    alpha = effective_alpha(n_members=n_sub)
                    if alpha > 0:
                        proto = (1 - alpha) * proto + alpha * gmean
                    proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto

                    P_list.append(proto)
                    centroid_moas.append(str(moa))
                    summary_rows.append(
                        {"moa": moa, "centroid_index": j, "n_members": n_sub,
                         "method": f"kmeans/{centroid_method}", "shrinkage_effective": float(alpha)}
                    )
            except Exception:
                # Fallback: single centroid if k-means unavailable
                proto = np.median(X_m, axis=0) if centroid_method == "median" else np.mean(X_m, axis=0)
                alpha = effective_alpha(n_members=n_members_moa)
                if alpha > 0:
                    proto = (1 - alpha) * proto + alpha * gmean
                proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto

                P_list.append(proto)
                centroid_moas.append(str(moa))
                summary_rows.append(
                    {"moa": moa, "centroid_index": 0, "n_members": n_members_moa,
                     "method": f"{centroid_method}(fallback_no_kmeans)", "shrinkage_effective": float(alpha)}
                )

    P = np.vstack(P_list) if P_list else np.zeros((0, X_all.shape[1]), dtype=float)
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, P, centroid_moas


# -------------------------- scoring: cosine / CSLS --------------------------- #

def cosine_scores(*, Q: np.ndarray, P: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """
    Compute cosine similarity matrix between query and centroid matrices.

    Parameters
    ----------
    Q : np.ndarray
        Query matrix (n_queries x d), rows must be L2-normalised.
    P : np.ndarray
        Centroid matrix (n_centroids x d), rows must be L2-normalised.
    batch_size : int, optional
        Batch size for matrix multiplication, by default 4096.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_queries, n_centroids).
    """
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)
    sims = np.empty((Q.shape[0], P.shape[0]), dtype=float)
    for start in range(0, Q.shape[0], batch_size):
        end = min(start + batch_size, Q.shape[0])
        sims[start:end, :] = Q[start:end, :] @ P.T
    return sims


def csls_scores(*, Q: np.ndarray, P: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute CSLS scores between query and centroid matrices.

    CSLS(q, p) = 2 * cos(q, p) - r_q - r_p
    where r_q is the average cosine of q to its top-k centroids,
          r_p is the average cosine of p to its top-k queries.

    Parameters
    ----------
    Q : np.ndarray
        Query matrix (n_queries x d), L2-normalised.
    P : np.ndarray
        Centroid matrix (n_centroids x d), L2-normalised.
    k : int, optional
        Neighbourhood size for local scaling, by default 10.

    Returns
    -------
    np.ndarray
        CSLS score matrix (n_queries x n_centroids).
    """
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)
    S = Q @ P.T  # cosine
    kq = min(k, P.shape[0])
    rp = min(k, Q.shape[0])
    if kq <= 0 or rp <= 0:
        return S.copy()
    # top-k per row (queries)
    part_q = np.partition(S, kth=S.shape[1] - kq, axis=1)[:, -kq:]
    r_q = part_q.mean(axis=1, keepdims=True)
    # top-k per column (centroids)
    part_p = np.partition(S, kth=S.shape[0] - rp, axis=0)[-rp:, :]
    r_p = part_p.mean(axis=0, keepdims=True)
    return 2.0 * S - r_q - r_p


# -------------------------- aggregation / decisions -------------------------- #

def build_moa_indexers(*, centroid_moas: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Build MOA list and per-MOA centroid index mapping.

    Parameters
    ----------
    centroid_moas : List[str]
        Per-centroid MOA labels.

    Returns
    -------
    Tuple[List[str], Dict[str, List[int]]]
        (moa_list, moa_to_centroid_indices)
    """
    moa_list = sorted(set(centroid_moas))
    moa_to_idx = {m: [i for i, mm in enumerate(centroid_moas) if mm == m] for m in moa_list}
    return moa_list, moa_to_idx


def agg_over_centroids(
    *,
    mat: Optional[np.ndarray],
    moa_list: List[str],
    moa_to_idx: Dict[str, List[int]],
    mode: str = "max",
) -> Optional[np.ndarray]:
    """
    Aggregate centroid scores to MOA scores.

    Parameters
    ----------
    mat : Optional[np.ndarray]
        Compound×centroid score matrix, or None.
    moa_list : List[str]
        Ordered list of MOA names.
    moa_to_idx : Dict[str, List[int]]
        Mapping MOA -> list of centroid column indices in 'mat'.
    mode : str, optional
        'max' or 'mean', by default 'max'.

    Returns
    -------
    Optional[np.ndarray]
        Compound×MOA score matrix after aggregation, or None if mat is None.
    """
    if mat is None:
        return None
    out = np.zeros((mat.shape[0], len(moa_list)), dtype=float)
    for j, m in enumerate(moa_list):
        cols = moa_to_idx[m]
        sub = mat[:, cols]
        out[:, j] = sub.max(axis=1) if mode == "max" else sub.mean(axis=1)
    return out


def choose_primary_matrix(
    *,
    M_cos: np.ndarray,
    M_csls: Optional[np.ndarray],
    rule: str = "auto",
    margin_threshold: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose the primary MOA-score matrix for decision-making.

    Parameters
    ----------
    M_cos : np.ndarray
        Compound×MOA matrix using cosine aggregation.
    M_csls : Optional[np.ndarray]
        Compound×MOA matrix using CSLS aggregation, or None if not computed.
    rule : str, optional
        One of {"cosine", "csls", "auto"}. With "auto", use cosine unless the
        cosine margin is below 'margin_threshold' for a given compound.
    margin_threshold : float, optional
        Margin switch threshold for "auto".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (M_primary, decisions) where M_primary has the chosen scores and
        'decisions' is a vector of strings ("cosine" or "csls") per compound.
    """
    n = M_cos.shape[0]
    decisions = np.full(n, "cosine", dtype=object)

    if rule == "cosine" or M_csls is None:
        return M_cos, decisions

    if rule == "csls":
        decisions[:] = "csls"
        return M_csls, decisions

    # auto: per-row switch if cosine margin is small
    M_primary = M_cos.copy()
    best_idx = np.argmax(M_cos, axis=1)
    best_val = M_cos[np.arange(n), best_idx]
    tmp = M_cos.copy()
    tmp[np.arange(n), best_idx] = -np.inf
    runner = np.max(tmp, axis=1)
    margins = best_val - runner
    switch = margins < float(margin_threshold)

    M_primary[switch, :] = M_csls[switch, :]
    decisions[switch] = "csls"
    return M_primary, decisions


# ------------------------------ permutation FDR ------------------------------ #

def benjamini_hochberg_q(*, pvals: np.ndarray) -> np.ndarray:
    """
    Compute Benjamini–Hochberg q-values for a vector of p-values.

    Parameters
    ----------
    pvals : np.ndarray
        Array of p-values.

    Returns
    -------
    np.ndarray
        Array of q-values (same shape).
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = p_sorted * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def estimate_fdr_by_permutation(
    *,
    S_primary: np.ndarray,
    proto_to_moa: Sequence[int],
    n_moas: int,
    agg_mode: str = "max",
    n_permutations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-compound FDR for the top-MOA 'margin' by shuffling the
    centroid→MOA assignment (preserves the number of centroids per MOA).

    Parameters
    ----------
    S_primary : np.ndarray
        Compound×centroid score matrix consistent with the decision rule
        (row-wise chosen cosine/CSLS). Shape (n_compounds, n_centroids).
    proto_to_moa : Sequence[int]
        Original centroid→MOA index mapping (length n_centroids).
    n_moas : int
        Number of MOAs (columns after aggregation).
    agg_mode : str
        'max' or 'mean' to aggregate centroids→MOA.
    n_permutations : int
        Number of permutations.
    rng : Optional[np.random.Generator]
        Random generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        p-values and BH q-values per compound.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_compounds, n_centroids = S_primary.shape
    proto_to_moa = np.asarray(proto_to_moa, dtype=int)
    counts = np.bincount(proto_to_moa, minlength=n_moas)

    # Observed MOA matrix and margins
    cols_obs = []
    for m in range(n_moas):
        idx = np.where(proto_to_moa == m)[0]
        if idx.size == 0:
            cols_obs.append(np.full((n_compounds, 1), -np.inf, dtype=float))
            continue
        block = S_primary[:, idx]
        if agg_mode == "mean":
            cols_obs.append(block.mean(axis=1, keepdims=True))
        else:
            cols_obs.append(block.max(axis=1, keepdims=True))
    M_obs = np.concatenate(cols_obs, axis=1)

    best_idx = np.argmax(M_obs, axis=1)
    best_val = M_obs[np.arange(n_compounds), best_idx]
    tmp = M_obs.copy()
    tmp[np.arange(n_compounds), best_idx] = -np.inf
    runner = np.max(tmp, axis=1)
    observed_margin = best_val - runner

    # Permutation null
    perm_margins = np.empty((n_compounds, n_permutations), dtype=float)
    base = np.arange(n_centroids)
    for b in range(n_permutations):
        rng.shuffle(base)
        cols = []
        start = 0
        for c in counts:
            if c > 0:
                idx = base[start:start + c]
                block = S_primary[:, idx]
                if agg_mode == "mean":
                    cols.append(block.mean(axis=1, keepdims=True))
                else:
                    cols.append(block.max(axis=1, keepdims=True))
                start += c
            else:
                cols.append(np.full((n_compounds, 1), -np.inf, dtype=float))
        M_perm = np.concatenate(cols, axis=1)

        bidx = np.argmax(M_perm, axis=1)
        bval = M_perm[np.arange(n_compounds), bidx]
        ttmp = M_perm.copy()
        ttmp[np.arange(n_compounds), bidx] = -np.inf
        rval = np.max(ttmp, axis=1)
        perm_margins[:, b] = bval - rval

    pvals = (1.0 + np.sum(perm_margins >= observed_margin[:, None], axis=1)) / (n_permutations + 1.0)
    qvals = benjamini_hochberg_q(pvals=pvals.astype(float))
    return pvals, qvals



# --------------------------------- main ------------------------------------- #

def main() -> None:
    """
    Parse arguments and run centroid MOA scoring with cosine/CSLS and optional FDR.

    Notes
    -----
    - Outputs are TSV (tab-separated) to comply with “never comma-separated”.
    - CSLS is computed by default; the decision rule defaults to 'auto'.
    """
    parser = argparse.ArgumentParser(description="Centroid-based MOA scoring (TSV I/O, UK English).")
    parser.add_argument("--embeddings_tsv", type=str, required=True, help="TSV with embeddings (well- or compound-level).")
    parser.add_argument("--anchors_tsv", type=str, required=True, help="TSV with anchors: id + MOA.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write outputs (TSV).")

    parser.add_argument("--id_col", type=str, default="cpd_id", help="Identifier column name (default: cpd_id).")
    parser.add_argument("--moa_col", type=str, default="moa_final", help="MOA label column in anchors (default: moa_final).")

    parser.add_argument("--aggregate_method", type=str, default="median",
                        choices=["median", "mean", "trimmed_mean", "geometric_median"],
                        help="Replicate aggregation method (default: median).")
    parser.add_argument("--trimmed_frac", type=float, default=0.1,
                        help="Trim fraction for trimmed_mean (default: 0.1).")

    parser.add_argument("--n_centroids_per_moa", type=int, default=1,
                        help="Sub-centroids per MOA (k-means if >1).")
    parser.add_argument("--centroid_method", type=str, default="median",
                        choices=["median", "mean"], help="Centroid estimator when n_centroids_per_moa<=1 (default: median).")
    parser.add_argument("--centroid_shrinkage", type=float, default=0.0,
                        help="Shrink centroids towards global mean (0..1).")
    parser.add_argument("--min_members_per_moa", type=int, default=1,
                        help="Minimum labelled members needed to form a centroid (default: 1 keeps all).")
    parser.add_argument("--skip_tiny_moas", action="store_true",
                        help="If set, MOAs with < min_members_per_moa are skipped (no centroid built).")
    parser.add_argument("--adaptive_shrinkage", action="store_true",
                        help=("If set, adds size-aware shrinkage for small centroids: "
                              "alpha_eff = centroid_shrinkage + min(adaptive_shrinkage_max, "
                              "adaptive_shrinkage_c / n_members)."))
    parser.add_argument("--adaptive_shrinkage_c", type=float, default=0.5,
                        help="C constant for size-aware shrinkage term (default: 0.5).")
    parser.add_argument("--adaptive_shrinkage_max", type=float, default=0.3,
                        help="Maximum extra shrinkage due to size-aware term (default: 0.3).")

    parser.add_argument("--moa_score_agg", type=str, default="max",
                        choices=["max", "mean"], help="Aggregate centroid→MOA score (default: max).")

    parser.add_argument("--use_csls", action="store_true",
                        help="Also compute CSLS scores (in addition to cosine).")
    parser.set_defaults(use_csls=True)  # CSLS ON by default

    parser.add_argument("--csls_k", type=int, default=-1,
                        help="Neighbourhood size for CSLS. Use -1 to auto-select k≈sqrt(#centroids), clipped to [5, 50].")

    parser.add_argument("--primary_score", type=str, default="auto",
                        choices=["cosine", "csls", "auto"],
                        help="Which score decides the top MOA. 'auto' uses cosine unless the cosine margin < threshold.")
    parser.add_argument("--auto_margin_threshold", type=float, default=0.02,
                        help="When --primary_score auto: if cosine margin < this, switch to CSLS for the decision.")

    parser.add_argument("--exclude_anchors_from_queries", action="store_true",
                        help="Do not produce predictions for anchor compounds.")
    parser.add_argument("--annotate_anchors", action="store_true",
                        help="Annotate outputs with is_anchor/anchor_moa and potential_inflation flags.")
    parser.set_defaults(annotate_anchors=True)

    parser.add_argument("--n_permutations", type=int, default=200,
                        help="Permutations for FDR (0 to disable). Default: 200.")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="Random seed for reproducibility (default: 0).")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare embeddings
    df = pd.read_csv(args.embeddings_tsv, sep="\t", low_memory=False)
    id_col = detect_id_column(df=df, id_col=args.id_col)

    agg = aggregate_compounds(
        df=df,
        id_col=id_col,
        method=args.aggregate_method,
        trimmed_frac=args.trimmed_frac,
    )
    # num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = numeric_feature_columns(agg)



    X = l2_normalise(X=agg[num_cols].to_numpy())
    ids = agg[id_col].astype(str).tolist()

    print(f"[dbg] using {len(num_cols)} numeric features; first 5: {num_cols[:5]}")


    # Save aggregated compound embeddings
    agg.to_csv(out_dir / "compound_embeddings.tsv", sep="\t", index=False)

    # Load anchors
    anchors = pd.read_csv(args.anchors_tsv, sep="\t", low_memory=False)
    if id_col not in anchors.columns:
        raise ValueError(f"Anchors file must contain id column '{id_col}'.")
    if args.moa_col not in anchors.columns:
        raise ValueError(f"Anchors file must contain MOA column '{args.moa_col}'.")

    # Anchor lookup for annotation
    anchor_map: Dict[str, str] = dict(
        anchors[[id_col, args.moa_col]]
        .dropna()
        .astype({id_col: str, args.moa_col: str})
        .itertuples(index=False, name=None)
    )

    # Build centroids
    centroids_df, P, centroid_moas = build_moa_centroids(
        embeddings=agg,
        anchors=anchors,
        id_col=id_col,
        moa_col=args.moa_col,
        n_centroids_per_moa=args.n_centroids_per_moa,
        centroid_method=args.centroid_method,
        centroid_shrinkage=args.centroid_shrinkage,
        min_members_per_moa=args.min_members_per_moa,
        skip_tiny_moas=args.skip_tiny_moas,
        adaptive_shrinkage=args.adaptive_shrinkage,
        adaptive_shrinkage_c=args.adaptive_shrinkage_c,
        adaptive_shrinkage_max=args.adaptive_shrinkage_max,
        random_seed=args.random_seed,
    )
    centroids_df.to_csv(out_dir / "centroids_summary.tsv", sep="\t", index=False)

    # If no centroids, write empty schemas and exit
    if P.shape[0] == 0:
        scores_cols = [id_col, "moa", "cosine", "csls", "is_anchor", "anchor_moa"]
        pd.DataFrame(columns=scores_cols).to_csv(out_dir / "compound_moa_scores.tsv", sep="\t", index=False)
        pred_cols = [
            id_col, "decision_rule", "top_moa", "top_score", "margin",
            "top_moa_cosine", "top_cosine", "top_moa_csls", "top_csls",
            "p_value", "q_value", "is_anchor", "anchor_moa", "anchor_same_as_top", "potential_inflation"
        ]
        pd.DataFrame(columns=pred_cols).to_csv(out_dir / "compound_predictions.tsv", sep="\t", index=False)
        return

    # Map centroids→MOA and define aggregation
    moa_list, moa_to_idx = build_moa_indexers(centroid_moas=centroid_moas)

    # Compute similarities
    S_cos = cosine_scores(Q=X, P=P, batch_size=4096)
    print("[dbg] S_cos shape:", S_cos.shape,
      "min:", float(S_cos.min()),
      "median:", float(np.median(S_cos)),
      "max:", float(S_cos.max()))
    print("[dbg] ||Q|| mean±sd:", float(np.linalg.norm(X, axis=1).mean()),
        float(np.linalg.norm(X, axis=1).std()))
    print("[dbg] ||P|| mean±sd:", float(np.linalg.norm(P, axis=1).mean()),
        float(np.linalg.norm(P, axis=1).std()))

    # Defensive clamp; should be almost never needed
    if (S_cos > 1.0 + 1e-6).any() or (S_cos < -1.0 - 1e-6).any():
        print("[warn] cosine out of bounds; clamping to [-1, 1]")
        S_cos = np.clip(S_cos, -1.0, 1.0)


    S_csls = None
    if args.use_csls:
        if args.csls_k is None or int(args.csls_k) <= 0:
            k_eff = int(np.sqrt(P.shape[0]))
            k_eff = int(np.clip(k_eff, 5, 50))
        else:
            k_eff = min(int(args.csls_k), max(1, P.shape[0] - 1))
        S_csls = csls_scores(Q=X, P=P, k=k_eff)

    # Aggregate to MOA and choose primary matrix
    M_cos = agg_over_centroids(mat=S_cos, moa_list=moa_list, moa_to_idx=moa_to_idx, mode=args.moa_score_agg)
    M_csls = agg_over_centroids(mat=S_csls, moa_list=moa_list, moa_to_idx=moa_to_idx, mode=args.moa_score_agg) if S_csls is not None else None
    M_primary, decision_rule_vec = choose_primary_matrix(
        M_cos=M_cos,
        M_csls=M_csls,
        rule=args.primary_score,
        margin_threshold=args.auto_margin_threshold,
    )

    # --- Build centroid-level matrix consistent with the decision rule ---
    # S_primary[i, :] = S_cos or S_csls row-for-row depending on decision_rule_vec
    if args.primary_score == "cosine" or S_csls is None:
        S_primary = S_cos
    elif args.primary_score == "csls":
        S_primary = S_csls
    else:
        S_primary = S_cos.copy()
        if S_csls is not None:
            mask = (decision_rule_vec == "csls")
            S_primary[mask, :] = S_csls[mask, :]


    # Long-form scores per compound×MOA (with anchor annotations)
    long_rows: List[Dict[str, object]] = []
    for i, cid in enumerate(ids):
        is_anchor = cid in anchor_map
        anchor_moa = anchor_map.get(cid, "")
        for j, moa in enumerate(moa_list):
            row = {
                id_col: cid,
                "moa": moa,
                "cosine": float(M_cos[i, j]),
                "is_anchor": int(is_anchor) if args.annotate_anchors else 0,
                "anchor_moa": anchor_moa if args.annotate_anchors else "",
            }
            if M_csls is not None:
                row["csls"] = float(M_csls[i, j])
            else:
                row["csls"] = np.nan
            long_rows.append(row)
    scores_df = pd.DataFrame(long_rows)

    # Predictions (primary rule + diagnostics) with anchor annotations
    pred_rows: List[Dict[str, object]] = []
    exclude_set = set(anchors[id_col].astype(str)) if args.exclude_anchors_from_queries else set()

    for i, cid in enumerate(ids):
        if cid in exclude_set:
            continue

        # Primary decision
        best_idx = int(np.argmax(M_primary[i, :]))
        best_moa = moa_list[best_idx]
        best = float(M_primary[i, best_idx])
        temp = M_primary[i, :].copy()
        temp[best_idx] = -np.inf
        runner = float(np.max(temp))
        margin = best - runner if np.isfinite(runner) else best

        row = {
            id_col: cid,
            "decision_rule": str(decision_rule_vec[i]),
            "top_moa": best_moa,
            "top_score": best,
            "margin": margin,
        }

        # Diagnostics: cosine
        c_idx = int(np.argmax(M_cos[i, :]))
        row["top_moa_cosine"] = moa_list[c_idx]
        row["top_cosine"] = float(M_cos[i, c_idx])

        # Diagnostics: CSLS (present or NaN)
        if M_csls is not None:
            s_idx = int(np.argmax(M_csls[i, :]))
            row["top_moa_csls"] = moa_list[s_idx]
            row["top_csls"] = float(M_csls[i, s_idx])
        else:
            row["top_moa_csls"] = ""
            row["top_csls"] = np.nan

        # Anchor annotations
        if args.annotate_anchors:
            is_anchor = cid in anchor_map
            anchor_moa = anchor_map.get(cid, "")
            potential_inflation = bool(is_anchor and (anchor_moa == best_moa))
            row["is_anchor"] = int(is_anchor)
            row["anchor_moa"] = anchor_moa
            row["anchor_same_as_top"] = (anchor_moa == best_moa) if is_anchor else ""
            row["potential_inflation"] = int(potential_inflation)

        pred_rows.append(row)

    preds_df = pd.DataFrame(pred_rows)

    # Optional permutation FDR (mirror the aggregator used for decisions)
    if args.n_permutations > 0:
        # Build centroid→MOA index vector consistent with P / centroid_moas
        moa_index_map = {m: i for i, m in enumerate(moa_list)}
        proto_to_moa_idx = np.array([moa_index_map[m] for m in centroid_moas], dtype=int)

        pvals, qvals = estimate_fdr_by_permutation(
            S_primary=S_primary,
            proto_to_moa=proto_to_moa_idx,
            n_moas=len(moa_list),
            agg_mode=args.moa_score_agg,
            n_permutations=args.n_permutations,
            rng=np.random.default_rng(args.random_seed),
        )



        preds_df["p_value"] = pvals
        preds_df["q_value"] = qvals
    else:
        preds_df["p_value"] = np.nan
        preds_df["q_value"] = np.nan

    # Write outputs (TSV only)
    scores_df.to_csv(out_dir / "compound_moa_scores.tsv", sep="\t", index=False)
    preds_df.to_csv(out_dir / "compound_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
