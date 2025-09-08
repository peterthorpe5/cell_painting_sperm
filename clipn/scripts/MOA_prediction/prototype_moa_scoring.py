#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prototype (Centroid) MOA Scoring for CLIPn or Feature-Space Embeddings
-----------------------------------------------------------------------

This script infers mode-of-action (MOA) by matching compound embeddings to
MOA *prototypes* (centroids). It is robust to replicate wells and does not rely
on single nearest neighbours.

Workflow
--------
1) Load an embeddings TSV containing at least one identifier column (e.g., 'cpd_id')
   and many numeric columns (the embedding dimensions).
2) Aggregate replicate rows per compound using a robust estimator
   (median / trimmed-mean / geometric-median).
3) Load an anchors TSV mapping a subset of compounds to MOA labels.
4) Build one or more prototypes per MOA (median/mean or k-means subclusters).
5) Score *all* compounds against prototypes with cosine (and optionally CSLS).
6) Aggregate prototype scores per MOA (max or mean), compute top prediction,
   margins, and optional permutation-based FDR.

Inputs (TSV)
------------
- embeddings_tsv:
    Columns:
      - 'cpd_id' (default; configurable via --id_col) or similar identifier.
      - optional replicate metadata (e.g., 'well_id', 'plate', ...).
      - numeric columns for embedding dimensions (auto-detected).
- anchors_tsv:
    Columns:
      - id column (same as embeddings, e.g., 'cpd_id')
      - 'moa' column (configurable via --moa_col)

Outputs (TSV; never comma-separated)
------------------------------------
- <out_dir>/compound_embeddings.tsv
    One row per compound after aggregation (id + numeric dims).
- <out_dir>/prototypes_summary.tsv
    One row per prototype: MOA, prototype_index, n_members, method, params.
- <out_dir>/compound_moa_scores.tsv
    Long-form scores: compound × MOA (cosine, optional CSLS, and margins).
- <out_dir>/compound_predictions.tsv
    One row per compound: top MOA, scores, margin, (optional) p-value, q-value.

Usage examples
--------------
# Minimal: cosine only, single prototype per MOA, robust median aggregation
python prototype_moa_scoring.py \
  --embeddings_tsv CLIPn_latent.tsv \
  --anchors_tsv anchors.tsv \
  --out_dir moa_scores \
  --id_col cpd_id \
  --aggregate_method median

# Add CSLS and allow 2 prototypes per MOA (k-means on MOA members)
python prototype_moa_scoring.py \
  --embeddings_tsv CLIPn_latent.tsv \
  --anchors_tsv anchors.tsv \
  --out_dir moa_scores_csls \
  --id_col cpd_id \
  --aggregate_method median \
  --n_prototypes_per_moa 2 \
  --use_csls \
  --csls_k 10

# Include permutation FDR with 200 permutations (can be slow on large sets)
python prototype_moa_scoring.py \
  --embeddings_tsv CLIPn_latent.tsv \
  --anchors_tsv anchors.tsv \
  --out_dir moa_scores_fdr \
  --id_col cpd_id \
  --aggregate_method median \
  --use_csls --csls_k 10 \
  --n_permutations 200 \
  --random_seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------- helpers: maths & io --------------------------- #

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
        1D array of length n_features representing the geometric median.
    """
    if X.shape[0] == 1:
        return X[0].copy()
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        # avoid division by zero: if y equals a data point, return it
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


def numeric_matrix_from_df(*, df: pd.DataFrame, exclude_cols: Sequence[str]) -> np.ndarray:
    """
    Extract numeric embedding matrix, excluding given columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    exclude_cols : Sequence[str]
        Column names to exclude from the numeric selection.

    Returns
    -------
    np.ndarray
        2D numeric matrix.
    """
    return df.drop(columns=list(exclude_cols), errors="ignore").select_dtypes(include=[np.number]).to_numpy()


# ------------------------ aggregation & prototypes ------------------------ #

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


def build_moa_prototypes(
    *,
    embeddings: pd.DataFrame,
    anchors: pd.DataFrame,
    id_col: str,
    moa_col: str,
    n_prototypes_per_moa: int = 1,
    prototype_method: str = "median",
    prototype_shrinkage: float = 0.0,
    random_seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Build one or more prototypes per MOA.

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
    n_prototypes_per_moa : int, optional
        Number of sub-prototypes per MOA (k-means within each MOA if >1),
        by default 1.
    prototype_method : str, optional
        'median' or 'mean' when n_prototypes_per_moa == 1, by default 'median'.
    prototype_shrinkage : float, optional
        Shrinkage towards the global mean (0..1), by default 0.0.
    random_seed : int, optional
        Random seed for subclustering, by default 0.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, np.ndarray]]
        (prototypes_summary_df, P_matrix, moa_names, moa_to_member_matrix)
        - prototypes_summary_df: columns [moa, proto_index, n_members, method]
        - P_matrix: prototype matrix (n_prototypes x d), L2-normalised
        - moa_names: list the same length as number of prototypes (per-row MOA)
        - moa_to_member_matrix: mapping from MOA to stacked member vectors (for permutation nulls)
    """
    rng = np.random.RandomState(random_seed)
    id_idx = {cid: i for i, cid in enumerate(embeddings[id_col].tolist())}
    num_cols = embeddings.select_dtypes(include=[np.number]).columns.tolist()
    X_all = embeddings[num_cols].to_numpy()
    X_all = l2_normalise(X=X_all)

    # Global mean direction for optional shrinkage
    gmean = X_all.mean(axis=0)
    gmean = gmean / np.linalg.norm(gmean) if np.linalg.norm(gmean) > 0 else gmean

    # Collect labelled members per MOA
    labelled = anchors[[id_col, moa_col]].dropna().copy()
    labelled = labelled[labelled[id_col].isin(id_idx.keys())]
    moa_groups = labelled.groupby(moa_col)
    P_list = []
    moa_names: List[str] = []
    summary_rows = []
    moa_to_members: Dict[str, np.ndarray] = {}

    for moa, sub in moa_groups:
        idxs = [id_idx[c] for c in sub[id_col].tolist()]
        X_m = X_all[idxs, :]
        moa_to_members[moa] = X_m.copy()

        if n_prototypes_per_moa <= 1 or X_m.shape[0] <= 2:
            if prototype_method == "median":
                proto = np.median(X_m, axis=0)
            elif prototype_method == "mean":
                proto = np.mean(X_m, axis=0)
            else:
                raise ValueError(f"Unknown prototype_method '{prototype_method}'")
            if prototype_shrinkage > 0:
                proto = (1 - prototype_shrinkage) * proto + prototype_shrinkage * gmean
            proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto
            P_list.append(proto)
            moa_names.append(str(moa))
            summary_rows.append(
                {"moa": moa, "proto_index": 0, "n_members": int(X_m.shape[0]), "method": f"{prototype_method}(shrink={prototype_shrinkage})"}
            )
        else:
            # k-means within this MOA to create sub-prototypes
            try:
                from sklearn.cluster import KMeans
                n_k = min(n_prototypes_per_moa, X_m.shape[0])
                km = KMeans(n_clusters=n_k, random_state=rng.randint(0, 10**6), n_init="auto")
                labels = km.fit_predict(X_m)
                for j in range(n_k):
                    sel = X_m[labels == j, :]
                    if sel.shape[0] == 0:
                        continue
                    proto = np.median(sel, axis=0) if prototype_method == "median" else np.mean(sel, axis=0)
                    if prototype_shrinkage > 0:
                        proto = (1 - prototype_shrinkage) * proto + prototype_shrinkage * gmean
                    proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto
                    P_list.append(proto)
                    moa_names.append(str(moa))
                    summary_rows.append(
                        {"moa": moa, "proto_index": j, "n_members": int(sel.shape[0]), "method": f"kmeans/{prototype_method}(shrink={prototype_shrinkage})"}
                    )
            except Exception as exc:
                # Fallback: single prototype if k-means unavailable
                proto = np.median(X_m, axis=0) if prototype_method == "median" else np.mean(X_m, axis=0)
                proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto
                P_list.append(proto)
                moa_names.append(str(moa))
                summary_rows.append(
                    {"moa": moa, "proto_index": 0, "n_members": int(X_m.shape[0]), "method": f"{prototype_method}(fallback_no_kmeans)"}
                )

    P = np.vstack(P_list) if P_list else np.zeros((0, X_all.shape[1]), dtype=float)
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, P, moa_names, moa_to_members


# -------------------------- scoring: cosine / CSLS ------------------------ #

def cosine_scores(
    *,
    Q: np.ndarray,
    P: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between query and prototype matrices.

    Parameters
    ----------
    Q : np.ndarray
        Query matrix (n_queries x d), rows must be L2-normalised.
    P : np.ndarray
        Prototype matrix (n_prototypes x d), rows must be L2-normalised.
    batch_size : int, optional
        Batch size for matrix multiplication, by default 4096.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_queries, n_prototypes).
    """
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)
    sims = np.empty((Q.shape[0], P.shape[0]), dtype=float)
    for start in range(0, Q.shape[0], batch_size):
        end = min(start + batch_size, Q.shape[0])
        sims[start:end, :] = Q[start:end, :] @ P.T
    return sims


def csls_scores(
    *,
    Q: np.ndarray,
    P: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Compute CSLS scores between query and prototype matrices.

    CSLS(q, p) = 2 * cos(q, p) - r_q - r_p
    where r_q is the average cosine of q to its top-k prototypes,
          r_p is the average cosine of p to its top-k queries.

    Parameters
    ----------
    Q : np.ndarray
        Query matrix (n_queries x d), L2-normalised.
    P : np.ndarray
        Prototype matrix (n_prototypes x d), L2-normalised.
    k : int, optional
        Neighbourhood size for local scaling, by default 10.

    Returns
    -------
    np.ndarray
        CSLS score matrix (n_queries x n_prototypes).
    """
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)
    S = Q @ P.T  # cosine
    # r_q: mean of top-k in each row
    kq = min(k, P.shape[0])
    rp = min(k, Q.shape[0])
    if kq <= 0 or rp <= 0:
        return S.copy()
    # top-k per row (queries)
    part_q = np.partition(S, kth=S.shape[1] - kq, axis=1)[:, -kq:]
    r_q = part_q.mean(axis=1, keepdims=True)
    # top-k per column (prototypes)
    part_p = np.partition(S, kth=S.shape[0] - rp, axis=0)[-rp:, :]
    r_p = part_p.mean(axis=0, keepdims=True)
    return 2.0 * S - r_q - r_p


# ---------------------------- permutation FDR ----------------------------- #

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
    n = pvals.size
    order = np.argsort(pvals, kind="mergesort")
    ranked = pvals[order]
    q = np.minimum.accumulate((ranked * n) / (np.arange(n) + 1)[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def sample_null_prototypes(
    *,
    moa_to_members: Dict[str, np.ndarray],
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, List[str]]:
    """
    Create one random prototype per MOA by resampling members within each MOA.

    Parameters
    ----------
    moa_to_members : Dict[str, np.ndarray]
        Mapping MOA -> member matrix (rows L2-normalised).
    rng : np.random.RandomState
        Random generator.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (P_null, moa_names) where P_null has one prototype per MOA.
    """
    protos = []
    names = []
    for moa, Xm in moa_to_members.items():
        if Xm.shape[0] == 0:
            continue
        # bootstrap sample with replacement, same size
        idx = rng.randint(low=0, high=Xm.shape[0], size=Xm.shape[0])
        proto = np.median(Xm[idx, :], axis=0)
        norm = np.linalg.norm(proto)
        proto = proto / norm if norm > 0 else proto
        protos.append(proto)
        names.append(moa)
    if not protos:
        return np.zeros((0, next(iter(moa_to_members.values())).shape[1])), names
    return np.vstack(protos), names


# ------------------------------- main logic ------------------------------- #

def main() -> None:
    """Parse arguments and run prototype MOA scoring."""
    parser = argparse.ArgumentParser(description="Prototype (centroid) MOA scoring (TSV I/O, UK English).")
    parser.add_argument("--embeddings_tsv", type=str, required=True, help="TSV with embeddings (well- or compound-level).")
    parser.add_argument("--anchors_tsv", type=str, required=True, help="TSV with anchors: id + MOA.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write outputs (TSV).")
    parser.add_argument("--id_col", type=str, default="cpd_id", help="Identifier column name (default: cpd_id).")
    parser.add_argument("--moa_col", type=str, default="moa", help="MOA label column in anchors (default: moa).")
    parser.add_argument("--aggregate_method", type=str, default="median", choices=["median", "mean", "trimmed_mean", "geometric_median"], help="Replicate aggregation method (default: median).")
    parser.add_argument("--trimmed_frac", type=float, default=0.1, help="Trim fraction for trimmed_mean (default: 0.1).")
    parser.add_argument("--n_prototypes_per_moa", type=int, default=1, help="Sub-prototypes per MOA (k-means if >1).")
    parser.add_argument("--prototype_method", type=str, default="median", choices=["median", "mean"], help="Prototype estimator when n_prototypes_per_moa<=1 (default: median).")
    parser.add_argument("--prototype_shrinkage", type=float, default=0.0, help="Shrink prototypes towards global mean (0..1).")
    parser.add_argument("--use_csls", action="store_true", help="Also compute CSLS scores (in addition to cosine).")
    parser.add_argument("--csls_k", type=int, default=10, help="k for CSLS local scaling (default: 10).")
    parser.add_argument("--moa_score_agg", type=str, default="max", choices=["max", "mean"], help="Aggregate prototype→MOA score (default: max).")
    parser.add_argument("--exclude_anchors_from_queries", action="store_true", help="Do not produce predictions for anchor compounds.")
    parser.add_argument("--n_permutations", type=int, default=0, help="Permutations for FDR (0 to disable).")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility (default: 0).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare embeddings
    df = pd.read_csv(args.embeddings_tsv, sep="\t")
    id_col = detect_id_column(df=df, id_col=args.id_col)

    # Aggregate replicates if needed
    agg = aggregate_compounds(
        df=df,
        id_col=id_col,
        method=args.aggregate_method,
        trimmed_frac=args.trimmed_frac,
    )
    # L2-normalise embeddings
    num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    X = l2_normalise(X=agg[num_cols].to_numpy())
    ids = agg[id_col].astype(str).tolist()

    # Save aggregated compound embeddings
    agg_out = agg.copy()
    agg_out.to_csv(out_dir / "compound_embeddings.tsv", sep="\t", index=False)

    # Load anchors
    anchors = pd.read_csv(args.anchors_tsv, sep="\t")
    if id_col not in anchors.columns:
        raise ValueError(f"Anchors file must contain id column '{id_col}'.")
    if args.moa_col not in anchors.columns:
        raise ValueError(f"Anchors file must contain MOA column '{args.moa_col}'.")

    # Build prototypes
    protos_df, P, proto_moas, moa_to_members = build_moa_prototypes(
        embeddings=agg_out,
        anchors=anchors,
        id_col=id_col,
        moa_col=args.moa_col,
        n_prototypes_per_moa=args.n_prototypes_per_moa,
        prototype_method=args.prototype_method,
        prototype_shrinkage=args.prototype_shrinkage,
        random_seed=args.random_seed,
    )
    protos_df.to_csv(out_dir / "prototypes_summary.tsv", sep="\t", index=False)

    # If no prototypes, bail gracefully
    if P.shape[0] == 0:
        # Create empty outputs
        pd.DataFrame(columns=[id_col, "moa", "cosine", "csls", "top_moa", "top_cosine", "top_moa_margin"]).to_csv(
            out_dir / "compound_moa_scores.tsv", sep="\t", index=False
        )
        pd.DataFrame(columns=[id_col, "top_moa", "top_cosine", "margin", "p_value", "q_value"]).to_csv(
            out_dir / "compound_predictions.tsv", sep="\t", index=False
        )
        return

    # Compute cosine and optional CSLS
    S_cos = cosine_scores(Q=X, P=P, batch_size=4096)
    S_csls = csls_scores(Q=X, P=P, k=args.csls_k) if args.use_csls else None

    # Aggregate per MOA (max or mean across prototypes of the same MOA)
    moa_list = sorted(set(proto_moas))
    moa_to_idx = {m: [i for i, mm in enumerate(proto_moas) if mm == m] for m in moa_list}

    def agg_over_protos(mat: np.ndarray) -> np.ndarray:
        out = np.zeros((mat.shape[0], len(moa_list)), dtype=float)
        for j, m in enumerate(moa_list):
            cols = moa_to_idx[m]
            sub = mat[:, cols]
            out[:, j] = sub.max(axis=1) if args.moa_score_agg == "max" else sub.mean(axis=1)
        return out

    M_cos = agg_over_protos(S_cos)
    M_csls = agg_over_protos(S_csls) if S_csls is not None else None

    # Build long-form scores per compound×MOA
    long_rows: List[Dict[str, object]] = []
    for i, cid in enumerate(ids):
        for j, moa in enumerate(moa_list):
            row = {
                id_col: cid,
                "moa": moa,
                "cosine": float(M_cos[i, j]),
            }
            if M_csls is not None:
                row["csls"] = float(M_csls[i, j])
            long_rows.append(row)
    scores_df = pd.DataFrame(long_rows)

    # Compute predictions (choose cosine as primary; report CSLS too if present)
    pred_rows: List[Dict[str, object]] = []
    exclude_set = set(anchors[id_col].astype(str)) if args.exclude_anchors_from_queries else set()
    for i, cid in enumerate(ids):
        if cid in exclude_set:
            continue
        # primary decision on cosine
        best_idx = int(np.argmax(M_cos[i, :]))
        best_moa = moa_list[best_idx]
        best = float(M_cos[i, best_idx])
        # runner-up (set that index to -inf temporarily)
        tmp = M_cos[i, :].copy()
        tmp[best_idx] = -np.inf
        runner = float(np.max(tmp))
        margin = best - runner if np.isfinite(runner) else best
        row = {
            id_col: cid,
            "top_moa": best_moa,
            "top_cosine": best,
            "margin": margin,
        }
        if M_csls is not None:
            best_idx_c = int(np.argmax(M_csls[i, :]))
            row["top_moa_csls"] = moa_list[best_idx_c]
            row["top_csls"] = float(M_csls[i, best_idx_c])
        pred_rows.append(row)
    preds_df = pd.DataFrame(pred_rows)

    # Optional permutation FDR
    if args.n_permutations > 0:
        rng = np.random.RandomState(args.random_seed)
        # observed top cosine per compound
        obs = preds_df.set_index(id_col)["top_cosine"].reindex(ids).to_numpy()
        null_max = np.zeros((args.n_permutations, len(ids)), dtype=float)

        for b in range(args.n_permutations):
            P_null, names_null = sample_null_prototypes(moa_to_members=moa_to_members, rng=rng)
            if P_null.shape[0] == 0:
                continue
            S_null = cosine_scores(Q=X, P=P_null, batch_size=4096)
            # aggregate per MOA (max)
            # names order may differ; just take row-wise maxima as null for "best MOA"
            null_max[b, :] = S_null.max(axis=1)

        # p-values: Pr(null >= observed)
        ge = (null_max >= obs[None, :]).sum(axis=0)
        pvals = (ge + 1.0) / (args.n_permutations + 1.0)
        qvals = benjamini_hochberg_q(pvals=pvals.astype(float))
        preds_df["p_value"] = pvals
        preds_df["q_value"] = qvals

    # Write outputs
    scores_df.to_csv(out_dir / "compound_moa_scores.tsv", sep="\t", index=False)
    preds_df.to_csv(out_dir / "compound_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
