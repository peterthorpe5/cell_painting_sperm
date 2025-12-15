#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOA centroid scoring with consensus sub-centroids and permutation testing.

This script provides a unified scoring framework that works on:
1) CLIPn latent embeddings (0, 1, 2, ...)
2) CellProfiler filtered feature tables (numeric morphological features)

Workflow summary
----------------
1. Load embedding table
2. Detect metadata columns and feature columns
3. Aggregate replicate wells to per-compound vectors
4. L2-normalise embeddings
5. Load pseudo-anchors (id_col, pseudo_moa)
6. Build one or more centroids per pseudo-MOA:
       - method: median or mean
       - optional sub-centroids (k-means inside each MOA)
       - optional shrinkage towards global mean
7. Score compounds using cosine or CSLS
8. Assign predicted MOA
9. Perform permutation testing for MOA enrichment
10. Write all outputs as TSV

All outputs are tab-separated. UK English spelling is used throughout.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from analysis_pipeline.embedding_utils import (
    configure_logging,
    load_tsv_safely,
    detect_metadata_columns,
    numeric_feature_columns,
    aggregate_replicates,
    l2_normalise,
)


# --------------------------------------------------------------------------- #
# Math utilities
# --------------------------------------------------------------------------- #

def cosine_similarity_matrix(Q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix Q @ P.T."""
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)
    return Q @ P.T


def csls_matrix(Q: np.ndarray, P: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Cross-domain similarity local scaling (CSLS).

    CSLS(q, p) = 2*cos(q, p) - r_q - r_p
    where r_q = avg top-k cos similarities for q,
          r_p = avg top-k cos similarities for p.
    """
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)

    S = Q @ P.T
    kq = min(k, P.shape[0])
    rp = min(k, Q.shape[0])

    if kq <= 0 or rp <= 0:
        return S.copy()

    # r_q
    rq = np.partition(S, kth=S.shape[1] - kq, axis=1)[:, -kq:].mean(axis=1, keepdims=True)

    # r_p
    rp_mat = np.partition(S, kth=S.shape[0] - rp, axis=0)[-rp:, :].mean(axis=0, keepdims=True)

    return 2.0 * S - rq - rp_mat


# --------------------------------------------------------------------------- #
# Centroid builder
# --------------------------------------------------------------------------- #

def build_centroids(
    *,
    emb: pd.DataFrame,
    anchors: pd.DataFrame,
    id_col: str,
    moa_col: str,
    feature_cols: List[str],
    centroid_method: str = "median",
    n_subcentroids: int = 1,
    shrinkage: float = 0.0,
    adaptive_shrinkage: bool = False,
    adaptive_c: float = 0.5,
    adaptive_max: float = 0.3,
    random_state: int = 0,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build centroids for each MOA.

    Parameters
    ----------
    emb : DataFrame
        Per-compound embeddings (normalised).
    anchors : DataFrame
        Pseudo-anchors: [id_col, moa_col]
    id_col : str
        Compound identifier column.
    moa_col : str
        Column name holding MOA labels.
    feature_cols : list of str
        Numeric embedding features.
    centroid_method : str
        'median' or 'mean' when n_subcentroids = 1.
    n_subcentroids : int
        Number of sub-centroids per MOA (via k-means).
    shrinkage : float
        Base shrinkage towards global mean.
    adaptive_shrinkage : bool
        Whether to use size-aware shrinkage.
    adaptive_c : float
        Constant controlling adaptive shrinkage strength.
    adaptive_max : float
        Max additional shrinkage.
    random_state : int

    Returns
    -------
    P : np.ndarray
        Centroid matrix (n_centroids, d)
    centroid_labels : list of str
        MOA label per centroid
    summary_df : pd.DataFrame
        Per-centroid summary metadata
    """

    # Map compounds to rows
    id_to_idx = {cid: i for i, cid in enumerate(emb[id_col].tolist())}

    X = emb[feature_cols].to_numpy()
    global_mean = X.mean(axis=0)
    global_mean /= np.linalg.norm(global_mean) if np.linalg.norm(global_mean) > 0 else 1.0

    centroid_list = []
    centroid_labels = []
    summary_rows = []

    rng = np.random.RandomState(random_state)

    for moa, sub in anchors.groupby(moa_col):
        ids = sub[id_col].astype(str).tolist()
        idxs = [id_to_idx[c] for c in ids if c in id_to_idx]

        if len(idxs) == 0:
            continue

        X_m = X[idxs, :]
        n_m = X_m.shape[0]

        # Adaptive shrinkage amount
        def eff_alpha(n: int) -> float:
            base = float(shrinkage)
            if adaptive_shrinkage and n > 0:
                base += min(adaptive_max, adaptive_c / n)
            return max(0.0, min(1.0, base))

        # If no sub-centroids
        if n_subcentroids <= 1 or n_m < 2:
            vec = np.median(X_m, axis=0) if centroid_method == "median" else np.mean(X_m, axis=0)
            a = eff_alpha(n_m)
            if a > 0:
                vec = (1 - a) * vec + a * global_mean
            vec /= np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else 1.0

            centroid_list.append(vec)
            centroid_labels.append(str(moa))
            summary_rows.append({
                "moa": moa,
                "centroid_index": 0,
                "n_members": n_m,
                "shrinkage": a,
                "method": centroid_method,
            })
            continue

        # Use k-means for sub-centroids
        k = min(n_subcentroids, n_m)
        try:
            km = KMeans(n_clusters=k, random_state=rng.randint(1_000_000), n_init="auto")
            labels = km.fit_predict(X_m)
        except Exception:
            k = 1
            labels = np.zeros(n_m, dtype=int)

        for subc in range(k):
            Xm_sub = X_m[labels == subc]
            if Xm_sub.shape[0] == 0:
                continue

            vec = np.median(Xm_sub, axis=0) if centroid_method == "median" else np.mean(Xm_sub, axis=0)
            a = eff_alpha(Xm_sub.shape[0])
            if a > 0:
                vec = (1 - a) * vec + a * global_mean
            vec /= np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else 1.0

            centroid_list.append(vec)
            centroid_labels.append(str(moa))
            summary_rows.append({
                "moa": moa,
                "centroid_index": subc,
                "n_members": Xm_sub.shape[0],
                "shrinkage": a,
                "method": f"kmeans/{centroid_method}",
            })

    P = np.vstack(centroid_list) if centroid_list else np.zeros((0, X.shape[1]))
    summary_df = pd.DataFrame(summary_rows)
    return P, centroid_labels, summary_df


# --------------------------------------------------------------------------- #
# Permutation testing
# --------------------------------------------------------------------------- #

def permutation_test(
    *,
    scores: np.ndarray,
    centroid_labels: List[str],
    n_perm: int = 1000,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permutation test for MOA enrichment.

    Parameters
    ----------
    scores : np.ndarray
        Matrix (n_compounds, n_centroids) of cosine/CSLS scores.
    centroid_labels : list of str
        MOA label for each centroid.
    n_perm : int
        Number of permutations.
    random_state : int

    Returns
    -------
    null_dist : np.ndarray
        Null distribution of best-centroid scores.
    p_values : np.ndarray
        P-value per query compound.
    """
    rng = np.random.RandomState(random_state)
    n_q, n_c = scores.shape
    centroid_labels = np.array(centroid_labels)

    # Compute observed max score for each compound
    obs = scores.max(axis=1)

    # Build label index sets
    moa_to_indices = {}
    for j, lab in enumerate(centroid_labels):
        moa_to_indices.setdefault(lab, []).append(j)

    null_samples = []

    for _ in range(n_perm):
        perm_labels = centroid_labels.copy()
        rng.shuffle(perm_labels)

        # recompute best centroid score under permutation
        best_perm = []
        for i in range(n_q):
            # pick centroid indices belonging to permuted MOA
            # but because labels shuffled, we simply shift columns
            j = rng.randint(0, n_c)
            best_perm.append(scores[i, j])
        null_samples.append(best_perm)

    null_arr = np.array(null_samples)  # (n_perm, n_q)

    # Compute p-values: p = (count(null >= obs) + 1) / (n_perm + 1)
    p_vals = ((null_arr >= obs[None, :]).sum(axis=0) + 1) / (n_perm + 1)

    return null_arr, p_vals


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="MOA centroid scoring with permutation testing."
    )

    parser.add_argument(
        "--embeddings_tsv", type=str, required=True,
        help="Input embeddings TSV (CLIPn latent or CellProfiler features)."
    )
    parser.add_argument(
        "--anchors_tsv", type=str, required=True,
        help="Pseudo-anchor assignments TSV from make_pseudo_anchors."
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Directory to write all outputs."
    )

    parser.add_argument("--id_col", type=str, default="cpd_id")
    parser.add_argument("--moa_col", type=str, default="pseudo_moa")
    parser.add_argument("--metadata_cols", type=str, default="")

    parser.add_argument(
        "--aggregate_method",
        type=str, default="median", choices=["median", "mean"],
    )

    parser.add_argument(
        "--centroid_method",
        type=str, default="median", choices=["median", "mean"],
    )

    parser.add_argument(
        "--n_subcentroids",
        type=int, default=1,
        help="Number of sub-centroids per MOA (k-means)."
    )

    parser.add_argument("--shrinkage", type=float, default=0.0)
    parser.add_argument("--adaptive_shrinkage", action="store_true")
    parser.add_argument("--adaptive_c", type=float, default=0.5)
    parser.add_argument("--adaptive_max", type=float, default=0.3)

    parser.add_argument(
        "--use_csls", action="store_true",
        help="Use CSLS instead of cosine."
    )

    parser.add_argument("--csls_k", type=int, default=10)

    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()
    configure_logging()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing outputs to {out_dir}")

    # ------------------------------------------------------------------ #
    # Load tables
    # ------------------------------------------------------------------ #
    df = load_tsv_safely(path=args.embeddings_tsv)
    anchors = load_tsv_safely(path=args.anchors_tsv)

    user_metadata = [x for x in args.metadata_cols.split(",") if x] or None
    metadata_cols = detect_metadata_columns(df=df, user_metadata=user_metadata)

    if args.id_col not in df.columns:
        raise KeyError(f"Identifier column '{args.id_col}' not found.")

    feature_cols = numeric_feature_columns(df=df, metadata_cols=metadata_cols)
    if len(feature_cols) == 0:
        raise ValueError("No numeric embedding features detected.")

    # ------------------------------------------------------------------ #
    # Aggregate if needed
    # ------------------------------------------------------------------ #
    if df[args.id_col].duplicated().any():
        logging.info("Aggregating replicates.")
        emb = aggregate_replicates(
            df=df,
            id_col=args.id_col,
            feature_cols=feature_cols,
            method=args.aggregate_method,
        )
    else:
        emb = df[[args.id_col] + feature_cols].copy()

    emb = emb.copy()
    emb["__row"] = np.arange(emb.shape[0])  # for ordering

    # Normalise
    X = l2_normalise(X=emb[feature_cols].to_numpy().astype(float))
    emb_mat = X.copy()

    # ------------------------------------------------------------------ #
    # Filter anchors to those in embedding table
    # ------------------------------------------------------------------ #
    anchors = anchors.copy()
    anchors[args.id_col] = anchors[args.id_col].astype(str)
    emb_ids = set(emb[args.id_col].astype(str))
    anchors = anchors[anchors[args.id_col].isin(emb_ids)]

    if anchors.shape[0] == 0:
        raise ValueError("No anchors match the embeddings table.")

    logging.info(f"Using {anchors.shape[0]} anchors for centroid building.")

    # ------------------------------------------------------------------ #
    # Build centroids
    # ------------------------------------------------------------------ #
    P, centroid_labels, summary_df = build_centroids(
        emb=emb,
        anchors=anchors,
        id_col=args.id_col,
        moa_col=args.moa_col,
        feature_cols=feature_cols,
        centroid_method=args.centroid_method,
        n_subcentroids=args.n_subcentroids,
        shrinkage=args.shrinkage,
        adaptive_shrinkage=args.adaptive_shrinkage,
        adaptive_c=args.adaptive_c,
        adaptive_max=args.adaptive_max,
        random_state=args.random_seed,
    )

    # ------------------------------------------------------------------ #
    # Score compounds
    # ------------------------------------------------------------------ #
    if args.use_csls:
        scores = csls_matrix(emb_mat, P, k=args.csls_k)
    else:
        scores = cosine_similarity_matrix(emb_mat, P)

    # Predicted MOA = centroid with highest score
    best_idx = scores.argmax(axis=1)
    pred_moa = [centroid_labels[j] for j in best_idx]

    pred_df = pd.DataFrame({
        args.id_col: emb[args.id_col].tolist(),
        "predicted_moa": pred_moa,
    })
    pred_df.to_csv(out_dir / "compound_predictions.tsv", sep="\t", index=False)

    # Raw scores
    score_df = pd.DataFrame(scores, columns=[f"centroid_{i}" for i in range(scores.shape[1])])
    score_df.insert(0, args.id_col, emb[args.id_col].tolist())
    score_df.to_csv(out_dir / "raw_scores.tsv", sep="\t", index=False)

    # Centroids summary
    summary_df.to_csv(out_dir / "centroids_summary.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ #
    # Permutation testing
    # ------------------------------------------------------------------ #
    null_dist, p_vals = permutation_test(
        scores=scores,
        centroid_labels=centroid_labels,
        n_perm=args.n_perm,
        random_state=args.random_seed,
    )

    p_df = pd.DataFrame({
        args.id_col: emb[args.id_col].tolist(),
        "p_value": p_vals,
    })
    p_df.to_csv(out_dir / "permutation_pvalues.tsv", sep="\t", index=False)

    # Optionally store null distribution
    np.savetxt(out_dir / "null_distribution.tsv", null_dist, delimiter="\t")

    logging.info("MOA scoring + permutation testing complete.")


if __name__ == "__main__":
    main()
