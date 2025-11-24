#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap consensus KMeans pseudo-anchor generation.

This script:
1) Loads CLIPn latent space or CellProfiler filtered features (TSV).
2) Detects metadata columns automatically.
3) Aggregates replicate wells per compound (median or mean).
4) L2-normalises the embedding matrix.
5) Performs bootstrap subsampling.
6) Runs KMeans for each bootstrap replicate across candidate k values.
7) Computes silhouette-based stability for each candidate k.
8) Selects the most stable k.
9) Clusters the full dataset with that k.
10) Outputs pseudo-MOA assignments and k-selection diagnostics.

Outputs
-------
out_anchors_tsv:
    [id_col, pseudo_moa]

out_summary_tsv:
    Summary of cluster sizes and embeddings per anchor.

out_clusters_tsv:
    Per-compound cluster assignments with cluster index.

out_k_selection_tsv:
    Stability scores per k.

All outputs are tab-separated.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analysis_pipeline.embedding_utils import (
    configure_logging,
    load_tsv_safely,
    detect_metadata_columns,
    numeric_feature_columns,
    aggregate_replicates,
    l2_normalise,
)


# --------------------------------------------------------------------------- #
# Consensus stability calculation
# --------------------------------------------------------------------------- #

def consensus_stability(
    *,
    labels_list: List[np.ndarray],
    X: np.ndarray,
) -> float:
    """
    Compute consensus stability for a set of bootstrap label vectors.

    Stability is defined as the mean silhouette score across all
    bootstrap solutions, using cosine distance.

    Parameters
    ----------
    labels_list : list of np.ndarray
        List of label arrays from bootstrap replicates.
    X : np.ndarray
        L2-normalised embedding matrix.

    Returns
    -------
    float
        Stability score (higher is better).
    """
    scores = []
    for labels in labels_list:
        try:
            s = silhouette_score(X, labels, metric="cosine")
            scores.append(s)
        except Exception:
            continue

    if len(scores) == 0:
        return -np.inf
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap consensus KMeans clustering for pseudo-anchor generation."
    )

    parser.add_argument(
        "--embeddings_tsv",
        type=str,
        required=True,
        help="Embeddings TSV (CLIPn latent or CellProfiler feature table)."
    )

    parser.add_argument(
        "--out_anchors_tsv",
        type=str,
        required=True,
        help="Output pseudo-anchor assignments TSV."
    )

    parser.add_argument(
        "--out_summary_tsv",
        type=str,
        required=True,
        help="Summary of cluster sizes."
    )

    parser.add_argument(
        "--out_clusters_tsv",
        type=str,
        required=True,
        help="Compound-to-cluster assignments."
    )

    parser.add_argument(
        "--out_k_selection_tsv",
        type=str,
        required=True,
        help="Stability scores for each candidate k."
    )

    parser.add_argument(
        "--id_col",
        type=str,
        default="cpd_id",
        help="Identifier column (default: cpd_id)."
    )

    parser.add_argument(
        "--metadata_cols",
        type=str,
        default="",
        help="Optional extra metadata columns, comma-separated."
    )

    parser.add_argument(
        "--aggregate_method",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Aggregation method for replicate wells."
    )

    parser.add_argument(
        "--k_candidates",
        type=str,
        default="8,12,16,24,32",
        help="Comma-separated list of candidate k values."
    )

    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap replicates."
    )

    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Fraction of points to sample in each bootstrap replicate."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed."
    )

    args = parser.parse_args()
    configure_logging()

    # ------------------------------------------------------------------ #
    # Load input
    # ------------------------------------------------------------------ #
    df = load_tsv_safely(path=args.embeddings_tsv)
    logging.info(f"Loaded table with {df.shape[0]} rows and {df.shape[1]} columns.")

    user_metadata = [x for x in args.metadata_cols.split(",") if x] or None
    metadata_cols = detect_metadata_columns(df=df, user_metadata=user_metadata)

    if args.id_col not in df.columns:
        raise KeyError(f"Identifier column '{args.id_col}' not found.")

    feature_cols = numeric_feature_columns(df=df, metadata_cols=metadata_cols)
    if len(feature_cols) == 0:
        raise ValueError("No numeric embedding features detected.")

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    if df[args.id_col].duplicated().any():
        logging.info("Aggregating replicates to per-compound vectors.")
        emb = aggregate_replicates(
            df=df,
            id_col=args.id_col,
            feature_cols=feature_cols,
            method=args.aggregate_method,
        )
    else:
        emb = df[[args.id_col] + feature_cols].copy()

    ids = emb[args.id_col].astype(str).tolist()
    X = l2_normalise(X=emb[feature_cols].to_numpy().astype(float))
    n = X.shape[0]

    # ------------------------------------------------------------------ #
    # Bootstrap consensus per k
    # ------------------------------------------------------------------ #
    rng = np.random.RandomState(args.random_seed)
    k_list = sorted([int(x) for x in args.k_candidates.split(",")])

    stability_results = []

    for k in k_list:
        if k >= n:
            logging.warning(f"Skipping k={k} because k>=n.")
            continue

        logging.info(f"Evaluating k={k} with {args.n_bootstrap} bootstrap replicates.")
        labels_bootstrap = []

        for b in range(args.n_bootstrap):
            idx = rng.choice(n, size=int(args.subsample * n), replace=False)
            X_sub = X[idx, :]

            try:
                km = KMeans(n_clusters=k, random_state=rng.randint(1_000_000), n_init="auto")
                labels = km.fit_predict(X_sub)
                full_labels = np.full(n, -1)

                # Assign bootstrap labels back to full dataset
                for ii, sample_idx in enumerate(idx):
                    full_labels[sample_idx] = labels[ii]

                # For unassigned points: assign nearest centroid
                centres = km.cluster_centers_
                unassigned = np.where(full_labels == -1)[0]
                if len(unassigned) > 0:
                    sims = X[unassigned] @ centres.T
                    full_labels[unassigned] = np.argmax(sims, axis=1)

                labels_bootstrap.append(full_labels)

            except Exception:
                continue

        score = consensus_stability(labels_list=labels_bootstrap, X=X)
        stability_results.append({"k": k, "stability": score})
        logging.info(f"k={k}: stability={score:.4f}")

    # ------------------------------------------------------------------ #
    # Select best k
    # ------------------------------------------------------------------ #
    if len(stability_results) == 0:
        raise RuntimeError("Could not compute stability for any k.")

    stab_df = pd.DataFrame(stability_results)
    stab_df = stab_df.sort_values("stability", ascending=False)
    best_k = int(stab_df.iloc[0]["k"])

    logging.info(f"Selected k={best_k} as most stable.")

    # ------------------------------------------------------------------ #
    # Final clustering with best k
    # ------------------------------------------------------------------ #
    km = KMeans(n_clusters=best_k, random_state=args.random_seed, n_init="auto")
    labels = km.fit_predict(X)

    uniq = sorted(np.unique(labels))
    mapping = {lab: f"PseudoMOA_{i+1:04d}" for i, lab in enumerate(uniq)}
    pseudo_moa = [mapping[int(lab)] for lab in labels]

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    anchors_df = pd.DataFrame({
        args.id_col: ids,
        "pseudo_moa": pseudo_moa,
    })
    anchors_df.to_csv(args.out_anchors_tsv, sep="\t", index=False)

    clusters_df = pd.DataFrame({
        args.id_col: ids,
        "cluster": labels,
        "pseudo_moa": pseudo_moa,
    })
    clusters_df.to_csv(args.out_clusters_tsv, sep="\t", index=False)

    summary_df = clusters_df["pseudo_moa"].value_counts().reset_index()
    summary_df.columns = ["pseudo_moa", "size"]
    summary_df.to_csv(args.out_summary_tsv, sep="\t", index=False)

    stab_df.to_csv(args.out_k_selection_tsv, sep="\t", index=False)

    logging.info(f"Wrote pseudo-anchors: {args.out_anchors_tsv}")
    logging.info(f"Wrote cluster summary: {args.out_summary_tsv}")
    logging.info(f"Wrote per-compound clusters: {args.out_clusters_tsv}")
    logging.info(f"Wrote k selection table: {args.out_k_selection_tsv}")


if __name__ == "__main__":
    main()
