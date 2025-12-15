#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate pseudo-anchors by clustering compound embeddings.

This script works for:
1) CLIPn latent embeddings (integer-named feature columns: "0", "1", ...)
2) CellProfiler processed feature tables (numerical features, metadata excluded)

Workflow
--------
1. Load embeddings TSV
2. Detect metadata columns automatically
3. Detect numeric feature columns automatically
4. Aggregate replicate wells to per-compound vectors
5. L2-normalise the embedding matrix
6. Cluster with KMeans (optionally auto-k)
7. Write pseudo-MOA assignments as a TSV

Outputs
-------
out_anchors_tsv:
    A TSV containing: [id_col, pseudo_moa]

All outputs are tab-separated (never comma-separated).
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
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
# Helper: auto-k selection
# --------------------------------------------------------------------------- #

def choose_k_auto(
    *,
    X: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 0,
) -> int:
    """
    Choose k automatically by maximising silhouette score on a subsample.

    Parameters
    ----------
    X : np.ndarray
        L2-normalised feature matrix.
    k_min : int
        Minimum number of clusters.
    k_max : int
        Maximum number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    int
        Selected cluster number.
    """
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n < 3:
        return 1

    # Define candidate ks
    ks = sorted(set([
        max(2, k_min),
        int(np.sqrt(n)),
        int(1.5 * np.sqrt(n)),
        int(2.0 * np.sqrt(n)),
        min(k_max, n - 1),
    ]))

    best_k = ks[0]
    best_score = -np.inf

    for k in ks:
        try:
            km = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init="auto"
            )
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create pseudo-MOA anchors by clustering compound embeddings."
    )

    parser.add_argument(
        "--embeddings_tsv",
        type=str,
        required=True,
        help="Input embeddings TSV (CLIPn latent or CellProfiler features)."
    )

    parser.add_argument(
        "--out_anchors_tsv",
        type=str,
        required=True,
        help="Output pseudo-anchors TSV."
    )

    parser.add_argument(
        "--id_col",
        type=str,
        default="cpd_id",
        help="Identifier column (default: cpd_id)."
    )

    parser.add_argument(
        "--aggregate_method",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Replicate aggregation method."
    )

    parser.add_argument(
        "--clusterer",
        type=str,
        default="kmeans",
        choices=["kmeans"],
        help="Clustering algorithm (default: kmeans)."
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=30,
        help="Number of clusters for KMeans, unless auto-k is used."
    )

    parser.add_argument(
        "--auto_k",
        action="store_true",
        help="Enable auto-k selection via silhouette score."
    )

    parser.add_argument(
        "--k_min",
        type=int,
        default=8,
        help="Minimum k when --auto_k is used."
    )

    parser.add_argument(
        "--k_max",
        type=int,
        default=64,
        help="Maximum k when --auto_k is used."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed."
    )

    parser.add_argument(
        "--metadata_cols",
        type=str,
        default="",
        help="Optional extra metadata columns, comma-separated. "
             "Ignored safely if missing."
    )

    args = parser.parse_args()
    configure_logging()

    # ------------------------------------------------------------------ #
    # Load table
    # ------------------------------------------------------------------ #
    df = load_tsv_safely(path=args.embeddings_tsv)
    logging.info(f"Loaded embeddings table with {df.shape[0]} rows and {df.shape[1]} columns.")

    # ------------------------------------------------------------------ #
    # Metadata + features
    # ------------------------------------------------------------------ #
    user_metadata = [x for x in args.metadata_cols.split(",") if x] or None
    metadata_cols = detect_metadata_columns(df=df, user_metadata=user_metadata)

    if args.id_col not in df.columns:
        raise KeyError(
            f"Identifier column '{args.id_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = numeric_feature_columns(df=df, metadata_cols=metadata_cols)

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns detected after excluding metadata.")

    logging.info(f"Detected {len(feature_cols)} feature columns.")
    logging.info(f"Detected metadata columns: {metadata_cols}")

    # ------------------------------------------------------------------ #
    # Aggregate to per-compound
    # ------------------------------------------------------------------ #
    if df[args.id_col].duplicated().any():
        logging.info("Aggregating replicates per compound.")
        emb = aggregate_replicates(
            df=df,
            id_col=args.id_col,
            feature_cols=feature_cols,
            method=args.aggregate_method,
        )
    else:
        logging.info("No replicate aggregation needed.")
        emb = df[[args.id_col] + feature_cols].copy()

    # ------------------------------------------------------------------ #
    # Prepare embedding matrix
    # ------------------------------------------------------------------ #
    X = emb[feature_cols].to_numpy().astype(float)
    X = l2_normalise(X=X)

    ids = emb[args.id_col].astype(str).tolist()
    n = X.shape[0]

    # ------------------------------------------------------------------ #
    # Choose k
    # ------------------------------------------------------------------ #
    if args.auto_k:
        k = choose_k_auto(
            X=X,
            k_min=args.k_min,
            k_max=args.k_max,
            random_state=args.random_seed,
        )
        logging.info(f"Auto-selected k = {k}")
    else:
        k = max(2, min(args.n_clusters, n))
        logging.info(f"Using fixed k = {k}")

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #
    if args.clusterer == "kmeans":
        km = KMeans(
            n_clusters=k,
            random_state=args.random_seed,
            n_init="auto"
        )
        labels = km.fit_predict(X)
    else:
        raise ValueError("Unsupported clusterer.")

    # ------------------------------------------------------------------ #
    # Build pseudo-MOA label names
    # ------------------------------------------------------------------ #
    uniq = sorted(np.unique(labels))
    mapping = {lab: f"PseudoMOA_{i+1:04d}" for i, lab in enumerate(uniq)}
    pseudo_moa = [mapping[int(lab)] for lab in labels]

    out = pd.DataFrame({
        args.id_col: ids,
        "pseudo_moa": pseudo_moa,
    })

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    out_path = Path(args.out_anchors_tsv)
    out.to_csv(out_path, sep="\t", index=False)
    logging.info(f"Wrote pseudo-anchors to: {out_path}")


if __name__ == "__main__":
    main()
