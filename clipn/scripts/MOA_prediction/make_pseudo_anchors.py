#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create pseudo-anchors (unsupervised clusters) for prototype MOA scoring.
-------------------------------------------------------------------------

Clusters aggregated compound embeddings and writes a 2-column TSV:
[id_col, moa], where moa is 'Cluster_0001', 'Cluster_0002', ...

- Input can be well-level; we aggregate to compound using a robust method.
- Clustering: k-means by default (fast, no extra deps). HDBSCAN optional.

Outputs (TSV):
- anchors_pseudo.tsv : columns [id_col, moa]
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

def detect_id_column(*, df: pd.DataFrame, id_col: str | None) -> str:
    """Detect or validate identifier column."""
    if id_col is not None:
        if id_col in df.columns:
            return id_col
        raise ValueError(f"Identifier column '{id_col}' not found.")
    for c in ["cpd_id", "compound_id", "Compound", "compound", "QueryID", "id"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect an identifier column.")

def l2_normalise(*, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation."""
    n = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    return X / n

def aggregate_compounds(
    *,
    df: pd.DataFrame,
    id_col: str,
    method: str = "median",
    trimmed_frac: float = 0.1,
) -> pd.DataFrame:
    """Aggregate replicate rows per compound into a single embedding."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows: List[dict] = []
    for cid, sub in df.groupby(id_col, sort=False):
        X = sub[num_cols].to_numpy()
        if method == "median":
            vec = np.median(X, axis=0)
        elif method == "mean":
            vec = np.mean(X, axis=0)
        elif method == "trimmed_mean":
            lo = int(np.floor(trimmed_frac * X.shape[0])); hi = int(np.ceil((1 - trimmed_frac) * X.shape[0]))
            Xs = np.sort(X, axis=0); vec = Xs[lo:hi, :].mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregate_method '{method}'")
        rows.append({"__id__": cid, **{c: v for c, v in zip(num_cols, vec)}})
    out = pd.DataFrame(rows).rename(columns={"__id__": id_col})
    return out[[id_col] + num_cols]

def main() -> None:
    """Parse arguments and write pseudo-anchors TSV."""
    parser = argparse.ArgumentParser(description="Make pseudo-anchors by clustering embeddings (TSV in/out).")
    parser.add_argument("--embeddings_tsv", type=str, required=True, help="TSV with embeddings (well- or compound-level).")
    parser.add_argument("--out_anchors_tsv", type=str, required=True, help="Output TSV path for pseudo-anchors.")
    parser.add_argument("--id_col", type=str, default="cpd_id", help="Identifier column name (default: cpd_id).")
    parser.add_argument("--aggregate_method", type=str, default="median", choices=["median", "mean", "trimmed_mean"], help="Replicate aggregation method.")
    parser.add_argument("--n_clusters", type=int, default=30, help="Number of clusters for k-means (default: 30).")
    parser.add_argument("--clusterer", type=str, default="kmeans", choices=["kmeans", "hdbscan"], help="Clustering algorithm (default: kmeans).")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    df = pd.read_csv(args.embeddings_tsv, sep="\t")
    id_col = detect_id_column(df=df, id_col=args.id_col)
    agg = aggregate_compounds(df=df, id_col=id_col, method=args.aggregate_method)
    num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    X = l2_normalise(X=agg[num_cols].to_numpy())
    ids = agg[id_col].astype(str).tolist()

    labels: np.ndarray
    if args.clusterer == "kmeans":
        from sklearn.cluster import KMeans
        k = max(2, min(args.n_clusters, X.shape[0]))
        km = KMeans(n_clusters=k, random_state=args.random_seed, n_init="auto")
        labels = km.fit_predict(X)
    else:
        try:
            import hdbscan  # type: ignore
        except Exception as exc:
            raise SystemExit("hdbscan is not installed; either install it or use --clusterer kmeans") from exc
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, int(0.01 * X.shape[0])), prediction_data=False)
        labels = clusterer.fit_predict(X)
        # Map noise (-1) to its own cluster id
        if (labels == -1).any():
            noise_mask = labels == -1
            if (~noise_mask).any():
                max_lab = labels[~noise_mask].max(initial=-1)
            else:
                max_lab = -1
            labels[noise_mask] = np.arange(max_lab + 1, max_lab + 1 + noise_mask.sum())

    # Build names Cluster_0001, Cluster_0002, ...
    uniq = np.unique(labels)
    mapping = {lab: f"Cluster_{i+1:04d}" for i, lab in enumerate(sorted(uniq))}
    moa = [mapping[int(lab)] for lab in labels]

    out = pd.DataFrame({id_col: ids, "moa": moa})
    out.to_csv(args.out_anchors_tsv, sep="\t", index=False)

if __name__ == "__main__":
    main()
