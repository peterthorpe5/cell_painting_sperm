#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster embeddings/features and produce hierarchical heatmaps and pairwise distances.

This script is intended for small-to-moderate numbers of compounds (e.g. ~10–5000),
where the full pairwise distance matrix fits in memory.

Workflow
--------
1) Load embeddings/features TSV (may be per-cell/per-image/per-well)
2) Detect numeric feature columns and aggregate to one vector per compound
3) Optionally scale features (robust z-score or standard z-score)
4) Compute pairwise distances:
      - cosine distance
      - Spearman distance (1 - Spearman correlation)
      - optional Euclidean distance (on scaled features)
5) Hierarchical clustering with scipy linkage
6) Save:
      - distance matrices (TSV)
      - nearest neighbours (TSV)
      - dendrogram leaf order (TSV)
      - clustered heatmap PDFs

All file I/O is tab-separated. UK English spelling is used throughout.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform, pdist


LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #

def configure_logging(*, level: int = logging.INFO) -> None:
    """
    Configure basic logging.

    Parameters
    ----------
    level : int
        Logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# --------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------- #

def load_tsv(*, path: str | Path) -> pd.DataFrame:
    """
    Load a tab-separated file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to TSV.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_tsv(*, df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame as TSV.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : str or Path
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------- #
# Feature detection and aggregation
# --------------------------------------------------------------------- #

_DEFAULT_METADATA = {
    "cpd_id",
    "cpd_type",
    "Plate_Metadata",
    "Well_Metadata",
    "Library",
    "Sample",
    "Dataset",
    "plate",
    "well",
    "row",
    "col",
}


def detect_metadata_columns(
    *,
    df: pd.DataFrame,
    user_metadata: Optional[List[str]] = None,
) -> List[str]:
    """
    Detect metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    user_metadata : list of str, optional
        Extra metadata columns to treat as metadata.

    Returns
    -------
    list of str
        Metadata columns present in df.
    """
    meta = set(_DEFAULT_METADATA)
    if user_metadata:
        meta.update(set(user_metadata))
    return [c for c in df.columns if c in meta]


def numeric_feature_columns(*, df: pd.DataFrame, metadata_cols: List[str]) -> List[str]:
    """
    Get numeric feature columns excluding metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    metadata_cols : list of str
        Columns to exclude.

    Returns
    -------
    list of str
        Feature columns.
    """
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if c not in set(metadata_cols)]


def aggregate_to_compounds(
    *,
    df: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    method: str,
) -> pd.DataFrame:
    """
    Aggregate rows to one vector per compound.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    id_col : str
        Compound identifier column.
    feature_cols : list of str
        Numeric feature columns.
    method : str
        "median" or "mean".

    Returns
    -------
    pd.DataFrame
        Aggregated table with [id_col] + feature_cols.
    """
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'.")

    rows = []
    for cid, sub in df.groupby(id_col, sort=False):
        x = sub[feature_cols].to_numpy(dtype=float)
        if method == "median":
            vec = np.nanmedian(x, axis=0)
        else:
            vec = np.nanmean(x, axis=0)

        row = {id_col: str(cid)}
        row.update({c: v for c, v in zip(feature_cols, vec)})
        rows.append(row)

    out = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------- #

def robust_scale(*, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Robustly scale features using median and IQR.

    scaled = (x - median) / IQR

    Parameters
    ----------
    x : np.ndarray
        Matrix (n, d).
    eps : float
        Numerical stabiliser.

    Returns
    -------
    np.ndarray
        Scaled matrix.
    """
    med = np.nanmedian(x, axis=0, keepdims=True)
    q1 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    q3 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    iqr = np.maximum(q3 - q1, eps)
    return (x - med) / iqr


def standard_scale(*, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Standardise features using mean and standard deviation.

    Parameters
    ----------
    x : np.ndarray
        Matrix (n, d).
    eps : float
        Numerical stabiliser.

    Returns
    -------
    np.ndarray
        Scaled matrix.
    """
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (x - mu) / sd


# --------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------- #

def cosine_distance_matrix(*, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute cosine distance matrix (1 - cosine similarity).

    Parameters
    ----------
    x : np.ndarray
        Matrix (n, d).
    eps : float
        Numerical stabiliser.

    Returns
    -------
    np.ndarray
        Distance matrix (n, n).
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    xn = x / norms
    sim = xn @ xn.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def spearman_distance_matrix(*, x: np.ndarray) -> np.ndarray:
    """
    Compute Spearman distance matrix (1 - Spearman correlation) robustly.

    This implementation is designed to be stable when some vectors have
    zero variance (or become constant after scaling), which would otherwise
    yield NaN correlations.

    Parameters
    ----------
    x : np.ndarray
        Matrix (n, d).

    Returns
    -------
    np.ndarray
        Distance matrix (n, n) with finite values.
    """
    # Rank each row across features (Spearman = Pearson on ranks)
    ranks = pd.DataFrame(x).rank(axis=1, method="average").to_numpy(dtype=float)

    # Correlation across rows (each row is a compound vector)
    corr = pd.DataFrame(ranks).T.corr(method="pearson").to_numpy(dtype=float)

    dist = 1.0 - corr

    # Replace non-finite distances: if corr is NaN, treat as unrelated (distance=1)
    dist = np.where(np.isfinite(dist), dist, 1.0)

    # Force symmetry and clean diagonal
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    # Clip to [0, 2] to avoid tiny numerical negatives
    dist = np.clip(dist, 0.0, 2.0)

    return dist



def euclidean_distance_matrix(*, x: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance matrix.

    Parameters
    ----------
    x : np.ndarray
        Matrix (n, d).

    Returns
    -------
    np.ndarray
        Distance matrix (n, n).
    """
    d = pdist(x, metric="euclidean")
    return squareform(d)


# --------------------------------------------------------------------- #
# Clustering and outputs
# --------------------------------------------------------------------- #

def hierarchical_order(
    *,
    dist: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering and return linkage and leaf order.

    Parameters
    ----------
    dist : np.ndarray
        Square distance matrix.
    method : str
        Linkage method (average, complete, single, ward).

    Returns
    -------
    Z : np.ndarray
        Linkage matrix.
    order : np.ndarray
        Leaf order indices.
    """
    if dist.shape[0] < 2:
        raise ValueError("Need at least 2 points for clustering.")
    condensed = squareform(dist, checks=False)

    if method == "ward":
        # Ward is defined for Euclidean distances; callers should ensure that.
        Z = linkage(condensed, method="ward")
    else:
        Z = linkage(condensed, method=method)

    order = leaves_list(Z)
    return Z, order


def nearest_neighbours(
    *,
    ids: List[str],
    dist: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    """
    Compute top-N nearest neighbours for each item.

    Parameters
    ----------
    ids : list of str
        Identifiers in the same order as dist.
    dist : np.ndarray
        Distance matrix (n, n).
    top_n : int
        Number of neighbours to report per item.

    Returns
    -------
    pd.DataFrame
        Long table with nearest neighbours.
    """
    n = dist.shape[0]
    rows = []
    for i in range(n):
        d = dist[i, :].copy()
        d[i] = np.inf
        nn_idx = np.argsort(d)[:top_n]
        for rank, j in enumerate(nn_idx, start=1):
            rows.append(
                {
                    "query_id": ids[i],
                    "neighbour_rank": rank,
                    "neighbour_id": ids[j],
                    "distance": float(dist[i, j]),
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap_pdf(
    *,
    dist: np.ndarray,
    ids: List[str],
    order: np.ndarray,
    out_pdf: str | Path,
    title: str,
) -> None:
    """
    Plot a clustered heatmap (distance matrix) and save as PDF.

    Parameters
    ----------
    dist : np.ndarray
        Square distance matrix.
    ids : list of str
        Identifiers.
    order : np.ndarray
        Leaf order indices.
    out_pdf : str or Path
        Output PDF path.
    title : str
        Title.
    """
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    dist_ord = dist[np.ix_(order, order)]
    ids_ord = [ids[i] for i in order.tolist()]

    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    im = ax.imshow(dist_ord, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(ids_ord)))
    ax.set_yticks(np.arange(len(ids_ord)))
    ax.set_xticklabels(ids_ord, rotation=90, fontsize=8)
    ax.set_yticklabels(ids_ord, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Cluster embeddings and create hierarchical heatmaps."
    )

    parser.add_argument("--embeddings_tsv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--id_col", type=str, default="cpd_id")
    parser.add_argument(
        "--aggregate_method",
        type=str,
        default="median",
        choices=["median", "mean"],
    )

    parser.add_argument(
        "--scale",
        type=str,
        default="none",
        choices=["none", "robust", "standard"],
        help="Optional feature scaling prior to distances (recommended: robust).",
    )

    parser.add_argument(
        "--distance",
        type=str,
        default="cosine,spearman",
        help="Comma-separated distances: cosine, spearman, euclidean",
    )

    parser.add_argument(
        "--linkage",
        type=str,
        default="average",
        choices=["average", "complete", "single", "ward"],
        help="Hierarchical linkage method.",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Nearest neighbours to report per compound.",
    )

    parser.add_argument("--random_seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    configure_logging()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_tsv(path=args.embeddings_tsv)
    meta_cols = detect_metadata_columns(df=df)

    if args.id_col not in df.columns:
        raise KeyError(
            f"Identifier column '{args.id_col}' not found. "
            f"Available columns: {list(df.columns)[:30]} ..."
        )

    feat_cols = numeric_feature_columns(df=df, metadata_cols=meta_cols)
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns detected.")

    LOGGER.info("Loaded table: rows=%s cols=%s", df.shape[0], df.shape[1])
    LOGGER.info("Detected %s feature columns; metadata=%s", len(feat_cols), meta_cols)

    agg = aggregate_to_compounds(
        df=df,
        id_col=args.id_col,
        feature_cols=feat_cols,
        method=args.aggregate_method,
    )
    ids = agg[args.id_col].astype(str).tolist()
    x = agg[feat_cols].to_numpy(dtype=float)

    LOGGER.info("Aggregated to %s compounds.", x.shape[0])

    if x.shape[0] < 2:
        raise ValueError("Need at least 2 compounds to compute distances/cluster.")

    # Optional scaling
    if args.scale == "robust":
        x_use = robust_scale(x=x)
        LOGGER.info("Applied robust scaling (median/IQR).")
    elif args.scale == "standard":
        x_use = standard_scale(x=x)
        LOGGER.info("Applied standard scaling (mean/SD).")
    else:
        x_use = x

    # Choose which distances to compute
    dist_names = [d.strip().lower() for d in args.distance.split(",") if d.strip()]
    dist_mats: Dict[str, np.ndarray] = {}

    for name in dist_names:
        if name == "cosine":
            dist_mats[name] = cosine_distance_matrix(x=x_use)
        elif name == "spearman":
            dist_mats[name] = spearman_distance_matrix(x=x_use)
        elif name == "euclidean":
            dist_mats[name] = euclidean_distance_matrix(x=x_use)
        else:
            raise ValueError(f"Unknown distance: {name}")

        LOGGER.info("Computed %s distance matrix.", name)

    # Save distance matrices + neighbours + heatmaps
    for name, dist in dist_mats.items():
        # Linkage constraints
        if args.linkage == "ward" and name != "euclidean":
            LOGGER.warning(
                "Ward linkage is only appropriate for Euclidean distance. "
                "Skipping %s with ward linkage.",
                name,
            )
            continue

        if not np.isfinite(dist).all():
            LOGGER.warning(
                "Distance matrix '%s' contains non-finite values; replacing with 1.0.",
                name,
            )
            dist = np.where(np.isfinite(dist), dist, 1.0)
            np.fill_diagonal(dist, 0.0)


        Z, order = hierarchical_order(dist=dist, method=args.linkage)

        # Leaf order output
        order_df = pd.DataFrame(
            {
                "leaf_rank": np.arange(order.shape[0]) + 1,
                args.id_col: [ids[i] for i in order.tolist()],
            }
        )
        write_tsv(df=order_df, path=out_dir / f"leaf_order_{name}_{args.linkage}.tsv")

        # Cluster labels (optional: cut into k clusters, where k=min(5,n))
        k = min(5, dist.shape[0])
        if k >= 2:
            labels = fcluster(Z, t=k, criterion="maxclust")
            cl_df = pd.DataFrame({args.id_col: ids, "cluster": labels.astype(int)})
            write_tsv(df=cl_df, path=out_dir / f"clusters_{name}_{args.linkage}_k{k}.tsv")

        # Distance matrix TSV (wide)
        dist_df = pd.DataFrame(dist, index=ids, columns=ids)
        dist_df.insert(0, args.id_col, ids)
        dist_path = out_dir / f"pairwise_distance_{name}.tsv"
        dist_df.to_csv(dist_path, sep="\t", index=False)

        # Nearest neighbours
        nn_df = nearest_neighbours(ids=ids, dist=dist, top_n=args.top_n)
        write_tsv(df=nn_df, path=out_dir / f"nearest_neighbours_{name}.tsv")

        # Heatmap PDF (clustered order)
        plot_heatmap_pdf(
            dist=dist,
            ids=ids,
            order=order,
            out_pdf=out_dir / f"heatmap_{name}_{args.linkage}.pdf",
            title=f"Distance heatmap ({name}), linkage={args.linkage}, n={len(ids)}",
        )

        LOGGER.info("Wrote outputs for distance=%s", name)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
