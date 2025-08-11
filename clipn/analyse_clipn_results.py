#!/usr/bin/env python3
# coding: utf-8

"""
CLIPn Post-Analysis
-------------------

Utilities for exploring CLIPn latent embeddings.

Features
--------
- Cluster summaries by Cluster × Dataset × cpd_type
- Nearest neighbours (configurable metric/k), with robust handling
- Test-vs-reference proximity assessment
- Optional interactive similarity network (pyvis), plus TSV nodes/edges
- Verbose logging and guard rails for large datasets

Conventions
-----------
- Inputs are TSV; all outputs are TSV (no comma-separated files).
- Latent feature columns are either digit-named (e.g. "0", "1", …)
  or start with an optional prefix (--latent_prefix).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Optional dependency: pyvis for interactive HTML networks
try:
    from pyvis.network import Network  # type: ignore[import]
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False


# =========
# Logging
# =========

def setup_logging(output_dir: str | Path, log_name: str = "post_analysis") -> logging.Logger:
    """
    Configure console and file logging.

    Parameters
    ----------
    output_dir : str | Path
        Directory where logs will be written.
    log_name : str
        Base name for the log file (without extension).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{log_name}.log"

    logger = logging.getLogger("clipn_post")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fileh = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(stream)
    logger.addHandler(fileh)
    logger.info("Logging to %s", log_path)
    return logger


# =========
# Helpers
# =========

def validate_columns(df: pd.DataFrame, required: Iterable[str], logger: logging.Logger) -> None:
    """
    Validate that required columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required : Iterable[str]
        Column names that must be present.
    logger : logging.Logger
        Logger.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")


def select_latent_features(df: pd.DataFrame, prefix: str | None, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select latent numeric feature columns for neighbour analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latent features and metadata.
    prefix : str | None
        If provided, select columns that start with this prefix and are numeric.
        Otherwise select digit-named columns (e.g. "0", "1", …) that are numeric.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (Feature matrix copy, list of selected feature column names).
    """
    if prefix:
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = [
            c for c in df.columns
            if (isinstance(c, str) and c.isdigit()) and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not cols:
        logger.error("No latent feature columns found (prefix=%s).", prefix)
        raise ValueError("No latent feature columns found. Check column names and --latent_prefix.")

    X = df[cols].copy()

    # Handle NaNs: fill with 0 (document and proceed)
    n_nans = int(X.isna().sum().sum())
    if n_nans:
        logger.warning("Latent feature matrix contains %d NaNs; filling with 0.", n_nans)
        X = X.fillna(value=0)

    logger.info("Selected %d latent columns. Feature matrix shape: %s", len(cols), X.shape)
    return X, cols


def write_tsv(df: pd.DataFrame, path: str | Path, logger: logging.Logger, index: bool = False) -> None:
    """
    Write a DataFrame to a TSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    path : str | Path
        Output file path.
    logger : logging.Logger
        Logger instance.
    index : bool
        Whether to write the index column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=path, sep="\t", index=index)
    logger.info("Wrote %s rows to %s", len(df), path)


# ===================
# Analysis functions
# ===================

def summarise_clusters(
    df: pd.DataFrame,
    output_dir: str | Path,
    cluster_col: str,
    id_col: str,
    dataset_col: str,
    type_col: str,
    logger: logging.Logger,
) -> None:
    """
    Generate a summary table of cluster composition by type and dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cluster and metadata columns.
    output_dir : str | Path
        Directory where output summary files will be saved.
    cluster_col : str
        Column holding cluster labels (e.g., 'Cluster').
    id_col : str
        Compound identifier column (e.g., 'cpd_id').
    dataset_col : str
        Dataset column name (e.g., 'Dataset').
    type_col : str
        Compound type column (e.g., 'cpd_type').
    logger : logging.Logger
        Logger instance.
    """
    if cluster_col not in df.columns:
        logger.warning("No '%s' column found. Skipping cluster summary.", cluster_col)
        return

    logger.info("Creating cluster summary...")
    # Long summary table
    cluster_summary = (
        df.groupby([cluster_col, type_col, dataset_col], dropna=False)[id_col]
        .size()
        .rename("count")
        .reset_index()
    )
    write_tsv(
        df=cluster_summary,
        path=Path(output_dir) / "cluster_summary_by_type_and_dataset.tsv",
        logger=logger,
        index=False,
    )

    # Optional: a pivoted view for readability
    try:
        pivot = cluster_summary.pivot_table(
            index=[cluster_col, type_col],
            columns=dataset_col,
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        pivot_path = Path(output_dir) / "cluster_summary_pivot.tsv"
        pivot.to_csv(path_or_buf=pivot_path, sep="\t")
        logger.info("Wrote pivoted cluster summary to %s", pivot_path)
    except Exception as exc:
        logger.debug("Could not create pivoted cluster summary: %s", exc)


def compute_nearest_neighbours(
    df: pd.DataFrame,
    n_neighbours: int,
    metric: str,
    prefix: str | None,
    id_col: str,
    dataset_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute nearest neighbours using latent space features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latent features and metadata (row order preserved).
    n_neighbours : int
        Number of neighbours to retrieve (excluding self).
    metric : str
        Distance metric (e.g., 'euclidean', 'cosine').
    prefix : str | None
        Prefix for latent feature columns; if None, uses digit-named columns.
    id_col : str
        Identifier column name.
    dataset_col : str
        Dataset column name.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Table with nearest neighbours per item and distances.
        Columns: [id_col, 'neighbour_id', 'distance', dataset_col, 'neighbour_dataset'].
    """
    X, cols = select_latent_features(df=df, prefix=prefix, logger=logger)

    n_samples = X.shape[0]
    k = min(n_neighbours + 1, max(1, n_samples))
    if k <= 1:
        logger.warning("Not enough rows (%d) to compute neighbours.", n_samples)
        return pd.DataFrame(columns=[id_col, "neighbour_id", "distance", dataset_col, "neighbour_dataset"])

    logger.info("Fitting NearestNeighbors on %d samples (%d dims), metric=%s, k=%d", n_samples, X.shape[1], metric, k)
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    rows = []
    # Build a fast view to avoid repeated iloc
    ids = df[id_col].astype(str).values
    dsets = df[dataset_col].astype(str).values

    for i in range(n_samples):
        src_id = ids[i]
        src_ds = dsets[i]
        # skip self at position 0
        for dist, j in zip(distances[i][1:], indices[i][1:]):
            rows.append(
                {
                    id_col: src_id,
                    "neighbour_id": ids[j],
                    "distance": float(dist),
                    dataset_col: src_ds,
                    "neighbour_dataset": dsets[j],
                }
            )

    out = pd.DataFrame(rows)
    logger.info("Computed %d neighbour relations.", len(out))
    return out


def analyse_test_vs_reference(
    df: pd.DataFrame,
    test_datasets: List[str],
    reference_ids: List[str],
    output_dir: str | Path,
    metric: str,
    top_k: int,
    prefix: str | None,
    id_col: str,
    dataset_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Count how many nearest neighbours of each test item match a reference ID.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latent features and metadata.
    test_datasets : list[str]
        Dataset names to treat as the test set.
    reference_ids : list[str]
        Reference compound identifiers.
    output_dir : str | Path
        Directory to save the output summary.
    metric : str
        Distance metric to use.
    top_k : int
        Number of neighbours to check for each test item (excluding self).
    prefix : str | None
        Latent feature prefix (see select_latent_features).
    id_col : str
        Identifier column name.
    dataset_col : str
        Dataset column name.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Per-test item summary of reference neighbour counts and fraction.
    """
    X, cols = select_latent_features(df=df, prefix=prefix, logger=logger)

    n_samples = X.shape[0]
    k = min(top_k + 1, max(1, n_samples))
    if k <= 1:
        logger.warning("Not enough rows to compute neighbours (n=%d).", n_samples)
        return pd.DataFrame(columns=[id_col, "reference_neighbours", "fraction_reference_neighbours"])

    logger.info("Building NN index for test-vs-reference (metric=%s, k=%d)", metric, k)
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    test_set_upper = {str(x).upper() for x in test_datasets}
    ref_ids_upper = {str(x).upper() for x in reference_ids}

    ids = df[id_col].astype(str).values
    dsets = df[dataset_col].astype(str).values

    results = []
    for i, neigh_idx in enumerate(indices):
        if str(dsets[i]).upper() not in test_set_upper:
            continue

        neigh_ids = [str(ids[j]).upper() for j in neigh_idx[1:]]  # drop self
        hits = sum(1 for nid in neigh_ids if nid in ref_ids_upper)
        frac = hits / max(1, len(neigh_ids))
        results.append(
            {
                id_col: ids[i],
                "reference_neighbours": int(hits),
                "fraction_reference_neighbours": float(frac),
                dataset_col: dsets[i],
            }
        )

    out = pd.DataFrame(results)
    write_tsv(
        df=out,
        path=Path(output_dir) / "test_reference_neighbour_overlap.tsv",
        logger=logger,
        index=False,
    )
    return out


def generate_network(
    df: pd.DataFrame,
    output_dir: str | Path,
    threshold: float,
    metric: str,
    prefix: str | None,
    id_col: str,
    dataset_col: str,
    type_col: str,
    max_edges_per_node: int,
    logger: logging.Logger,
) -> None:
    """
    Generate an interactive compound similarity network and TSV edges/nodes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latent features and metadata.
    output_dir : str | Path
        Directory where network files will be saved.
    threshold : float
        Distance threshold to draw an edge (<= threshold).
        Note: for cosine distance, values lie in [0, 2]; smaller is more similar.
    metric : str
        Distance metric for NearestNeighbours.
    prefix : str | None
        Latent feature prefix for selection.
    id_col : str
        Identifier column name.
    dataset_col : str
        Dataset column name.
    type_col : str
        Compound type column name.
    max_edges_per_node : int
        Cap the number of kept edges per node (prevents hairballs).
    logger : logging.Logger
        Logger instance.
    """
    X, cols = select_latent_features(df=df, prefix=prefix, logger=logger)
    n_samples = X.shape[0]
    if n_samples < 2:
        logger.warning("Not enough rows (%d) to build a network.", n_samples)
        return

    k = min(100, n_samples)  # initial local neighbourhood for candidate edges
    logger.info("Finding candidate edges with k=%d (metric=%s), threshold=%.4f", k, metric, threshold)
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    ids = df[id_col].astype(str).values
    dsets = df[dataset_col].astype(str).values
    types = df[type_col].astype(str).values if type_col in df.columns else np.array([""] * n_samples)

    # Build undirected edge list with thresholding and per-node cap
    edges = []
    kept_per_node = {i: 0 for i in range(n_samples)}
    for i in range(n_samples):
        for dist, j in zip(distances[i][1:], indices[i][1:]):  # skip self
            if dist <= threshold:
                if kept_per_node[i] >= max_edges_per_node:
                    continue
                kept_per_node[i] += 1
                a, b = sorted((i, j))
                edges.append((a, b, float(dist)))

    # Deduplicate undirected edges
    unique_edges = {}
    for a, b, w in edges:
        unique_edges[(a, b)] = min(unique_edges.get((a, b), w), w)
    logger.info("Kept %d unique edges after dedup.", len(unique_edges))

    # Write TSV nodes/edges
    nodes_df = pd.DataFrame(
        {id_col: ids, dataset_col: dsets, type_col: types}
    )
    edges_df = pd.DataFrame(
        [{"source": ids[a], "target": ids[b], "distance": w} for (a, b), w in unique_edges.items()]
    )

    write_tsv(df=nodes_df, path=Path(output_dir) / "network_nodes.tsv", logger=logger, index=False)
    write_tsv(df=edges_df, path=Path(output_dir) / "network_edges.tsv", logger=logger, index=False)

    # Optional HTML visualisation
    if not PYVIS_AVAILABLE:
        logger.warning("pyvis not available. Skipping interactive HTML network.")
        return

    try:
        net = Network(height="800px", width="100%", notebook=False, directed=False)
        # Map datasets to groups/colours
        for i in range(n_samples):
            net.add_node(n_id=ids[i], label=ids[i], title=f"{dataset_col}: {dsets[i]}", group=dsets[i])
        for (a, b), w in unique_edges.items():
            net.add_edge(source=ids[a], to=ids[b], value=max(1.0, 1.0 / (w + 1e-6)))
        html_path = Path(output_dir) / "compound_similarity_network.html"
        net.write_html(path=html_path)
        logger.info("Interactive network visualisation saved to %s", html_path)
    except Exception as exc:
        logger.error("Failed to render network HTML: %s", exc)


# =====
# Main
# =====

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CLIPn latent post-analysis")
    parser.add_argument("--latent_csv", required=True, help="TSV with latent embeddings + metadata (despite the name, expects TSV)")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    # Column names
    parser.add_argument("--id_col", default="cpd_id", help="Identifier column name (default: cpd_id)")
    parser.add_argument("--dataset_col", default="Dataset", help="Dataset column name (default: Dataset)")
    parser.add_argument("--type_col", default="cpd_type", help="Type/label column name (default: cpd_type)")
    parser.add_argument("--cluster_col", default="Cluster", help="Cluster label column name (default: Cluster)")

    # Latent features
    parser.add_argument("--latent_prefix", default=None, help="Prefix for latent columns (default: use digit-named columns)")

    # Nearest neighbour settings
    parser.add_argument("--nn_metric", default="cosine", help="Metric for nearest neighbour search (e.g., cosine, euclidean)")
    parser.add_argument("--n_neighbours", type=int, default=100, help="Number of neighbours to export per item (default: 100)")

    # Test vs reference
    parser.add_argument("--test_dataset", nargs="+", default=None, help="Test dataset name(s)")
    parser.add_argument("--reference_ids", nargs="+", default=None, help="Reference compound IDs")
    parser.add_argument("--top_k", type=int, default=5, help="Neighbours to count for test-vs-reference (default: 5)")

    # Network
    parser.add_argument("--network", action="store_true", help="If set, build the network outputs/HTML")
    parser.add_argument("--threshold", type=float, default=0.2, help="Distance threshold for network edges (default: 0.2)")
    parser.add_argument("--network_max_edges_per_node", type=int, default=10, help="Cap of edges kept per node (default: 10)")

    return parser.parse_args()


def main() -> None:
    """
    Main driver for CLIPn post-analysis.
    """
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir=args.output_dir, log_name="clipn_post_analysis")

    logger.info("Arguments: %s", vars(args))
    logger.info("Reading latent TSV: %s", args.latent_csv)
    df = pd.read_csv(filepath_or_buffer=args.latent_csv, sep="\t")

    # Validate expected columns exist
    validate_columns(
        df=df,
        required=[args.id_col, args.dataset_col],
        logger=logger,
    )
    if args.type_col not in df.columns:
        logger.warning("Type column '%s' not found; some summaries will omit it.", args.type_col)

    # Summaries (Cluster × Dataset × cpd_type)
    summarise_clusters(
        df=df,
        output_dir=args.output_dir,
        cluster_col=args.cluster_col,
        id_col=args.id_col,
        dataset_col=args.dataset_col,
        type_col=args.type_col,
        logger=logger,
    )

    # Nearest neighbours across all items
    nn_df = compute_nearest_neighbours(
        df=df,
        n_neighbours=args.n_neighbours,
        metric=args.nn_metric,
        prefix=args.latent_prefix,
        id_col=args.id_col,
        dataset_col=args.dataset_col,
        logger=logger,
    )
    write_tsv(df=nn_df, path=Path(args.output_dir) / "nearest_neighbours.tsv", logger=logger, index=False)

    # Test vs reference (optional)
    if args.test_dataset and args.reference_ids:
        analyse_test_vs_reference(
            df=df,
            test_datasets=args.test_dataset,
            reference_ids=args.reference_ids,
            output_dir=args.output_dir,
            metric=args.nn_metric,
            top_k=args.top_k,
            prefix=args.latent_prefix,
            id_col=args.id_col,
            dataset_col=args.dataset_col,
            logger=logger,
        )
    else:
        logger.info("Skipping test-vs-reference: provide both --test_dataset and --reference_ids to enable.")

    # Network (optional)
    if args.network:
        generate_network(
            df=df,
            output_dir=args.output_dir,
            threshold=args.threshold,
            metric=args.nn_metric,
            prefix=args.latent_prefix,
            id_col=args.id_col,
            dataset_col=args.dataset_col,
            type_col=args.type_col,
            max_edges_per_node=args.network_max_edges_per_node,
            logger=logger,
        )
    else:
        logger.info("Skipping interactive network (enable with --network).")

    logger.info("Post-analysis complete.")


if __name__ == "__main__":
    main()
