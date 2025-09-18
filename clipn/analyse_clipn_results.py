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
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
import re
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



# Never treat these as features, even if numeric
BANNED_FEATURES_EXACT = {
    "ImageNumber",
    "Number_Object_Number",
    "ObjectNumber",
    "TableNumber",
}

# Heuristics for metadata/housekeeping columns (case-insensitive)
BANNED_FEATURES_REGEX = re.compile(
    r"""(?ix)
        ( ^metadata($|_)         # Metadata*, *_Metadata
        | _metadata$
        | ^filename_             # FileName_*
        | ^pathname_             # PathName_*
        | ^url_                  # URL_*
        | ^parent_               # Parent_*
        | ^children_             # Children_*
        | (^|_)imagenumber$      # ImageNumber (optionally with a prefix_)
        | ^number_object_number$ # Number_Object_Number
        | ^objectnumber$         # ObjectNumber
        | ^tablenumber$          # TableNumber
        )
    """
)

def _is_metadata_like(col: str) -> bool:
    """
    Return True if a column name should be treated as metadata/housekeeping and
    excluded from feature analyses.

    Parameters
    ----------
    col : str
        Column name to test.

    Returns
    -------
    bool
        True if the column is metadata-like, otherwise False.
    """
    cname = str(col)
    if cname in BANNED_FEATURES_EXACT:
        return True
    return bool(BANNED_FEATURES_REGEX.search(cname.lower()))


def select_latent_features(df: pd.DataFrame, prefix: str | None, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select latent numeric feature columns for neighbour analysis, excluding
    metadata/housekeeping columns such as ImageNumber and Number_Object_Number.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latent features and metadata.
    prefix : str | None
        If provided, select columns that start with this prefix and are numeric.
        Otherwise select digit-named columns (e.g. "0", "1", ...) that are numeric.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A copy of the feature matrix and the list of selected feature names.

    Raises
    ------
    ValueError
        If no usable latent feature columns are found after exclusions.
    """
    if prefix:
        candidates = [
            c for c in df.columns
            if isinstance(c, str) and c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        candidates = [
            c for c in df.columns
            if isinstance(c, str) and c.isdigit() and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not candidates:
        logger.error("No latent feature columns found (prefix=%s).", prefix)
        raise ValueError("No latent feature columns found. Check column names and --latent_prefix.")

    # Drop metadata-like names defensively
    cols = [c for c in candidates if not _is_metadata_like(c)]
    dropped = [c for c in candidates if c not in cols]
    if dropped:
        logger.info(
            "Excluded %d metadata/housekeeping columns from latent set (first few: %s)",
            len(dropped), ", ".join(map(str, dropped[:10]))
        )

    if not cols:
        logger.error("All candidate latent columns were excluded as metadata/housekeeping.")
        raise ValueError("No usable latent features remain after exclusions.")

    X = df[cols].copy()

    # Handle NaNs: fill with 0 (document and proceed)
    n_nans = int(X.isna().sum().sum())
    if n_nans:
        logger.warning("Latent feature matrix contains %d NaNs; filling with 0.", n_nans)
        X = X.fillna(value=0)

    logger.info("Selected %d latent columns. Feature matrix shape: %s", len(cols), X.shape)
    return X, cols




def build_mutual_knn_edges(*, nn_df: pd.DataFrame, k_mutual: int = 10) -> List[Tuple[str, str, float]]:
    """
    Construct an undirected edge set using mutual k-NN to reduce hubbiness.

    Parameters
    ----------
    nn_df : pandas.DataFrame
        Columns: ['cpd_id', 'neighbour_id', 'distance'] (smaller = closer).
    k_mutual : int, optional
        k for mutual k-NN (default: 10).

    Returns
    -------
    list[tuple[str, str, float]]
        Edges as (source, target, distance), undirected and de-duplicated.
    """
    ranks: Dict[str, Dict[str, int]] = {}
    for src, group in nn_df.groupby("cpd_id"):
        g = group.sort_values("distance", ascending=True).reset_index(drop=True)
        ranks[src] = {row["neighbour_id"]: int(i) for i, row in g.head(k_mutual).iterrows()}

    pairs = set()
    for i, neighs in ranks.items():
        for j in neighs:
            if i in ranks.get(j, {}):
                a, b = sorted([i, j])
                pairs.add((a, b))

    dist_lookup: Dict[Tuple[str, str], float] = {}
    for _, r in nn_df.iterrows():
        a, b = sorted([r["cpd_id"], r["neighbour_id"]])
        d = float(r["distance"])
        if (a, b) not in dist_lookup or d < dist_lookup[(a, b)]:
            dist_lookup[(a, b)] = d

    return [(a, b, dist_lookup[(a, b)]) for a, b in pairs]


def local_scaling_weights(
    *, nn_df: pd.DataFrame, edges: List[Tuple[str, str, float]], k_sigma: int = 7
) -> List[Tuple[str, str, float, float]]:
    """
    Apply Zelnik-Manor & Perona local scaling to compute robust affinities.

    w_ij = exp( - d_ij^2 / (sigma_i * sigma_j) ), where sigma_i is the
    distance to the k_sigma-th neighbour of node i.

    Parameters
    ----------
    nn_df : pandas.DataFrame
        Neighbour table with 'cpd_id', 'neighbour_id', 'distance'.
    edges : list[tuple[str, str, float]]
        Undirected edges as (source, target, distance).
    k_sigma : int, optional
        Rank used to estimate local scale (default: 7).

    Returns
    -------
    list[tuple[str, str, float, float]]
        (source, target, distance, local_scaled_weight).
    """
    sigma: Dict[str, float] = {}
    for src, group in nn_df.groupby("cpd_id"):
        g = group.sort_values("distance", ascending=True).reset_index(drop=True)
        s = float(g.iloc[min(k_sigma - 1, len(g) - 1)]["distance"])
        sigma[src] = s if s > 0.0 else 1e-12

    out: List[Tuple[str, str, float, float]] = []
    for a, b, d in edges:
        si, sj = sigma.get(a, 1e-12), sigma.get(b, 1e-12)
        denom = max(si * sj, 1e-12)
        w = math.exp(- (d * d) / denom)
        out.append((a, b, float(d), float(w)))
    return out


def cap_degrees_greedily(
    *, weighted_edges: List[Tuple[str, str, float, float]], max_degree: int = 6
) -> List[Tuple[str, str, float, float]]:
    """
    Greedily keep strongest edges while enforcing degree ≤ max_degree.

    Parameters
    ----------
    weighted_edges : list[tuple[str, str, float, float]]
        Edges as (source, target, distance, weight).
    max_degree : int, optional
        Degree cap per node (default: 6).

    Returns
    -------
    list[tuple[str, str, float, float]]
        Pruned edge list.
    """
    deg: Dict[str, int] = {}
    kept: List[Tuple[str, str, float, float]] = []
    for a, b, d, w in sorted(weighted_edges, key=lambda x: x[3], reverse=True):
        if deg.get(a, 0) < max_degree and deg.get(b, 0) < max_degree:
            kept.append((a, b, d, w))
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    return kept


def rescale_for_vis(*, weights: np.ndarray, new_min: float = 1.0, new_max: float = 10.0) -> np.ndarray:
    """
    Rescale weights into a range suitable for vis-network edge widths.

    Parameters
    ----------
    weights : numpy.ndarray
        Original weights (e.g., local scaling affinities in [0, 1]).
    new_min : float, optional
        New minimum (default: 1.0).
    new_max : float, optional
        New maximum (default: 10.0).

    Returns
    -------
    numpy.ndarray
        Rescaled weights in [new_min, new_max].
    """
    w = weights.astype(float)
    w_min, w_max = float(w.min()), float(w.max())
    if w_max == w_min:
        return np.full_like(w, new_min)
    return new_min + (w - w_min) * (new_max - new_min) / (w_max - w_min)


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
    Generate an interactive compound similarity network and TSV nodes/edges
    using mutual k-NN + local scaling, with a degree cap.

    Notes
    -----
    The `threshold` parameter now applies to the **locally scaled weight**
    in [0, 1] (higher = stronger), rather than raw distance. If you previously
    used a small distance threshold (e.g. 0.2), start with 0.4–0.6 here.
    Lower values keep more edges; higher values prune more.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with latent features and metadata.
    output_dir : str | Path
        Output directory for TSVs and HTML.
    threshold : float
        Local-scaling weight threshold in [0, 1] to keep edges.
    metric : str
        Nearest neighbour metric (e.g., 'cosine', 'euclidean').
    prefix : str | None
        Latent feature prefix; if None, uses digit-named columns.
    id_col : str
        Identifier column name (e.g., 'cpd_id').
    dataset_col : str
        Dataset/Library column name.
    type_col : str
        Optional type/label column name (e.g., 'cpd_type'); can be missing.
    max_edges_per_node : int
        Degree cap per node (higher = denser graph).
    logger : logging.Logger
        Logger instance.
    """
    # 1) Select latent features
    X, latent_cols = select_latent_features(df=df, prefix=prefix, logger=logger)
    n_samples = X.shape[0]
    logger.info("Selected %d latent columns. Feature matrix shape: %s", len(latent_cols), X.shape)

    if n_samples < 2:
        logger.warning("Not enough samples for a network. Skipping.")
        return

    # 2) Nearest neighbours
    k_all = min(n_samples, max(2, max_edges_per_node * 2))
    nn = NearestNeighbors(n_neighbors=k_all, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    ids = df[id_col].astype(str).values
    dsets = df[dataset_col].astype(str).values
    types = df[type_col].astype(str).values if type_col in df.columns else np.array([""] * n_samples)

    # 3) Build full directional NN table (exclude self)
    records = []
    for i in range(n_samples):
        for dist, j in zip(distances[i][1:], indices[i][1:]):
            records.append((ids[i], ids[j], float(dist)))
    nn_df = pd.DataFrame(records, columns=["cpd_id", "neighbour_id", "distance"])

    # 4) Mutual k-NN + local scaling + degree cap
    k_mutual = max_edges_per_node * 2  # a little more generous
    k_sigma = min(7, max(3, k_all - 1))

    mutual_edges = build_mutual_knn_edges(nn_df=nn_df, k_mutual=k_mutual)
    weighted = local_scaling_weights(nn_df=nn_df, edges=mutual_edges, k_sigma=k_sigma)

    # Apply local-scale weight threshold
    if threshold is not None:
        weighted = [e for e in weighted if e[3] >= float(threshold)]

    # Enforce degree cap
    weighted = cap_degrees_greedily(weighted_edges=weighted, max_degree=max_edges_per_node)

    # 5) TSV outputs (nodes + edges)
    nodes_df = pd.DataFrame({id_col: ids, dataset_col: dsets})
    if type_col in df.columns:
        nodes_df[type_col] = types

    edges_df = pd.DataFrame(weighted, columns=["source", "target", "distance", "weight_local_scaled"])
    edges_df["weight_for_vis"] = rescale_for_vis(
        weights=edges_df["weight_local_scaled"].to_numpy(), new_min=1.0, new_max=10.0
    )

    write_tsv(df=nodes_df, path=Path(output_dir) / "network_nodes.tsv", logger=logger, index=False)
    write_tsv(df=edges_df, path=Path(output_dir) / "network_edges.tsv", logger=logger, index=False)

    # 6) Optional HTML visualisation
    if not PYVIS_AVAILABLE:
        logger.warning("pyvis not available. Skipping interactive HTML network.")
        return

    try:
        net = Network(height="800px", width="100%", notebook=False, directed=False)

        # Colour nodes: prefer type_col if present, else dataset_col
        colour_by = type_col if type_col in df.columns else dataset_col
        for i in range(n_samples):
            label = ids[i]
            group = (types[i] if colour_by == type_col else dsets[i])
            title = f"{colour_by}: {group}"
            net.add_node(n_id=label, label=label, title=title, group=group)

        for r in edges_df.itertuples(index=False):
            net.add_edge(source=r.source, to=r.target, value=float(r.weight_for_vis))

        html_path = Path(output_dir) / "compound_similarity_network.html"
        try:
            net.write_html(str(html_path))
        except TypeError:
            net.save_graph(str(html_path))
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
    parser.add_argument("--dataset_col", default="Library", help="Dataset column name (default: Library)")
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
    # make dataset_col optional/fallback ---
    if args.dataset_col not in df.columns:
        if "Library" in df.columns:
            df[args.dataset_col] = df["Library"].astype(str)
            logger.warning(
                "Missing dataset column '%s' in %s; using 'Library' as fallback.",
                args.dataset_col, args.latent_csv,
            )
        else:
            df[args.dataset_col] = "all"
            logger.warning(
                "Missing dataset column '%s' in %s; creating constant 'all'.",
                args.dataset_col, args.latent_csv,
            )



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
