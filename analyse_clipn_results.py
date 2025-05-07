#!/usr/bin/env python3
# coding: utf-8

"""
CLIPn Post-Analysis Script
---------------------------
Performs post-analysis of CLIPn latent space embeddings.

Features
--------
- Cluster summaries by Cluster, Dataset, cpd_type
- Test compound proximity to reference dataset or compound list
- Nearest neighbours analysis
- Optional interactive network output (via pyvis)
- Robust feature selection and flexible distance metrics

Inputs
------
--latent_csv           TSV with latent embeddings + metadata
--output_dir           Output folder
--test_dataset         One or more test dataset names
--reference_dataset    One or more reference dataset names or compound IDs
--threshold            Distance threshold for network edges
--nn_metric            Distance metric for nearest neighbours (e.g. euclidean, cosine)
--latent_prefix        Optional prefix to identify latent columns (default: digit-only columns)

Outputs
-------
- Cluster summary TSV
- Nearest neighbours TSV
- Test compound proximity report
- Interactive compound similarity network (if pyvis works)
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx

# Try to import pyvis
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def select_latent_features(df, prefix=None):
    """
    Select latent numeric columns for neighbour analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with latent features and metadata.
    prefix : str or None
        Prefix to select latent columns. If None, use digit-only column names.

    Returns
    -------
    pd.DataFrame
        Subset of input dataframe with selected latent features.
    """
    if prefix:
        return df[[col for col in df.columns if col.startswith(prefix) and pd.api.types.is_numeric_dtype(df[col])]].copy()
    else:
        return df[[col for col in df.columns if col.isdigit() and pd.api.types.is_numeric_dtype(df[col])]].copy()

def summarise_clusters(df, output_dir):
    """
    Generate summary table of cluster composition by type and dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with cluster and metadata columns.
    output_dir : str
        Directory where output summary file will be saved.
    """
    if "Cluster" not in df.columns:
        logger.warning("No Cluster column found in input. Skipping summary.")
        return
    logger.info("Creating cluster summary...")
    cluster_summary = df.groupby(["Cluster", "cpd_type", "Dataset"]).agg({"cpd_id": "count"}).reset_index()
    summary_path = os.path.join(output_dir, "cluster_summary_by_type_and_dataset.tsv")
    cluster_summary.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Cluster summary saved to {summary_path}")


def compute_nearest_neighbours(df, n_neighbours=100, metric="euclidean", prefix=None):
    """
    Compute nearest neighbours using latent space features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with latent features and metadata.
    n_neighbours : int
        Number of neighbours to retrieve (excluding self).
    metric : str
        Distance metric to use for neighbour search.
    prefix : str or None
        Prefix to select latent features. If None, use digit-only columns.

    Returns
    -------
    pd.DataFrame
        Table with nearest neighbours per compound and distances.
    """
    logger.info(f"Computing nearest neighbours using metric: {metric}")
    features = select_latent_features(df, prefix)
    nn = NearestNeighbors(n_neighbors=n_neighbours + 1, metric=metric).fit(features)
    distances, indices = nn.kneighbors(features)

    rows = []
    for i, (d_row, idx_row) in enumerate(zip(distances, indices)):
        source = df.iloc[i]["cpd_id"]
        for dist, j in zip(d_row[1:], idx_row[1:]):
            neighbour = df.iloc[j]["cpd_id"]
            rows.append({"cpd_id": source, "neighbour_id": neighbour, "distance": dist})
    return pd.DataFrame(rows)


def analyse_test_vs_reference(df, test_datasets, reference_list, output_dir):
    """
    Count how many neighbours of each test compound match reference list.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with latent features and metadata.
    test_datasets : list
        Names of test datasets.
    reference_list : list
        List of reference compound IDs.
    output_dir : str
        Directory to save the output summary.
    """
    logger.info("Evaluating test compound proximity to reference set...")
    features = select_latent_features(df)
    nn = NearestNeighbors(n_neighbors=6).fit(features)
    distances, indices = nn.kneighbors(features)

    reference_list = set(x.upper() for x in reference_list)
    results = []
    for i, idx_row in enumerate(indices):
        focal = df.iloc[i]
        if str(focal["Dataset"]).upper() not in [ds.upper() for ds in test_datasets]:
            continue
        focal_id = str(focal["cpd_id"]).upper()
        reference_hits = 0
        for j in idx_row[1:]:
            neighbour = str(df.iloc[j]["cpd_id"]).upper()
            if neighbour in reference_list:
                reference_hits += 1
        results.append({"cpd_id": focal_id, "reference_neighbours": reference_hits})

    out_path = os.path.join(output_dir, "test_reference_neighbour_overlap.tsv")
    pd.DataFrame(results).to_csv(out_path, sep="\t", index=False)
    logger.info(f"Test compound reference neighbour stats saved to {out_path}")


def generate_network(df, output_dir, threshold, metric="euclidean", prefix=None):
    """
    Generate interactive compound similarity network based on threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latent features and compound IDs.
    output_dir : str
        Directory to save network HTML.
    threshold : float
        Distance threshold to draw an edge between nodes.
    metric : str
        Distance metric to use.
    prefix : str or None
        Prefix for latent feature columns.
    """
    logger.info("Generating interactive network (if pyvis is installed)...")
    if not PYVIS_AVAILABLE:
        logger.warning("Pyvis not available. Skipping network visualisation.")
        return

    try:
        features = select_latent_features(df, prefix)
        nn = NearestNeighbors(n_neighbors=100, metric=metric).fit(features)
        distances, indices = nn.kneighbors(features)

        g = nx.Graph()
        for i, (d_row, idx_row) in enumerate(zip(distances, indices)):
            source = df.iloc[i]["cpd_id"]
            g.add_node(source)
            for dist, j in zip(d_row[1:], idx_row[1:]):
                target = df.iloc[j]["cpd_id"]
                if dist < threshold:
                    g.add_edge(source, target, weight=dist)

        net = Network(height="800px", width="100%")
        net.from_nx(g)
        output_html = os.path.join(output_dir, "compound_similarity_network.html")
        net.write_html(output_html)
        logger.info(f"Interactive network visualisation saved to '{output_html}'")

    except Exception as e:
        logger.error(f"Failed to render network. Error: {e}")

def main():
    """
    Main driver function to run CLIPn post-analysis.
    """
    parser = argparse.ArgumentParser(description="CLIPn Latent Post-Analysis")
    parser.add_argument("--latent_csv", required=True, help="TSV with latent space and metadata")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3, help="Distance threshold for network edges")
    parser.add_argument("--test_dataset", nargs="+", help="Test dataset(s) to evaluate proximity")
    parser.add_argument("--reference_dataset", nargs="+", default=[
        "MCP09", "MCP05", "DDD02387619", "DDD02443214", "DDD02454019", "DDD02454403",
        "DDD02459457", "DDD02487111", "DDD02487311", "DDD02589868", "DDD02591200",
        "DDD02591362", "DDD02941115", "DDD02941193", "DDD02947912", "DDD02947919",
        "DDD02948915", "DDD02948916", "DDD02948926", "DDD02952619", "DDD02952620",
        "DDD02955130", "DDD02958365"
    ], help="Reference datasets or compound IDs")
    parser.add_argument("--nn_metric", type=str, default="euclidean", help="Metric for nearest neighbour search")
    parser.add_argument("--latent_prefix", type=str, default=None, help="Prefix for latent features (default uses digit-only columns)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Reading latent CSV: {args.latent_csv}")
    df = pd.read_csv(args.latent_csv, sep="\t")

    summarise_clusters(df, args.output_dir)

    nn_df = compute_nearest_neighbours(df, metric=args.nn_metric, prefix=args.latent_prefix)
    nn_out = os.path.join(args.output_dir, "nearest_neighbours.tsv")
    nn_df.to_csv(nn_out, sep="\t", index=False)
    logger.info(f"Nearest neighbours saved to {nn_out}")

    if args.test_dataset and args.reference_dataset:
        analyse_test_vs_reference(df, args.test_dataset, args.reference_dataset, args.output_dir)

    generate_network(df, args.output_dir, threshold=args.threshold, metric=args.nn_metric, prefix=args.latent_prefix)

    logger.info("Post-analysis complete.")

if __name__ == "__main__":
    main()
