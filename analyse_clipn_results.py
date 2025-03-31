#!/usr/bin/env python3
"""
CLIPn Post-Analysis Script
---------------------------
Performs post-analysis of CLIPn latent space embeddings.

Features
--------
- Cluster summaries by Cluster, Dataset, cpd_type
- Test compound proximity to reference dataset
- Nearest neighbours analysis
- Optional interactive network output (via pyvis)
- Logging and future-ready for toxic tagging & enrichment stats

Inputs
------
--latent_csv           TSV with latent embeddings + metadata
--output_dir           Output folder
--test_dataset         One or more test dataset names
--reference_dataset    One or more reference dataset names
--threshold            Distance threshold for network edges

Outputs
-------
- Cluster summary TSV
- Nearest neighbours TSV
- Test compound proximity report
- Interactive compound similarity network (if pyvis works)

Author: Auto-generated for custom CLIPn analysis
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


def summarise_clusters(df, output_dir):
    """
    Summarise clusters by cluster, cpd_type, and Dataset.
    """
    if "Cluster" not in df.columns:
        logger.warning("No Cluster column found in input. Skipping summary.")
        return

    logger.info("Creating cluster summary...")
    cluster_summary = df.groupby(["Cluster", "cpd_type", "Dataset"]).agg({"cpd_id": "count"}).reset_index()
    summary_path = os.path.join(output_dir, "cluster_summary_by_type_and_dataset.tsv")
    cluster_summary.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Cluster summary saved to {summary_path}")


def compute_nearest_neighbours(df, n_neighbours=5):
    """
    Compute nearest neighbours for each compound using Euclidean distance.
    """
    logger.info("Computing nearest neighbours...")
    numeric = df.select_dtypes(include=[np.number])
    nn = NearestNeighbors(n_neighbors=n_neighbours + 1, metric="euclidean").fit(numeric)
    distances, indices = nn.kneighbors(numeric)

    rows = []
    for i, row in enumerate(indices):
        source = df.iloc[i]["cpd_id"]
        for j in row[1:]:
            neighbour = df.iloc[j]["cpd_id"]
            dist = np.linalg.norm(numeric.iloc[i] - numeric.iloc[j])
            rows.append({"cpd_id": source, "neighbour_id": neighbour, "distance": dist})

    return pd.DataFrame(rows)


def analyse_test_vs_reference(df, test_datasets, reference_datasets, output_dir):
    """
    For each test compound, determine how many of its top N neighbours are from reference.
    """
    logger.info("Evaluating test compound proximity to reference dataset...")
    numeric = df.select_dtypes(include=[np.number])
    nn = NearestNeighbors(n_neighbors=6).fit(numeric)
    distances, indices = nn.kneighbors(numeric)

    results = []
    for i, row in enumerate(indices):
        focal = df.iloc[i]
        if focal["Dataset"] not in test_datasets:
            continue
        focal_id = focal["cpd_id"]
        reference_hits = 0
        for j in row[1:]:  # skip self
            neighbour = df.iloc[j]
            if neighbour["Dataset"] in reference_datasets:
                reference_hits += 1
        results.append({"cpd_id": focal_id, "reference_neighbours": reference_hits})

    out_path = os.path.join(output_dir, "test_reference_neighbour_overlap.tsv")
    pd.DataFrame(results).to_csv(out_path, sep="\t", index=False)
    logger.info(f"Test compound reference neighbour stats saved to {out_path}")


def generate_network(df, output_dir, threshold):
    """
    Generate an interactive similarity network (if pyvis is installed).
    """
    logger.info("Generating interactive network (if pyvis is installed)...")
    if not PYVIS_AVAILABLE:
        logger.warning("Pyvis not available or failed to import. Skipping network visualisation.")
        return

    try:
        numeric = df.select_dtypes(include=[np.number])
        nn = NearestNeighbors(n_neighbors=6).fit(numeric)
        distances, indices = nn.kneighbors(numeric)

        g = nx.Graph()
        for i, row in enumerate(indices):
            source = df.iloc[i]["cpd_id"]
            g.add_node(source)
            for j in row[1:]:
                target = df.iloc[j]["cpd_id"]
                dist = np.linalg.norm(numeric.iloc[i] - numeric.iloc[j])
                if dist < threshold:
                    g.add_edge(source, target, weight=dist)

        net = Network(height="800px", width="100%")
        net.from_nx(g)

        output_html = os.path.join(output_dir, "compound_similarity_network.html")
        net.write_html(output_html)
        logger.info(f"Interactive network visualisation saved to '{output_html}'.")

    except Exception as e:
        logger.error(f"Failed to render HTML with pyvis (template missing?). Skipping. Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="CLIPn Latent Post-Analysis")
    parser.add_argument("--latent_csv", required=True, help="TSV with latent space and metadata")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3, help="Distance threshold for network edges")
    parser.add_argument("--test_dataset", nargs="+", help="Test dataset(s) to evaluate proximity")
    parser.add_argument("--reference_dataset", nargs="+", help="Reference dataset(s) to compare with")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Reading latent CSV: {args.latent_csv}")
    df = pd.read_csv(args.latent_csv, sep="\t")

    summarise_clusters(df, args.output_dir)

    nn_df = compute_nearest_neighbours(df)
    nn_out = os.path.join(args.output_dir, "nearest_neighbours.tsv")
    nn_df.to_csv(nn_out, sep="\t", index=False)
    logger.info(f"Nearest neighbours saved to {nn_out}")

    if args.test_dataset and args.reference_dataset:
        analyse_test_vs_reference(df, args.test_dataset, args.reference_dataset, args.output_dir)

    generate_network(df, args.output_dir, threshold=args.threshold)

    logger.info("Post-analysis complete.")


if __name__ == "__main__":
    main()
