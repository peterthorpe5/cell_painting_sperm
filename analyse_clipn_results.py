#!/usr/bin/env python3
"""
CLIPn Post-Analysis Script
---------------------------
This script performs post-analysis on CLIPn latent representations.

Inputs
------
--latent_csv : Path to a TSV file with latent space coordinates, compound metadata (cpd_id, cpd_type, Dataset, etc.)
--output_dir : Directory to save all outputs (plots, summaries, network HTML)

Outputs
-------
- Cluster summary (per cluster, per dataset, per cpd_type)
- Nearest neighbours table
- Interactive compound similarity network (optional)
- Full logging throughout the run

Author: Auto-generated for custom CLIPn analysis
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from pyvis.network import Network

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_nearest_neighbours(df, n_neighbours=5):
    """Compute nearest neighbours from numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    nn = NearestNeighbors(n_neighbors=n_neighbours + 1, metric="euclidean").fit(numeric)
    distances, indices = nn.kneighbors(numeric)

    neighbour_table = []
    for i, row in enumerate(indices):
        focal = df.iloc[i]["cpd_id"]
        for j in row[1:]:  # skip self
            neighbour = df.iloc[j]["cpd_id"]
            dist = np.linalg.norm(numeric.iloc[i] - numeric.iloc[j])
            neighbour_table.append({
                "cpd_id": focal,
                "neighbour_id": neighbour,
                "distance": dist
            })

    return pd.DataFrame(neighbour_table)


def summarise_clusters(df, output_dir):
    """Create cluster summary grouped by cpd_type and dataset."""
    if "Cluster" not in df.columns:
        logger.warning("No Cluster column found in input. Skipping summary.")
        return

    summary = df.groupby(["Cluster", "cpd_type", "Dataset"]).agg({
        "cpd_id": "count"
    }).reset_index()

    summary.to_csv(os.path.join(output_dir, "cluster_summary_by_type_and_dataset.tsv"),
                   sep="\t", index=False)
    logger.info("Cluster summary saved.")


def generate_similarity_network(df, output_html, threshold=0.3):
    """Generate interactive compound network based on nearest neighbour distances."""
    numeric = df.select_dtypes(include=[np.number])
    nn = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(numeric)
    distances, indices = nn.kneighbors(numeric)

    g = nx.Graph()
    for idx, neighbours in enumerate(indices):
        source = df.iloc[idx]["cpd_id"]
        g.add_node(source)
        for j in neighbours[1:]:
            target = df.iloc[j]["cpd_id"]
            dist = np.linalg.norm(numeric.iloc[idx] - numeric.iloc[j])
            if dist < threshold:
                g.add_edge(source, target, weight=dist)

    net = Network(height="800px", width="100%", notebook=False)
    net.from_nx(g)
    net.show(output_html)
    logger.info(f"Interactive network visualisation saved to '{output_html}'.")


def main():
    parser = argparse.ArgumentParser(description="Post-analysis for CLIPn latent space.")
    parser.add_argument("--latent_csv", required=True,
                        help="TSV with latent coordinates and compound metadata.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save outputs.")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Distance threshold for network edges.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Reading latent CSV: {args.latent_csv}")
    df = pd.read_csv(args.latent_csv, sep="\t")

    logger.info("Performing cluster summary...")
    summarise_clusters(df, args.output_dir)

    logger.info("Computing nearest neighbours...")
    nn_df = compute_nearest_neighbours(df)
    nn_file = os.path.join(args.output_dir, "nearest_neighbours.tsv")
    nn_df.to_csv(nn_file, sep="\t", index=False)
    logger.info(f"Nearest neighbours saved to {nn_file}")

    logger.info("Generating interactive network...")
    network_html = os.path.join(args.output_dir, "compound_similarity_network.html")
    generate_similarity_network(df, network_html, threshold=args.threshold)

    logger.info("Post-analysis complete.")


if __name__ == "__main__":
    main()
