
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
- Interactive compound similarity network (if successful)
- Test-to-reference neighbour analysis (optional)
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
    """Create cluster summary grouped by Cluster, cpd_type, and Dataset."""
    if "Cluster" not in df.columns:
        logger.warning("No Cluster column found in input. Skipping summary.")
        return

    summary = df.groupby(["Cluster", "cpd_type", "Dataset"]).agg({
        "cpd_id": "count"
    }).reset_index()

    summary.to_csv(os.path.join(output_dir, "cluster_summary_by_type_and_dataset.tsv"),
                   sep="\t", index=False)
    logger.info("Cluster summary saved.")


def compute_test_to_reference_neighbour_overlap(df, test_label, reference_label, n_neighbours=5):
    """
    For each test compound, count how many of its nearest neighbours are from the reference dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with latent space columns and 'Dataset' column.
    test_label : str
        Label identifying test dataset (e.g. 'SelleckChem').
    reference_label : str
        Label identifying reference dataset (e.g. 'stb').
    n_neighbours : int
        Number of neighbours to check for each test compound.

    Returns
    -------
    pd.DataFrame
        Test compounds with counts of reference neighbours.
    """
    numeric = df.select_dtypes(include=[np.number])
    nn = NearestNeighbors(n_neighbors=n_neighbours + 1, metric="euclidean").fit(numeric)
    distances, indices = nn.kneighbors(numeric)

    test_mask = df["Dataset"] == test_label
    test_indices = df[test_mask].index

    results = []
    for idx in test_indices:
        neighbour_ids = indices[idx][1:]  # exclude self
        neighbour_datasets = df.iloc[neighbour_ids]["Dataset"].values
        reference_count = np.sum(neighbour_datasets == reference_label)
        results.append({
            "cpd_id": df.iloc[idx]["cpd_id"],
            "test_dataset": test_label,
            "reference_neighbours": reference_count,
            "total_checked": n_neighbours
        })

    return pd.DataFrame(results)


def generate_similarity_network(df, output_html, threshold=0.3):
    """Generate interactive compound network based on nearest neighbour distances."""
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error("pyvis is not installed. Skipping network generation.")
        return

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

    try:
        net.show(output_html)
        logger.info(f"Interactive network visualisation saved to '{output_html}'.")
    except AttributeError:
        logger.error("Failed to render HTML with pyvis (template missing). Skipping.")


def main():
    parser = argparse.ArgumentParser(description="Post-analysis for CLIPn latent space.")
    parser.add_argument("--latent_csv", required=True,
                        help="TSV with latent coordinates and compound metadata.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save outputs.")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Distance threshold for network edges.")
    parser.add_argument("--test_dataset", type=str, default="SelleckChem",
                        help="Label of test dataset in Dataset column.")
    parser.add_argument("--reference_dataset", type=str, default="stb",
                        help="Label of reference dataset in Dataset column.")
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

    logger.info("Evaluating test compound proximity to reference dataset...")
    overlap_df = compute_test_to_reference_neighbour_overlap(
        df,
        test_label=args.test_dataset,
        reference_label=args.reference_dataset,
        n_neighbours=5
    )
    overlap_file = os.path.join(args.output_dir, "test_reference_neighbour_overlap.tsv")
    overlap_df.to_csv(overlap_file, sep="\t", index=False)
    logger.info(f"Test compound reference neighbour stats saved to {overlap_file}")

    logger.info("Generating interactive network (if pyvis is installed)...")
    network_html = os.path.join(args.output_dir, "compound_similarity_network.html")
    generate_similarity_network(df, network_html, threshold=args.threshold)

    logger.info("Post-analysis complete.")


if __name__ == "__main__":
    main()
