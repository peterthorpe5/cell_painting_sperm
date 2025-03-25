#!/usr/bin/env python3
"""
Loads CLIPn results and generates visualisations, including UMAP plots,
heatmaps, dendrograms, and summary tables.

Inputs:
    - CLIPn_latent_representations.npz
    - CLIPn_latent_representations_with_cpd_id.csv
    - dataset_index_mapping.csv
    - label_mappings.csv

Outputs:
    - UMAP visualisations (PDF)
    - Compound distance heatmap (PDF)
    - Clustering dendrogram (PDF)
    - Compound similarity summary (CSV)
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from plot import (
    plot_distance_heatmap, 
    plot_dendrogram,
    generate_umap, 
    plot_umap_coloured_by_experiment,
    load_latent_data
)
from process_data import (
    generate_similarity_summary, compute_pairwise_distances
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    os.makedirs(args.plots, exist_ok=True)
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Using Logfile: {log_filename}")
    logger.info(f"Logging initialized at {time.asctime()}")

    combined_latent_df = load_latent_data(os.path.join(args.input, "CLIPn_latent_representations_with_cpd_id.csv"))

    # UMAP Visualisation
    umap_file = os.path.join(args.plots, "clipn_UMAP.pdf")
    umap_df = generate_umap(combined_latent_df, args.input, umap_file, args, add_labels=True)

    # Pairwise distances
    numeric_df = combined_latent_df.select_dtypes(include=[np.number])
    dist_df = compute_pairwise_distances(numeric_df)

    # Save distances
    distance_csv = os.path.join(args.input, "pairwise_compound_distances.csv")
    dist_df.to_csv(distance_csv)

    # Heatmap & dendrogram
    heatmap_pdf = os.path.join(args.plots, "compound_distance_heatmap.pdf")
    dendrogram_pdf = os.path.join(args.plots, "compound_clustering_dendrogram.pdf")

    plot_distance_heatmap(dist_df, heatmap_pdf)
    plot_dendrogram(dist_df, dendrogram_pdf)

    # Similarity summary
    summary_df = generate_similarity_summary(dist_df)
    summary_csv = os.path.join(args.input, "compound_similarity_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # UMAP summary
    umap_summary_csv = os.path.join(args.input, "umap_cluster_summary.csv")
    umap_df.reset_index().groupby("Cluster").agg({
        "cpd_type": lambda x: sorted(set(x)),
        "cpd_id": lambda x: sorted(set(x))
    }).to_csv(umap_summary_csv)

    logger.info("All plots and summaries generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise CLIPn outputs.")
    parser.add_argument("--input", required=True, help="Folder with CLIPn outputs.")
    parser.add_argument("--plots", required=True, help="Output folder for plots.")

    args = parser.parse_args()
    main(args)
