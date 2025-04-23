#!/usr/bin/env python3
"""
Visualise CLIPn Outputs
------------------------
This script loads CLIPn latent space representations with compound IDs,
and generates post-analysis plots and summaries:
    - UMAP visualisation (coloured by cluster and dataset)
    - Pairwise distance heatmap
    - Dendrogram
    - Compound similarity summary
    - UMAP cluster summary

Inputs:
    --latent_csv  : TSV file with latent features, compound metadata, and Dataset/Sample indices.
    --plots       : Output directory for plots and summary files.

Output files:
    - clipn_UMAP.pdf
    - clipn_UMAP_labeled.pdf
    - compound_distance_heatmap.pdf
    - compound_clustering_dendrogram.pdf
    - compound_similarity_summary.tsv
    - pairwise_compound_distances.tsv
    - umap_cluster_summary.tsv
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import set_config
set_config(transform_output="pandas")
from cell_painting.plot import (
    plot_distance_heatmap,
    plot_dendrogram,
    generate_umap,
    assign_clusters
)
from cell_painting.process_data import (
    generate_similarity_summary,
    compute_pairwise_distances
)

try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_and_reorder_latent_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns and reorder key metadata columns first.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame from CLIPn latent space TSV.

    Returns
    -------
    pd.DataFrame
        Cleaned and reordered DataFrame.
    """
    columns_to_drop = [col for col in df.columns if col.endswith("_x") or col.endswith("_y") or col == "index"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    meta_cols = ["cpd_id", "cpd_type", "Library", "Dataset", "Sample"]
    meta_cols = [col for col in meta_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in meta_cols and pd.api.types.is_numeric_dtype(df[col])]
    df = df[meta_cols + other_cols]

    return df

def main(args):
    os.makedirs(args.plots, exist_ok=True)

    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")

    logger.info(f"Reading latent TSV: {args.latent_csv}")
    df = pd.read_csv(args.latent_csv, sep="\t")
    df = clean_and_reorder_latent_df(df)

    

    # Assign clusters if none exist
    if not any(col for col in df.columns if col.startswith("Cluster_")):
        logger.info("No cluster assignments found. Running KMeans and HDBSCAN...")
        df = assign_clusters(df, logger=logger)

    for cluster_col in ["Cluster_KMeans", "Cluster_HDBSCAN"]:
        if cluster_col in df.columns:
            logger.info(f"Generating UMAP for {cluster_col}")
            output_file = os.path.join(args.plots, f"clipn_UMAP_{cluster_col}.pdf")
            try:
                generate_umap(df.copy(), args.plots, output_file, args=args, add_labels=True, colour_by=cluster_col)
            except Exception as e:
                logger.warning(f"UMAP failed for {cluster_col}: {e}")



    for cluster_col in ["Cluster_KMeans", "Cluster_HDBSCAN"]:
        if cluster_col in df.columns:
            summary_path = os.path.join(args.plots, f"umap_summary_{cluster_col}.tsv")
            df.reset_index().groupby(cluster_col).agg({
                "cpd_type": lambda x: sorted(set(x)),
                "cpd_id": lambda x: sorted(set(x))
            }).to_csv(summary_path, sep="\t")
            logger.info(f"Saved UMAP cluster summary to: {summary_path}")

    # Generate UMAP without labels
    logger.info("Generating UMAP visualisation")
    umap_file = os.path.join(args.plots, "clipn_UMAP.pdf")
    try:
        _ = generate_umap(df.copy(), args.plots, umap_file, args=args, add_labels=False)
    except ValueError as e:
        logger.error(f"UMAP failed: {e}")
        return

    # Generate UMAP with labels
    logger.info("Generating UMAP visualisation with labels")
    umap_file_labeled = os.path.join(args.plots, "clipn_UMAP_labeled.pdf")
    try:
        umap_df = generate_umap(df.copy(), args.plots, umap_file_labeled, args=args, add_labels=True)
    except ValueError as e:
        logger.error(f"Labeled UMAP failed: {e}")
        return

    # Pairwise distances
    logger.info("Computing pairwise distances")
    numeric_df = df.select_dtypes(include=[np.number])
    dist_df = compute_pairwise_distances(numeric_df)
    dist_csv = os.path.join(args.plots, "pairwise_compound_distances.tsv")
    dist_df.to_csv(dist_csv, sep="\t")

    # Heatmap and dendrogram
    logger.info("Generating heatmap and dendrogram")
    logger.info("this takes a long time ...")

    if args.include_heatmap:
        logger.info("Generating heatmap and dendrogram (may be slow)...")
        plot_dendrogram(dist_df, os.path.join(args.plots, "compound_clustering_dendrogram.pdf"))
        plot_distance_heatmap(dist_df, os.path.join(args.plots, "compound_distance_heatmap.pdf"))
    else:
        logger.info("Skipping heatmap and dendrogram (set --include_heatmap to enable)")

    # Similarity summary
    logger.info("Generating similarity summary")
    summary_df = generate_similarity_summary(dist_df)
    summary_df.to_csv(os.path.join(args.plots, "compound_similarity_summary.tsv"), sep="\t", index=False)

    # UMAP cluster summary
    if "Cluster" in umap_df.columns:
        logger.info("Generating UMAP cluster summary")
        umap_summary_csv = os.path.join(args.plots, "umap_cluster_summary.tsv")
        umap_df.reset_index().groupby("Cluster").agg({
            "cpd_type": lambda x: sorted(set(x)),
            "cpd_id": lambda x: sorted(set(x))
        }).to_csv(umap_summary_csv, sep="\t")

    logger.info("All plots and summaries generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise CLIPn outputs.")
    parser.add_argument("--latent_csv", 
                        required=True, 
                        help="TSV with latent space + metadata from CLIPn.")
    parser.add_argument("--plots", 
                        required=True, 
                        help="Directory to store visualisation outputs.")
    parser.add_argument("--umap_n_neighbors", 
                        type=int, 
                        default=15,
                        help="Number of neighbours to use for UMAP (default: 15).")
    parser.add_argument("--umap_min_dist", 
                        type=float, 
                        default=0.25,
                        help="Minimum distance parameter for UMAP (default: 0.25).")
    parser.add_argument("--umap_metric", 
                        type=str, 
                        default="euclidean",
                        help="Distance metric for UMAP (e.g., euclidean, cosine, manhattan).")
    parser.add_argument("--num_clusters", 
                        type=int, 
                        default=15,
                        help="Number of clusters for KMeans (default: 15).")
    parser.add_argument("--include_heatmap", action="store_true",
                        help="Include heatmap and dendrogram plots (slow).")


    args = parser.parse_args()
    main(args)
