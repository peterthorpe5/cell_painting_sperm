#!/usr/bin/env python
# coding: utf-8

"""
library of plotting modules. 
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import umap.umap_ as umap
from pathlib import Path
import numpy as np
import hdbscan
from sklearn.cluster import KMeans
from sklearn import set_config
set_config(transform_output="pandas")

from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)




def plot_umap_coloured_by_experiment(umap_df, output_file, color_map=None):
    """
    Generates a UMAP visualization coloured by experiment vs. STB.

    Parameters
    ----------
    umap_df : pd.DataFrame
        DataFrame containing UMAP coordinates with a MultiIndex (`cpd_id`, `Library`).
    output_file : str
        Path to save the UMAP plot.
    color_map : dict, optional
        A dictionary mapping dataset types (e.g., "Experiment", "STB") to colors.
        Default is {"Experiment": "red", "STB": "blue"}.

    Returns
    -------
    None
    """
    try:
        logger.info("Generating UMAP visualization highlighting Experiment vs. STB data.")

        # Default color map if none provided
        if color_map is None:
            color_map = {"Experiment": "red", "STB": "blue"}

        # Ensure 'Library' exists in MultiIndex
        if "Library" not in umap_df.index.names:
            logger.warning("Warning: 'Library' not found in MultiIndex! Attempting to use column instead.")
            if "Library" in umap_df.columns:
                umap_df = umap_df.set_index("Library")
            else:
                logger.error("Error: 'Library' column not found! Skipping UMAP experiment coloring.")
                return

        # Map colors based on 'Library'
        dataset_labels = umap_df.index.get_level_values("Library")  # Extract library info
        dataset_colors = [color_map.get(label, "gray") for label in dataset_labels]  # Assign colors

        # Create scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], s=5, alpha=0.7, c=dataset_colors)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP Visualization: Experiment (Red) vs. STB (Blue)")
        plt.tick_params(axis='both', labelsize=6)

        # Save plot
        plt.savefig(output_file, dpi=1200)
        plt.close()
        logger.info(f"UMAP visualization (experiment vs. STB) saved as '{output_file}'.")

    except Exception as e:
        logger.error(f"Error generating UMAP experiment visualization: {e}. Continuing script execution.")


def load_latent_data(latent_csv_path):
    """
    Load latent representations with MultiIndex.

    Parameters
    ----------
    latent_csv_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Latent data with MultiIndex.
    """
    return pd.read_csv(latent_csv_path, index_col=[0, 1, 2])


def plot_dendrogram(dist_df, output_file, method="ward", figsize=(14, 10), label_fontsize=2):
    """
    Plots and saves a hierarchical clustering dendrogram.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Pairwise distance matrix.
    output_file : str
        Path to save the dendrogram plot.
    method : str, optional
        Linkage method to use for clustering. Default is 'ward'.
    figsize : tuple, optional
        Size of the figure. Default is (14, 10).
    label_fontsize : int, optional
        Font size for axis tick labels. Default is 2.
    """

    condensed_dist = squareform(dist_df.values)
    linkage_matrix = linkage(condensed_dist, method=method)

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        linkage_matrix,
        labels=dist_df.index.tolist(),
        leaf_rotation=90,
        leaf_font_size=label_fontsize
    )
    ax.tick_params(axis="x", labelsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize)

    plt.tight_layout()
    plt.savefig(output_file, dpi=1200)
    plt.close()


def plot_distance_heatmap(dist_df, output_path):
    """
    Generate and save a heatmap of pairwise compound distances with clustering.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Pairwise distance matrix.
    output_path : str
        Path to save the heatmap PDF file.
    """

    linkage_matrix = linkage(squareform(dist_df.values), method="ward")

    cg = sns.clustermap(
        dist_df,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="viridis",
        figsize=(10, 10),
        xticklabels=False,
        yticklabels=False
    )

    if hasattr(cg, "fig") and hasattr(cg.fig, "suptitle"):
        cg.fig.suptitle("Pairwise Compound Distance Heatmap", y=1.02)

    cg.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close()


from sklearn.cluster import KMeans
from hdbscan import HDBSCAN

def assign_clusters(df, logger=None, n_clusters=15):
    """
    Assigns clusters to the latent space using both KMeans and HDBSCAN.

    Parameters
    ----------
    df : pd.DataFrame
        Latent space DataFrame (includes metadata + latent features).
    logger : logging.Logger, optional
        Logger for logging.
    n_clusters : int
        Number of clusters for KMeans.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns: Cluster_KMeans and Cluster_HDBSCAN.
    """
    from sklearn.preprocessing import StandardScaler

    feature_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    latent = df[feature_cols].copy()

    # Scale before clustering
    scaled = StandardScaler().fit_transform(latent)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    df["Cluster_KMeans"] = kmeans.fit_predict(scaled)
    if logger:
        logger.info("Assigned clusters using KMeans")

    # HDBSCAN
    try:
        hdb = HDBSCAN(min_cluster_size=10)
        df["Cluster_HDBSCAN"] = hdb.fit_predict(scaled)
        if logger:
            logger.info("Assigned clusters using HDBSCAN")
    except Exception as e:
        df["Cluster_HDBSCAN"] = -1
        if logger:
            logger.warning(f"HDBSCAN clustering failed: {e}")

    return df



def generate_umap(combined_latent_df, output_folder,
                  umap_plot_file, args, add_labels=False):
    """
    Generates UMAP embeddings, performs KMeans clustering, and saves the results.

    Parameters
    ----------
    combined_latent_df : pd.DataFrame
        Full DataFrame containing both metadata and latent features.

    output_folder : str
        Folder to save plots and coordinates.

    umap_plot_file : str
        Full path to save the UMAP plot (PDF).

    args : argparse.Namespace
        Parsed CLI arguments with UMAP and clustering parameters.

    add_labels : bool, optional
        Whether to add cpd_id labels to the UMAP plot (default: False).

    Returns
    -------
    pd.DataFrame
        UMAP coordinates + cluster assignments + metadata.
    """
    import matplotlib.pyplot as plt
    import umap.umap_ as umap
    from sklearn.cluster import KMeans

    logger.info(f"Generating UMAP visualization with n_neighbors={args.umap_n_neighbors}, "
                f"min_dist={args.umap_min_dist}, metric={args.umap_metric} and "
                f"{args.num_clusters} clusters.")

    metadata_cols = ["cpd_id", "cpd_type", "Library", "Dataset", "Sample"]
    metadata = combined_latent_df[metadata_cols].copy()
    latent_features = combined_latent_df.drop(columns=metadata_cols, errors="ignore")

    if latent_features.empty:
        raise ValueError("No numeric latent features available for UMAP projection.")

    umap_model = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_components=2,
        random_state=42
    )
    latent_umap = umap_model.fit_transform(latent_features)

    logger.info(f"Running KMeans clustering with {args.num_clusters} clusters.")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_umap)

    # Build output DataFrame
    umap_df = metadata.copy()
    umap_df["UMAP1"] = latent_umap[:, 0]
    umap_df["UMAP2"] = latent_umap[:, 1]
    umap_df["Cluster"] = cluster_labels

    # Save coordinate table
    coords_file = os.path.join(output_folder, "clipn_umap_coordinates.tsv")
    umap_df.to_csv(coords_file, sep="\t", index=False)
    logger.info(f"UMAP coordinates saved to '{coords_file}'.")

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        umap_df["UMAP1"],
        umap_df["UMAP2"],
        s=3,
        alpha=0.5,
        c=cluster_labels,
        cmap="tab10"
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("CLIPn UMAP: Clustered")
    plt.colorbar(scatter, label="Cluster ID")

    if add_labels:
        logger.info("Adding `cpd_id` labels to UMAP plot.")
        for _, row in umap_df.iterrows():
            plt.text(row["UMAP1"], row["UMAP2"], str(row["cpd_id"]), fontsize=1, alpha=0.6)
        umap_plot_file = umap_plot_file.replace(".pdf", "_labeled.pdf")

    plt.savefig(umap_plot_file, dpi=1200)
    plt.close()
    logger.info(f"UMAP plot saved to '{umap_plot_file}'.")

    return umap_df
