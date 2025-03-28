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
from sklearn.cluster import KMeans
from sklearn import set_config
set_config(transform_output="pandas")

from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


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
        Size of the figure. Default is (12, 8).
    label_fontsize : int, optional
        Font size for axis tick labels. Default is 4.
    """
    # Compute linkage
    linkage_matrix = linkage(dist_df, method=method)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(linkage_matrix, labels=dist_df.index.tolist(), leaf_rotation=90, leaf_font_size=label_fontsize)
    ax.tick_params(axis="x", labelsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize)

    plt.tight_layout()
    plt.savefig(output_file, dpi=1200)
    plt.close()



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


def plot_distance_heatmap(dist_df, output_path):
    """
    Generate and save a heatmap of pairwise compound distances.

    Parameters:
    -----------
    dist_df : pd.DataFrame
        Pairwise distance matrix with `cpd_id` as row and column labels.
    output_path : str
        Path to save the heatmap PDF file.
    """
    plt.figure(figsize=(14, 12))
    htmap = sns.clustermap(dist_df, cmap="viridis", method="ward",
                           figsize=(12, 10),
                           xticklabels=True,
                           yticklabels=True)

    # Rotate labels for better readability
    plt.setp(htmap.ax_heatmap.get_xticklabels(), rotation=90, fontsize=2)
    plt.setp(htmap.ax_heatmap.get_yticklabels(), rotation=0, fontsize=2)

    plt.title("Pairwise Distance Heatmap of Compounds")
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close()


def generate_umap(combined_latent_df, output_folder, umap_plot_file, args, 
                  n_neighbors=15, num_clusters=10, add_labels=False
):
    """
    Generates UMAP embeddings, performs KMeans clustering, and saves the results.

    Parameters
    ----------
    combined_latent_df : pd.DataFrame
        Latent representations indexed by (cpd_id, Library, cpd_type).
    
    output_folder : str
        Path to save the UMAP outputs.

    umap_plot_file : str
        Full file path to save the UMAP plot (PDF).

    args : argparse.Namespace
        Parsed command-line arguments containing hyperparameters (latent_dim, lr, epoch).

    n_neighbors : int, optional
        Number of neighbors for UMAP (default: 15).
    
    num_clusters : int, optional
        Number of clusters for KMeans clustering (default: 10).

    add_labels : bool, optional
        Whether to label points with `cpd_id` on the UMAP plot (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame containing UMAP coordinates and cluster labels, indexed by (cpd_id, Library, cpd_type).
    """

    logger.info(f"Generating UMAP visualization with n_neighbors={n_neighbors} and {num_clusters} clusters.")

    # Perform UMAP dimensionality reduction
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    numeric_df = combined_latent_df.select_dtypes(include=[np.number])
    latent_umap = umap_model.fit_transform(numeric_df)




    latent_umap = umap_model.fit_transform(combined_latent_df.drop(columns=["dataset"], errors="ignore"))


    # Create DataFrame with MultiIndex
    umap_df = pd.DataFrame(latent_umap, columns=["UMAP1", "UMAP2"], index=combined_latent_df.index)

    # Perform clustering
    logger.info(f"Running KMeans clustering with {num_clusters} clusters.")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_umap)
    
    # Add cluster labels to DataFrame
    umap_df["Cluster"] = cluster_labels

    # Save UMAP results
    umap_file = os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_UMAP.csv")
    umap_df.to_csv(umap_file)
    logger.info(f"UMAP coordinates saved to '{umap_file}'.")

    # Generate and save UMAP plot with cluster colors
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], alpha=0.7, s=5, c=cluster_labels, cmap="tab10")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("CLIPn UMAP Visualization (coloured by Cluster)")
    plt.colorbar(scatter, label="Cluster ID")

    # Add `cpd_id` labels if enabled
    if add_labels:
        logger.info("Adding `cpd_id` labels to UMAP plot.")
        for (cpd_id, _), (x, y) in zip(umap_df.index, latent_umap):
            plt.text(x, y, str(cpd_id), fontsize=2, alpha=0.7)

    # Adjust filename if labels are included
    if add_labels:
        umap_plot_file = umap_plot_file.replace(".pdf", "_labeled.pdf")

    # Save plot
    plt.savefig(umap_plot_file, dpi=1200)
    plt.close()

    logger.info(f"UMAP visualization with clusters saved to '{umap_plot_file}'.")

    return umap_df
