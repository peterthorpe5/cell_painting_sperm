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
import plotly.express as px
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

    cg.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def assign_clusters(df, logger=None, num_clusters=15):
    """
    Assign KMeans and HDBSCAN clusters to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latent features and metadata.
    logger : logging.Logger, optional
        Logger for debug information.
    num_clusters : int, optional
        Number of clusters for KMeans (default: 15).

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'Cluster_KMeans' and 'Cluster_HDBSCAN' columns.
    """
    from sklearn.cluster import KMeans
    from hdbscan import HDBSCAN

    numeric_df = df.select_dtypes(include=[float, int])

    # KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster_KMeans"] = kmeans.fit_predict(numeric_df)
    if logger:
        logger.info("Assigned clusters using KMeans")

    # HDBSCAN
    if HDBSCAN is not None:
        hdb = HDBSCAN(min_cluster_size=5, prediction_data=True)
        df["Cluster_HDBSCAN"] = hdb.fit_predict(numeric_df)
        if logger:
            logger.info("Assigned clusters using HDBSCAN")
    else:
        if logger:
            logger.warning("HDBSCAN not available. Skipping HDBSCAN clustering.")

    return df



def generate_umap(df, output_dir, output_file, args=None, add_labels=False, colour_by="cpd_type"):
    """
    Generate and save UMAP plots (matplotlib and optional interactive Plotly).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric features and metadata.
    output_dir : str
        Directory to save the plot.
    output_file : str
        Path to the output plot file.
    args : Namespace, optional
        Parsed CLI arguments (for UMAP and interactive settings).
    add_labels : bool, optional
        Whether to add compound labels to the plot.
    colour_by : str, optional
        Column to colour points by (e.g., 'cpd_type', 'Library', 'Cluster').

    Returns
    -------
    pd.DataFrame
        DataFrame with UMAP coordinates added.
    """
    if args is None:
        n_neighbors = 15
        min_dist = 0.25
        metric = "euclidean"
        compound_file = None
    else:
        n_neighbors = args.umap_n_neighbors
        min_dist = args.umap_min_dist
        metric = args.umap_metric
        compound_file = getattr(args, "compound_metadata", None)

    # Add compound metadata if available
    if compound_file and os.path.isfile(compound_file):
        try:
            meta_df = pd.read_csv(compound_file, sep="\t")
            meta_dedup = meta_df.drop_duplicates(subset="cpd_id")
            df = pd.merge(df, meta_dedup[["cpd_id", "published_phenotypes", "publish own other", "published_target"]],
                          on="cpd_id", how="left")
        except Exception as e:
            logging.warning(f"Failed to merge compound metadata: {e}")

    # UMAP projection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embedding = reducer.fit_transform(df[numeric_cols])
    df["UMAP1"] = embedding[:, 0]
    df["UMAP2"] = embedding[:, 1]

    # Static plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if colour_by in df.columns:
        sns.scatterplot(x="UMAP1", y="UMAP2", hue=colour_by, data=df, ax=ax, s=10, linewidth=0, alpha=0.8)
        ax.legend(loc="best", fontsize="small", markerscale=1, frameon=False)
    else:
        sns.scatterplot(x="UMAP1", y="UMAP2", data=df, ax=ax, s=10, linewidth=0, alpha=0.8)

    if add_labels and "cpd_id" in df.columns:
        for _, row in df.iterrows():
            ax.text(row["UMAP1"], row["UMAP2"], str(row["cpd_id"]), fontsize=3, alpha=0.7)

    ax.set_title(f"CLIPn UMAP ({metric})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=1200)
    plt.close()

    # Optional interactive Plotly plot
    if args is not None and getattr(args, "interactive", False):
        hover_cols = [col for col in [
            "cpd_id", "cpd_type", "Library", "Dataset", colour_by,
            "published_phenotypes", "publish own other", "published_target"
        ] if col in df.columns]

        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color=colour_by if colour_by in df.columns else None,
            hover_data=hover_cols,
            title=f"CLIPn UMAP ({metric}, coloured by {colour_by})",
            template="plotly_white"
        )
        html_name = os.path.splitext(os.path.basename(output_file))[0] + ".html"
        html_path = os.path.join(output_dir, html_name)
        fig.write_html(html_path)
        logging.info(f"Saved interactive Plotly UMAP to: {html_path}")

    return df

