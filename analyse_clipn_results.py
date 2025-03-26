#!/usr/bin/env python3
# coding: utf-8
"""
Analyse CLIPn Latent Space Representations
-------------------------------------------
This script loads latent representations from a `.npz` file (CLIPn output)
and performs post-analysis including:
    - UMAP visualisation
    - PCA scatter plots
    - Recall@k calculation
    - Confusion matrix generation
    - Cluster summary table of cpd_id and cpd_type per cluster

Outputs are saved to a specified output directory with filenames indicating the analysis type.

Usage:
    python analyse_clipn_latent.py \
        --latent path/to/CLIPn_latent_representations.npz \
        --labels path/to/label_mapping.csv \
        --output analysis_results

Command-Line Arguments:
------------------------
    --latent         : Path to latent representation NPZ file.
    --labels         : Path to CSV file with label mappings.
    --output         : Output directory for plots and results.

All analyses run by default.

Logging:
--------
The script logs progress and saves outputs with informative filenames in the specified folder.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def umap_scatter_save(Z: dict, y: dict, path: str) -> None:
    """
    Generate and save UMAP visualisation coloured by label and dataset.

    Parameters
    ----------
    Z : dict
        Dictionary of latent representations, keyed by dataset index.
    y : dict
        Dictionary of labels corresponding to Z, keyed by dataset index.
    path : str
        Path prefix for saving the output UMAP images (e.g., 'output/umap').
    """


    keys = list(Z.keys())
    latent = np.vstack([Z[k] for k in keys])
    labels = np.concatenate([y[k] for k in keys])
    datasets = np.concatenate([[k] * len(y[k]) for k in keys])

    umap_coords = UMAP().fit_transform(latent)
    df = pd.DataFrame({
        "umap-1": umap_coords[:, 0],
        "umap-2": umap_coords[:, 1],
        "label": labels,
        "dataset": datasets
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for label in sorted(df['label'].unique()):
        axes[0].scatter(df[df.label == label]["umap-1"], df[df.label == label]["umap-2"], s=1, label=label)
    axes[0].set_title("UMAP: coloured by label")

    for ds in sorted(df['dataset'].unique()):
        axes[1].scatter(df[df.dataset == ds]["umap-1"], df[df.dataset == ds]["umap-2"], s=1, label=ds)
    axes[1].set_title("UMAP: coloured by dataset")

    for ax in axes:
        ax.axis("square")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path + "_labels.png", dpi=1200)
    plt.savefig(path + "_datasets.png", dpi=1200)
    plt.close()


def recall_k(Z_train: dict, Z_test: dict, y_train: dict, y_test: dict, k: list = [1, 5, 10, 20, 50]) -> np.ndarray:
    """
    Compute recall@k for latent embeddings.

    Parameters
    ----------
    Z_train : dict
        Latent representations for training datasets.
    Z_test : dict
        Latent representations for testing datasets.
    y_train : dict
        Labels for training datasets.
    y_test : dict
        Labels for testing datasets.
    k : list, optional
        List of k values for recall computation.

    Returns
    -------
    np.ndarray
        Recall values at specified k thresholds.
    """
    Z_train_mat = np.vstack(list(Z_train.values()))
    y_train_vec = np.concatenate(list(y_train.values()))
    Z_test_mat = np.vstack(list(Z_test.values()))
    y_test_vec = np.concatenate(list(y_test.values()))

    sim = Z_test_mat @ Z_train_mat.T
    rank = np.argsort(-sim, axis=1)
    label_mat_train = np.vstack([y_train_vec] * Z_test_mat.shape[0])
    label_rank = np.take_along_axis(label_mat_train, rank, axis=1)

    compare = label_rank == y_test_vec[:, None]
    recall = np.array([np.any(compare[:, :i], axis=1).mean() for i in k])
    return recall


def confusion_matrix(Z_train: dict, y_train: dict, Z_test: dict, y_test: dict) -> np.ndarray:
    """
    Compute a confusion matrix using top-1 nearest neighbour classification.

    Parameters
    ----------
    Z_train : dict
        Latent space vectors from training data.
    y_train : dict
        Corresponding labels.
    Z_test : dict
        Latent space vectors from test data.
    y_test : dict
        Corresponding labels.

    Returns
    -------
    np.ndarray
        Confusion matrix (label x predicted label).
    """
    Z_train_mat = np.vstack(list(Z_train.values()))
    y_train_vec = np.concatenate(list(y_train.values()))
    Z_test_mat = np.vstack(list(Z_test.values()))
    y_test_vec = np.concatenate(list(y_test.values()))

    sim = Z_test_mat @ Z_train_mat.T
    rank = np.argsort(-sim, axis=1)
    pred_labels = y_train_vec[rank[:, 0]]

    max_label = max(y_test_vec.max(), y_train_vec.max()) + 1
    cm = np.zeros((max_label, max_label), dtype=int)
    for true, pred in zip(y_test_vec, pred_labels):
        cm[true, pred] += 1
    return cm / cm.sum(axis=1, keepdims=True)



def load_latent_and_labels(latent_path: str, labels_path: str) -> tuple:
    """Load CLIPn latent representations and corresponding labels.

    Parameters
    ----------
    latent_path : str
        Path to .npz file containing latent representations.
    labels_path : str
        Path to .csv file containing label mappings.

    Returns
    -------
    tuple
        Latent representations (dict), labels (dict)
    """
    logger.info(f"Loading latent representations from: {latent_path}")
    latent_npz = np.load(latent_path, allow_pickle=True)
    Z = {int(k): np.array(v) for k, v in latent_npz.items()}

    logger.info(f"Loading label mappings from: {labels_path}")
    label_df = pd.read_csv(labels_path, index_col=0)
    y = {int(k): label_df[str(k)].dropna().astype(int).values for k in Z}

    return Z, y

def save_recall_at_k(Z: dict, y: dict, output_path: str) -> None:
    """Compute and save recall@k values to CSV.

    Parameters
    ----------
    Z : dict
        Dictionary of latent representations.
    y : dict
        Dictionary of corresponding labels.
    output_path : str
        Output folder to save the CSV.
    """
    logger.info("Computing recall@k")
    recall_values = recall_k(Z, Z, y, y)
    k = [1, 5, 10, 20, 50]
    recall_df = pd.DataFrame({'k': k, 'recall': recall_values})
    recall_file = os.path.join(output_path, "recall_at_k.csv")
    recall_df.to_csv(recall_file, index=False)
    logger.info(f"Saved recall@k to {recall_file}")

def save_confusion_matrix(Z: dict, y: dict, output_path: str) -> None:
    """Compute and save confusion matrix.

    Parameters
    ----------
    Z : dict
        Dictionary of latent representations.
    y : dict
        Dictionary of corresponding labels.
    output_path : str
        Output folder to save the matrix.
    """
    logger.info("Computing confusion matrix")
    cm = confusion_matrix(Z, y, Z, y)
    cm_df = pd.DataFrame(cm)
    cm_file = os.path.join(output_path, "confusion_matrix.csv")
    cm_df.to_csv(cm_file, index=False)
    logger.info(f"Saved confusion matrix to {cm_file}")

def save_cluster_summary(Z: dict, y: dict, output_path: str) -> None:
    """Generate a summary of latent space clusters by compound ID and type.

    Parameters
    ----------
    Z : dict
        Latent representations.
    y : dict
        Encoded labels for each representation.
    output_path : str
        Path to save the summary CSV file.
    """
    logger.info("Generating cluster summary by label")
    df_list = []
    for key, z_array in Z.items():
        cluster_ids = y[key]
        temp_df = pd.DataFrame(z_array)
        temp_df['Cluster'] = cluster_ids
        temp_df['Dataset'] = key
        df_list.append(temp_df)

    combined_df = pd.concat(df_list)
    summary = (
        combined_df.groupby("Cluster")
        .agg(Dataset_Count=("Dataset", lambda x: len(set(x))))
        .reset_index()
    )
    summary_path = os.path.join(output_path, "clipn_cluster_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Cluster summary saved to {summary_path}")

def generate_visualisations(Z: dict, y: dict, output_path: str) -> None:
    """Generate and save UMAP and PCA visualisations.

    Parameters
    ----------
    Z : dict
        Latent representations.
    y : dict
        Corresponding labels.
    output_path : str
        Folder where plots are saved.
    """
    logger.info("Generating UMAP plots")
    umap_scatter_save(Z, y, os.path.join(output_path, "umap"))

    logger.info("Generating PCA scatter plot")
    pca_scatter(Z, y)

def main():
    parser = argparse.ArgumentParser(description="Analyse CLIPn latent space outputs.")
    parser.add_argument("--latent", required=True, help="Path to .npz file with latent outputs.")
    parser.add_argument("--labels", required=True, help="Path to CSV file with label mappings.")
    parser.add_argument("--output", required=True, help="Output directory for results.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    Z, y = load_latent_and_labels(args.latent, args.labels)

    generate_visualisations(Z, y, args.output)
    save_recall_at_k(Z, y, args.output)
    save_confusion_matrix(Z, y, args.output)
    save_cluster_summary(Z, y, args.output)

if __name__ == "__main__":
    main()
