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
    python analyse_clipn_results.py \
        --latent path/to/CLIPn_latent_representations.npz \
        --labels path/to/label_mapping.csv \
        --output analysis_results \
        [--metadata path/to/CLIPn_latent_representations_with_cpd_id.csv]

Command-Line Arguments:
------------------------
    --latent         : Path to latent representation NPZ file.
    --labels         : Path to CSV file with label mappings.
    --output         : Output directory for plots and results.
    --metadata       : Optional path to metadata CSV with MultiIndex (cpd_id, Library, cpd_type).

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
    from umap import UMAP
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    latent_list = []
    label_list = []
    dataset_list = []

    for key in Z:
        if key not in y:
            logger.warning(f"Key {key} in Z but not found in y. Skipping.")
            continue

        z_array = Z[key]
        y_array = y[key]

        if len(z_array) != len(y_array):
            logger.warning(f"Length mismatch for key {key}: latent={len(z_array)}, labels={len(y_array)}. Skipping.")
            continue

        latent_list.append(z_array)
        label_list.append(y_array)
        dataset_list.append(np.full(len(y_array), key))

    if not latent_list:
        logger.error("No valid dataset-label pairs found for UMAP plotting. Exiting.")
        return

    latent = np.vstack(latent_list)
    labels = np.concatenate(label_list)
    datasets = np.concatenate(dataset_list)

    logger.info(f"Running UMAP on {latent.shape[0]} samples from {len(latent_list)} datasets.")

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
    plt.savefig(path + "_labels.pdf", dpi=1200)
    plt.savefig(path + "_datasets.pdf", dpi=1200)
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

    # Validate consistency
    for k in Z:
        if len(Z[k]) != len(y[k]):
            raise ValueError(f"Mismatch in latent and label sizes for dataset {k}: {len(Z[k])} vs {len(y[k])}")

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

def save_cluster_summary(Z: dict, y: dict, metadata_path: str, output_path: str) -> None:
    """Generate a summary of latent space clusters by compound ID and type.

    Parameters
    ----------
    Z : dict
        Latent representations.
    y : dict
        Encoded labels for each representation.
    metadata_path : str
        Path to metadata file containing MultiIndex with cpd_id and cpd_type.
    output_path : str
        Path to save the summary CSV file.
    """
    logger.info("Generating cluster summary with compound metadata")
    df_list = []
    for key, z_array in Z.items():
        cluster_ids = y[key]
        temp_df = pd.DataFrame(z_array)
        temp_df['Cluster'] = cluster_ids
        temp_df['Dataset'] = key
        df_list.append(temp_df)

    combined_df = pd.concat(df_list).reset_index(drop=True)

    if metadata_path and os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path, index_col=[0, 1, 2])
        metadata_df = metadata_df.reset_index(drop=False)
        metadata_df = metadata_df.iloc[:len(combined_df)]  # truncate to match if needed
        combined_df = pd.concat([combined_df, metadata_df], axis=1)

        summary = (
            combined_df.groupby("Cluster")
            .agg({
                "cpd_id": lambda x: list(sorted(set(x))),
                "cpd_type": lambda x: list(sorted(set(x))),
                "Dataset": lambda x: list(sorted(set(x)))
            })
            .rename(columns={
                "cpd_id": "cpd_ids_in_cluster",
                "cpd_type": "cpd_types_in_cluster",
                "Dataset": "datasets_present"
            })
            .reset_index()
        )
    else:
        logger.warning("Metadata not found or not provided. Generating summary without compound info.")
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



def generate_label_mapping_from_metadata(metadata_csv: str, output_path: str) -> str:
    """Generate a label mapping CSV from MultiIndex metadata.

    Parameters
    ----------
    metadata_csv : str
        Path to CSV file containing metadata with a MultiIndex.
    output_path : str
        Output directory where the label mapping CSV will be saved.

    Returns
    -------
    str
        Path to the saved label mapping CSV.
    """
    logger.info(f"Generating label mappings from metadata: {metadata_csv}")
    df = pd.read_csv(metadata_csv, index_col=[0, 1, 2])

    if 'cpd_type' not in df.columns:
        raise ValueError("Expected 'cpd_type' column in metadata for label generation.")

    unique_datasets = sorted(df.groupby(level=0).groups.keys())
    label_map = {}
    label_idx = 0
    label_mapping_df = pd.DataFrame()

    for dataset in unique_datasets:
        subset = df.loc[dataset]
        mapping = subset['cpd_type'].astype(str).astype('category').cat.codes
        label_mapping_df[str(dataset)] = mapping.values

    output_file = os.path.join(output_path, "label_mappings.csv")
    label_mapping_df.to_csv(output_file)
    logger.info(f"Auto-generated label mappings saved to {output_file}")
    return output_file

def clean_label_mapping_format(label_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format label mapping into CLIPn-compatible numeric format.

    Parameters
    ----------
    label_df : pd.DataFrame
        DataFrame with string labels as rows and dataset indices as columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with integer label rows.
    """
    df_clean = label_df.apply(pd.to_numeric, errors='coerce').dropna(how="all")
    df_clean = df_clean.astype("Int64")
    return df_clean

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
    label_df = clean_label_mapping_format(label_df)

    y = {}
    for k in Z:
        if str(k) not in label_df.columns:
            raise ValueError(f"Dataset {k} not found in label mapping columns")
        labels = label_df[str(k)].dropna().astype(int).values
        if len(Z[k]) != len(labels):
            raise ValueError(f"Mismatch in latent and label sizes for dataset {k}: {len(Z[k])} vs {len(labels)}")
        y[k] = labels

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
    parser.add_argument("--labels", required=False, help="Path to CSV file with label mappings.")
    parser.add_argument("--output", required=True, help="Output directory for results.")
    parser.add_argument("--metadata", required=False, help="Optional metadata CSV with MultiIndex.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Auto-generate labels if not provided but metadata exists
    if not args.labels and args.metadata:
        args.labels = generate_label_mapping_from_metadata(args.metadata, args.output)

    if not args.labels:
        raise ValueError("Either --labels or --metadata must be provided.")

    Z, y = load_latent_and_labels(args.latent, args.labels)

    generate_visualisations(Z, y, args.output)
    save_recall_at_k(Z, y, args.output)
    save_confusion_matrix(Z, y, args.output)
    save_cluster_summary(Z, y, args.output)

if __name__ == "__main__":
    main()
