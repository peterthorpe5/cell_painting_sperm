#!/usr/bin/env python3
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
        --labels path/to/label_mapping.tsv \
        --output analysis_results \
        [--metadata path/to/CLIPn_latent_representations_with_cpd_id.tsv]

Command-Line Arguments:
------------------------
    --latent         : Path to latent representation NPZ file.
    --labels         : Path to TSV file with label mappings.
    --output         : Output directory for plots and results.
    --metadata       : Optional path to metadata TSV with MultiIndex (cpd_id, Library, cpd_type).

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
from cell_painting.plot import pca_scatter
from cell_painting.process_data import generate_similarity_summary
from analyse_helpers import (
    umap_scatter_save,
    recall_k,
    confusion_matrix,
    clean_label_mapping_format,
    generate_label_mapping_from_metadata
)

def load_latent_and_labels(latent_path: str, labels_path: str) -> tuple:
    logger.info(f"Loading latent representations from: {latent_path}")
    latent_npz = np.load(latent_path, allow_pickle=True)
    Z = {int(k): np.array(v) for k, v in latent_npz.items() if not str(k).startswith("cpd_ids_")}

    logger.info(f"Loading label mappings from: {labels_path}")
    label_df = pd.read_csv(labels_path, sep="\t", index_col=0)
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

def save_recall_at_k(Z, y, output_path):
    logger.info("Computing recall@k")
    recall_values = recall_k(Z, Z, y, y)
    k = [1, 5, 10, 20, 50]
    recall_df = pd.DataFrame({'k': k, 'recall': recall_values})
    recall_file = os.path.join(output_path, "recall_at_k.tsv")
    recall_df.to_csv(recall_file, sep="\t", index=False)
    logger.info(f"Saved recall@k to {recall_file}")

def save_confusion_matrix(Z, y, output_path):
    logger.info("Computing confusion matrix")
    cm = confusion_matrix(Z, y, Z, y)
    cm_df = pd.DataFrame(cm)
    cm_file = os.path.join(output_path, "confusion_matrix.tsv")
    cm_df.to_csv(cm_file, sep="\t", index=False)
    logger.info(f"Saved confusion matrix to {cm_file}")

def save_cluster_summary(Z, y, output_path):
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
    summary_path = os.path.join(output_path, "clipn_cluster_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Cluster summary saved to {summary_path}")

def generate_visualisations(Z, y, output_path):
    logger.info("Generating UMAP plots")
    umap_scatter_save(Z, y, os.path.join(output_path, "umap"))

    logger.info("Generating PCA scatter plot")
    pca_scatter(Z, y)

def main():
    parser = argparse.ArgumentParser(description="Analyse CLIPn latent space outputs.")
    parser.add_argument("--latent", required=True, help="Path to .npz file with latent outputs.")
    parser.add_argument("--labels", required=False, help="Path to TSV file with label mappings.")
    parser.add_argument("--output", required=True, help="Output directory for results.")
    parser.add_argument("--metadata", required=False, help="Optional metadata TSV with MultiIndex.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    main()
