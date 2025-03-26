#!/usr/bin/env python3
# coding: utf-8
"""
Script for running CLIPn clustering on multiple datasets with robust handling of
encoding, consistency checks, and detailed logging.

Two modes of operation:
    1. Train on a reference dataset and then apply to additional datasets.
    2. Train and run CLIPn on all provided datasets simultaneously.

Key features:
- Automatic handling of pre-encoded datasets or raw datasets that need encoding.
- Consistency checks for encoded labels, column alignment across datasets.
- Logging of file hashes/checksums to detect changes.

Usage:
    python run_clipn.py \
        --datasets dataset_paths.txt \
        --output output_folder \
        --reference reference_dataset_name \
        [--use-pre-encoded] \
        [--latent-dim 20] \
        [--epochs 200] \
        [--lr 1e-5]
"""

import argparse
import hashlib
import logging
import os
import pandas as pd
from clipn import CLIPn
from sklearn.preprocessing import LabelEncoder
from process_data import prepare_data_for_clipn, encode_cpd_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



def load_datasets(dataset_list_file, use_pre_encoded=False):
    """
    Load datasets from file and optionally encode them.

    Parameters
    ----------
    dataset_list_file : str
        Path to the text file listing dataset names and folder paths.
    use_pre_encoded : bool, optional
        Whether to use pre-encoded CSVs (default: False).

    Returns
    -------
    dict
        Dictionary of dataset name to DataFrame.
    dict
        Dictionary of file paths to checksum hashes.
    """
    datasets = {}
    checksums = {}

    with open(dataset_list_file, 'r') as f:
        for line in f:
            name, path = line.strip().split()
            csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

            dfs = []
            for file in csv_files:
                df = pd.read_csv(file, index_col=[0, 1, 2])
                dfs.append(df)
                checksums[file] = compute_checksum(file)

            dataset_df = pd.concat(dfs)

            if not use_pre_encoded:
                encoded_results = encode_cpd_data({name: dataset_df}, encode_labels=True)
                dataset_df = encoded_results[name]["data"]

            datasets[name] = dataset_df
            logger.info(f"Loaded {name} with shape {dataset_df.shape}")

    return datasets, checksums


def save_latent_representations(Z, output_folder, dataset_mapping, index_lookup):
    """
    Saves latent representations in NPZ and CSV formats with MultiIndex.

    Parameters
    ----------
    Z : dict
        Latent representations from CLIPn.
    output_folder : str
        Folder to save outputs.
    dataset_mapping : dict
        Maps dataset indices to dataset names.
    index_lookup : dict
        Original indices for datasets.
    """
    Z_named = {str(dataset_mapping[k]): v.tolist() for k, v in Z.items()}
    np.savez(os.path.join(output_folder, "CLIPn_latent_representations.npz"), **Z_named)
    logger.info("Latent representations saved successfully in NPZ format.")

    combined_latent_df = reconstruct_combined_latent_df(Z, dataset_mapping, index_lookup)
    combined_output_file = os.path.join(output_folder, "CLIPn_latent_representations_with_cpd_id.csv")
    combined_latent_df.to_csv(combined_output_file)
    logger.info(f"Combined latent DataFrame saved to {combined_output_file}.")
    # Save metadata (cpd_id, cpd_type, Library) for downstream analysis
    metadata_cols = ["cpd_id", "cpd_type"]
    if "Library" in combined_latent_df.index.names:
        metadata_cols.append("Library")

    metadata_df = combined_latent_df.reset_index()[metadata_cols]
    metadata_df.to_csv(os.path.join(output_folder, "latent_metadata.csv"), index=False)
    logger.info("Latent metadata saved to 'latent_metadata.csv'")


def run_clipn(datasets, reference, output_folder, latent_dim, epochs, lr, mode="reference"):
    """
    Run CLIPn in reference or all mode.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset name to DataFrame.
    reference : str
        Reference dataset name (only used in reference mode).
    output_folder : str
        Path to the output directory.
    latent_dim : int
        Number of latent dimensions for CLIPn.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    mode : str, optional
        Either "reference" or "all" (default: "reference").

    Returns
    -------
    None
    """
    X, y, _, dataset_mapping = prepare_data_for_clipn(datasets)

    if mode == "reference":
        reference_idx = [idx for idx, name in dataset_mapping.items() if name == reference][0]
        logger.info(f"Training CLIPn on reference dataset: {reference}")
        clipn_model = CLIPn({reference_idx: X[reference_idx]}, {reference_idx: y[reference_idx]}, latent_dim=latent_dim)
        clipn_model.fit({reference_idx: X[reference_idx]}, {reference_idx: y[reference_idx]}, lr=lr, epochs=epochs)

        Z = {}
        for idx in X:
            Z[idx] = clipn_model.predict({idx: X[idx]})[idx]
    else:
        logger.info("Training CLIPn on all datasets simultaneously.")
        clipn_model = CLIPn(X, y, latent_dim=latent_dim)
        clipn_model.fit(X, y, lr=lr, epochs=epochs)
        Z = clipn_model.predict(X)

    for idx, latent in Z.items():
        dataset_name = dataset_mapping[idx]
        output_path = os.path.join(output_folder, f"{dataset_name}_latent.csv")
        pd.DataFrame(latent, index=datasets[dataset_name].index).to_csv(output_path)
        logger.info(f"Latent representation for {dataset_name} saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIPn clustering on multiple datasets.")
    parser.add_argument("--datasets", 
                        required=True, 
                        help="Path to the dataset list file.")
    parser.add_argument("--output", 
                        required=True, 
                        help="Output folder to save results.")
    parser.add_argument("--reference", 
                        required=False, 
                        help="Reference dataset name (for reference mode).")
    parser.add_argument("--use-pre-encoded", 
                        action="store_true", 
                        help="Use pre-encoded data (skip encoding).")
    parser.add_argument("--latent-dim", 
                        type=int, 
                        default=20, 
                        help="Number of latent dimensions.")
    parser.add_argument("--epochs", 
                        type=int, 
                        default=300, 
                        help="Number of training epochs.")
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-5, 
                        help="Learning rate for CLIPn.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Using Logfile: {log_filename}")
    logger.info(f"Logging initialized at {time.asctime()}")

    datasets, checksums = load_datasets(args.datasets, use_pre_encoded=args.use_pre_encoded)

    mode = "reference" if args.reference else "all"
    run_clipn(datasets, args.reference, args.output, args.latent_dim, args.epochs, args.lr, mode=mode)

