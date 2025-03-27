#!/usr/bin/env python3
# coding: utf-8

"""
Run CLIPn Integration on Cell Painting Data
-------------------------------------------

This script:
- Loads and merges multiple reference and query datasets.
- Harmonises column features across datasets.
- Encodes labels for compatibility with CLIPn.
- Runs CLIPn integration analysis (either train on references or integrate all).
- Decodes labels post-analysis, restoring original annotations.
- Outputs results, including latent representations and similarity matrices.

Command-line arguments:
-----------------------
    --datasets_csv      : Path to CSV listing dataset names and paths.
    --out               : Directory to save outputs.
    --experiment        : Experiment name for file naming.
    --mode              : Operation mode ('reference_only' or 'integrate_all').
    --clipn_param       : Optional CLIPn parameter for tuning (e.g., number of epochs).
    --latent_dim        : Dimensionality of latent space (default: 20).
    --lr                : Learning rate for CLIPn (default: 1e-5).
    --epoch             : Number of training epochs (default: 300).

Logging:
--------
Logs detailed info and debug-level outputs.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import set_config
import csv
from cell_painting.process_data import (

        prepare_data_for_clipn_from_df,
        run_clipn_simple,
        standardise_metadata_columns
)


set_config(transform_output="pandas")

def setup_logging(out_dir, experiment):
    """Configure logging with stream and file handlers."""
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{experiment}_clipn.log"

    logger = logging.getLogger("clipn_logger")
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {experiment}")

    return logger


def detect_csv_delimiter(csv_path):
    """Detect the delimiter of a CSV file."""
    with open(csv_path, 'r', newline='') as csvfile:
        sample = csvfile.read(2048)
        csvfile.seek(0)
        if ',' in sample and '\t' not in sample:
            return ','
        elif '\t' in sample and ',' not in sample:
            return '\t'
        else:
            # default to comma if both are found or none
            return ','

# this is the problem function when we loose cpd_id
def load_and_harmonise_datasets(datasets_csv, logger, mode=None):
    """
    Load datasets from CSV and harmonise numeric feature columns,
    while preserving metadata columns such as cpd_type, cpd_id, and Library.

    Parameters
    ----------
    datasets_csv : str
        Path to the CSV file listing dataset names and their paths.
    logger : logging.Logger
        Logger instance for logging progress and messages.
    mode : str, optional
        Mode flag (e.g., 'reference_only') to control which datasets to harmonise.

    Returns
    -------
    tuple
        dataframes : dict
            Dictionary of harmonised pandas DataFrames indexed by dataset name.
        common_cols : list
            List of numeric feature columns that were harmonised across datasets.
    """
    delimiter = detect_csv_delimiter(datasets_csv)
    datasets_df = pd.read_csv(datasets_csv, delimiter=delimiter)
    dataset_paths = datasets_df.set_index('dataset')['path'].to_dict()

    dataframes = {}
    metadata_dict = {}
    all_numeric_cols = set()
    metadata_cols = ["cpd_id", "cpd_type", "Library"]

    logger.info("Loading datasets")
    for name, path in dataset_paths.items():
        df = pd.read_csv(path, index_col=0)

        df = standardise_metadata_columns(df, logger=logger, dataset_name=name)

        # Stash metadata separately before subsetting
        available_metadata = [col for col in metadata_cols if col in df.columns]
        metadata_dict[name] = df[available_metadata].copy()

        logger.debug(f"After standardisation, columns in '{name}': {df.columns.tolist()}")
        dataframes[name] = df
        all_numeric_cols.update(df.select_dtypes(include=[np.number]).columns)
        logger.debug(f"Loaded {name}: shape {df.shape}")

    if mode == "reference_only":
        logger.info("Running in reference_only mode: loading and harmonising reference datasets only.")
        reference_dfs = {k: v for k, v in dataframes.items() if "reference" in k.lower()}

        common_cols = sorted(list(
            set.intersection(*(set(df.select_dtypes(include=[np.number]).columns) for df in reference_dfs.values()))
        ))
        logger.info(f"Harmonised feature columns across references: {len(common_cols)}")

        for name in reference_dfs:
            df = reference_dfs[name]
            logger.debug(f"[{name}] Columns BEFORE subsetting: {df.columns.tolist()}")
            df = df[common_cols].copy()

            if name in metadata_dict:
                meta = metadata_dict[name].reindex(df.index)
                df = pd.concat([df, meta], axis=1)
                logger.debug(f"[{name}] Metadata reattached. Final columns: {df.columns.tolist()}")

            dataframes[name] = df

        return dataframes, common_cols

    # For integrate_all mode
    common_cols = sorted(list(
        all_numeric_cols.intersection(*[set(df.columns) for df in dataframes.values()])
    ))
    logger.info(f"Harmonised feature columns count: {len(common_cols)}")

    for name in dataframes:
        df = dataframes[name]
        logger.debug(f"[{name}] Columns BEFORE subsetting: {df.columns.tolist()}")
        df = df[common_cols].copy()

        if name in metadata_dict:
            meta = metadata_dict[name].reindex(df.index)
            df = pd.concat([df, meta], axis=1)
            logger.debug(f"[{name}] Metadata reattached. Final columns: {df.columns.tolist()}")

        dataframes[name] = df
    return dataframes, common_cols


def standardise_numeric_columns_preserving_metadata(
    df: pd.DataFrame, 
    meta_columns: list[str]
) -> pd.DataFrame:
    """
    Standardise numeric columns in a MultiIndex DataFrame, preserving metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with MultiIndex (dataset, sample).
    meta_columns : list of str
        List of columns to exclude from standardisation.

    Returns
    -------
    pd.DataFrame
        Standardised DataFrame with metadata columns preserved.
    """
    df = df.copy()
    metadata = df[meta_columns]
    numeric = df.drop(columns=meta_columns)

    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(numeric),
        columns=numeric.columns,
        index=df.index,
    )

    return pd.concat([numeric_scaled, metadata], axis=1)




def encode_labels(df, logger):
    """Encode categorical columns and return encoders for decoding later."""
    encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        logger.debug(f"Encoded column {col}")
    return df, encoders


def decode_labels(df, encoders, logger):
    """Decode categorical columns to original labels."""
    for col, le in encoders.items():
        df[col] = le.inverse_transform(df[col])
        logger.debug(f"Decoded column {col}")
    return df


def run_clipn_integration(df, logger, clipn_param, output_path, latent_dim, lr, epochs):
    """Run the actual CLIPn integration logic."""
    logger.info(f"Running CLIPn integration with param: {clipn_param}")
    meta_cols = ["cpd_id", "cpd_type", "Library"]  # or any others
    # Ensure all meta_columns exist before scaling
    missing_meta = [col for col in meta_cols if col not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing metadata columns in input DataFrame before scaling: {missing_meta}")

    df_scaled = standardise_numeric_columns_preserving_metadata(df, meta_columns=meta_cols)

    X, y, label_mappings, ids = prepare_data_for_clipn_from_df(df_scaled)

    data_dict, label_dict, label_mappings, cpd_ids = prepare_data_for_clipn_from_df(df_scaled)

    latent_dict = run_clipn_simple(data_dict, label_dict, latent_dim=latent_dim, lr=lr, epochs=epochs)

    latent_combined = pd.concat(latent_dict.values(), keys=latent_dict.keys(), names=["Dataset", "Sample"])

    latent_file = Path(output_path) / "CLIPn_latent_representations.npz"
    np.savez(latent_file, **latent_dict)
    logger.info(f"Latent representations saved to: {latent_file}")


    for dataset, latent in latent_dict.items():
        per_dataset_path = Path(output_path) / f"{dataset}_latent.csv"
        latent_df = pd.DataFrame(latent)

        # Add cpd_id back, if available
        if dataset in cpd_ids:
            latent_df[id_col] = cpd_ids[dataset]

        latent_df.to_csv(per_dataset_path, index=False)
        logger.info(f"Saved latent CSV for {dataset} to {per_dataset_path}")


    return latent_combined, cpd_ids

def main(args):
    """Main function to execute CLIPn integration pipeline."""
    logger = setup_logging(args.out, args.experiment)

    dataframes, common_cols = load_and_harmonise_datasets(args.datasets_csv, logger, mode=args.mode)

    combined_df = pd.concat(dataframes.values(), keys=dataframes.keys(), names=['Dataset', 'Sample'])
    logger.debug(f"Columns at this stage, combined: {combined_df.columns.tolist()}")
    

    combined_df, encoders = encode_labels(combined_df, logger)

    if args.mode == "reference_only":
        reference_names = [name for name in dataframes if 'reference' in name.lower()]
        reference_df = combined_df.loc[reference_names]
        logger.info(f"Training CLIPn on references: {reference_names}")
        latent_df, cpd_ids = run_clipn_integration(reference_df, logger, args.clipn_param, args.out, 
                                                   args.latent_dim, args.lr, args.epoch)

    else:
        logger.info("Training and integrating CLIPn on all datasets")
        latent_df, cpd_ids = run_clipn_integration(combined_df, logger, args.clipn_param, args.out, 
                                                   args.latent_dim, args.lr, args.epoch)


  
    decoded_df = decode_labels(latent_df.copy(), encoders, logger)
    decoded_path = Path(args.out) / f"{args.experiment}_decoded.csv"
    decoded_df.to_csv(decoded_path)
    logger.info(f"Decoded data saved to {decoded_path}")

    # Save renamed latent with original compound IDs (e.g., cpd_id)
    try:
        decoded_with_index = decoded_df.reset_index()

        # Add cpd_id from cpd_ids dict
        decoded_with_index["cpd_id"] = decoded_with_index.apply(
            lambda row: cpd_ids.get(row["Dataset"], [None])[row["Sample"]],
            axis=1
        )

        renamed_path = Path(args.out) / f"{experiment}_CLIPn_latent_representations_with_cpd_id.csv"
        decoded_with_index.to_csv(renamed_path, index=False)
    except Exception as e:
        logger.warning(f"Failed to save renamed latent file: {e}")
    logger.info(f"Columns at this stage, encoded: {combined_df.columns.tolist()}")

    # Save label encoder mappings
    try:
        mapping_dir = Path(args.out)
        mapping_dir.mkdir(parents=True, exist_ok=True)

        for column, encoder in encoders.items():
            mapping_path = mapping_dir / f"label_mapping_{column}.csv"
            mapping_df = pd.DataFrame({
                column: encoder.classes_,
                f"{column}_encoded": range(len(encoder.classes_))
            })
            mapping_df.to_csv(mapping_path, index=False)
            logger.info(f"Saved label mapping for {column} to {mapping_path}")
    except Exception as e:
        logger.warning(f"Failed to save label encoder mappings: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIPn Integration.")
    parser.add_argument("--datasets_csv", required=True, help="CSV listing dataset names and paths.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--experiment", required=True, help="Experiment name.")
    parser.add_argument("--mode", choices=['reference_only', 'integrate_all'], required=True,
                        help="Mode of CLIPn operation.")
    parser.add_argument("--clipn_param", type=str, default="default",
                        help="Optional CLIPn parameter for tuning (e.g., number of epochs).")
    parser.add_argument("--latent_dim", type=int, default=20,
                        help="Dimensionality of latent space (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for CLIPn (default: 1e-5)")
    parser.add_argument("--epoch", type=int, default=300,
                        help="Number of training epochs (default: 300)")

    args = parser.parse_args()
    main(args)