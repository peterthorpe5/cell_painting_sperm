#!/usr/bin/env python3
# coding: utf-8

"""
Script for imputing missing data, feature selection, and optional annotation merging.

The script processes input CSV files from Cell Painting assays by:
- Performing median or KNN imputation
- Applying feature selection (variance and correlation thresholds)
- Optionally merging annotation data
- Logging all operations and key data snapshots

Command-Line Arguments:
-----------------------
    --input_dir                : Directory containing input CSV files.
    --out               : Directory where outputs will be saved.
    --experiment          : Name of the experiment.
    --impute_method            : Method for imputing missing values ('median' or 'knn').
    --knn_neighbors            : Number of neighbors for KNN imputation.
    --correlation_threshold    : Threshold for removing highly correlated features.
    --annotation_file          : (Optional) Path to the annotation CSV file.

Example Usage:
--------------
    python impute_missing_data.py \
        --input_dir ./data/raw \
        --out ./data/processed \
        --experiment MyExperiment \
        --impute_method knn \
        --knn_neighbors 5 \
        --correlation_threshold 0.98 \
        --annotation_file ./annotations.csv

Logging:
--------
Logs key details including dataset shapes, operations performed, and warnings/errors.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
# so we keep the index .. fingers crossed!
from sklearn import set_config
set_config(transform_output="pandas")

from cell_painting.process_data import (
    group_and_filter_data,
    VarianceThreshold,
    standardise_metadata_columns,
    variance_threshold_selector,
    correlation_filter,
    load_annotation,
    standardise_annotation_columns,
    ensure_multiindex,
    restore_multiindex
)


if sys.version_info[:1] != (3,):
    # e.g. sys.version_info(major=3, minor=9, micro=7,
    # releaselevel='final', serial=0)
    # break the program
    print ("currently using:", sys.version_info,
           "  version of python")
    raise ImportError("Python 3.x is required")
    print ("did you activate the virtual environment?")
    print ("this is to deal with module imports")
    sys.exit(1)

VERSION = "cell painting: intergration: v0.0.1"
if "--version" in sys.argv:
    print(VERSION)
    sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", 
                        required=True,
                        help="input folder of cvs. OR a single csv, normalised")
    parser.add_argument("--out", 
                        default="processed")
    parser.add_argument("--experiment", 
                        default="Experiment")
    parser.add_argument("--impute", 
                        choices=["median", "knn"], 
                        default="knn")
    parser.add_argument("--knn_neighbors", 
                        type=int, default=5)
    parser.add_argument("--correlation_threshold", 
                        type=float, default=0.99)
    parser.add_argument("--annotation_file", 
                        required=False)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Setup logging
    log_dir = Path(args.out)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{args.experiment}_integration.log"

    logger = logging.getLogger("imputation_logger")
    logger.setLevel(logging.DEBUG)  # Log everything; filter per handler

    # Console handler (stream)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)  # Console shows INFO+

    # File handler (logfile)
    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # File captures all logs

    # Register handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {args.experiment}")



    # === Load Data ===
    input_path = Path(args.input)

    if input_path.suffix == ".csv":
        # Use the single specified CSV file
        input_files = [input_path]
        logger.info(f"Single input file provided: {input_path.name}")
    else:
        # Load all CSV files from directory
        input_files = list(input_path.glob("*.csv"))
        if not input_files:
            logger.error(f"No CSV files found in directory: {input_path}")
            sys.exit(1)
        logger.info(f"Found {len(input_files)} CSV files in directory: {input_path}")


    dataframes = [pd.read_csv(f, index_col=0) for f in input_files]
    df = pd.concat(dataframes, axis=0)

    # Normalise column naming
    if "library" in df.columns and "Library" not in df.columns:
        df.rename(columns={"library": "Library"}, inplace=True)
        logger.info("Renamed column 'library' to 'Library' for consistency.")

    # Extra to fix more errors
    # Ensure 'Library' column is standardised and present
    if "Library" not in df.columns:
        candidates = [col for col in df.columns if col.lower() == "library"]
        if candidates:
            df.rename(columns={candidates[0]: "Library"}, inplace=True)
            logger.info(f"Renamed column '{candidates[0]}' to 'Library' for consistency.")
        else:
            logger.warning("No 'Library' column found in input data. Adding 'Library' column using experiment name.")
            df["Library"] = args.experiment


    logger.info(f"Initial data shape: {df.shape}")

    df = standardise_metadata_columns(df)
    df = ensure_multiindex(df, logger=logger, dataset_name=args.experiment)

    if df.index.isnull().any():
        logger.warning(f" MultiIndex contains nulls after restore: {df.index[df.index.isnull().any(axis=1)]}")
        df = df[~df.index.to_frame(index=False).isnull().any(axis=1)]
        
    # Replace infinities and drop NaN columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(axis=1, how='all', inplace=True)

    # === Imputation ===
    logger.info("Starting imputation.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Backup index if it exists
    index_backup = df.index.to_frame(index=False) if isinstance(df.index, pd.MultiIndex) else None


    # Reset index without dropping, so we preserve cpd_id etc. as columns
    df = df.reset_index(drop=False)


    # Extract numeric part
    numeric_df = df[numeric_cols].copy()
    imputer = KNNImputer(n_neighbors=args.knn_neighbors) if args.impute == "knn" else SimpleImputer(strategy="median")
    imputed_numeric_df = imputer.fit_transform(numeric_df)

    # Convert back to DataFrame
    numeric_df = pd.DataFrame(imputed_numeric_df, columns=numeric_cols)

    # Merge numeric and non-numeric
    non_numeric_df = df.drop(columns=numeric_cols)
    df = pd.concat([non_numeric_df, numeric_df], axis=1)

    # Restore MultiIndex
    if index_backup is not None:
        df = restore_multiindex(df, index_backup=index_backup, dataset_name=args.experiment)

    logger.info(f"Imputation ({args.impute}) completed. Final shape: {df.shape}")

    # === Optional Annotation Merge ===
    if args.annotation_file:
        try:
            annotation_df = pd.read_csv(args.annotation_file)
            logger.info(f"Annotation file loaded: {args.annotation_file}")
            annotation_df = standardise_annotation_columns(annotation_df)

            # Set appropriate index for joining
            annotation_df = annotation_df.set_index(["Plate_Metadata", "Well_Metadata"])
            df = df.join(annotation_df, how="inner")
            logger.info(f"Data merged with annotation. New shape: {df.shape}")
        except Exception as e:
            logger.warning(f"Annotation file could not be processed: {e}")


    # === Save ungrouped version after correlation + variance filter, keeping metadata ===
    logger.info("Creating a version of the data with correlation and variance filtering before grouping.")
    try:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        if df["cpd_id"].isnull().any() or (df["cpd_id"] == "").any():
            logger.warning(" After reset_index: Some cpd_id values are blank or null — check for index misalignment or data loss.")

        blank_rows = df["cpd_id"].isnull() | (df["cpd_id"] == "")
        if blank_rows.any():
            logger.warning(f" After reset_index: {blank_rows.sum()} rows have blank or null cpd_id values — check for index misalignment or trimming.")



        # Explicitly preserve metadata columns
        metadata_cols_to_preserve = [
            "cpd_id", "cpd_type", "Library",
            "Plate_Metadata", "Well_Metadata"
        ]
        metadata_cols = [col for col in metadata_cols_to_preserve if col in df.columns]
        metadata_df = df[metadata_cols].copy()

        # Debug: Check for blank or missing cpd_id
        if metadata_df["cpd_id"].isnull().any() or (metadata_df["cpd_id"] == "").any():
            logger.warning("Some cpd_id values are blank or null in the metadata.")

        # Select numeric columns only for feature filtering
        feature_df = df.select_dtypes(include=[np.number]).copy()

        # Apply correlation and variance filters
        filtered_features = correlation_filter(feature_df, threshold=args.correlation_threshold)
        filtered_features = variance_threshold_selector(filtered_features)

        logger.debug(f"Preserved metadata columns: {metadata_df.columns.tolist()}")
        logger.debug(f"Filtered feature columns: {filtered_features.columns.tolist()}")

        # Join metadata and features
        df_ungrouped_filtered = pd.concat([metadata_df, filtered_features], axis=1)

        ungrouped_filtered_path = Path(args.out) / f"{args.experiment}_imputed_ungrouped_filtered.csv"
        df_ungrouped_filtered.to_csv(ungrouped_filtered_path, index=False)
        logger.info(f"Ungrouped, correlation- and variance-filtered data (with metadata) saved to {ungrouped_filtered_path}")
    except Exception as e:
        logger.warning(f"Could not process ungrouped correlation- and variance-filtered output: {e}")


    # === Grouping and Filtering ===
    logger.info("Grouping and filtering data by 'cpd_id' and 'Library'.")
    try:

        required_cols = ["cpd_id", "Library", "cpd_type"]
        if not isinstance(df.index, pd.MultiIndex):
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns for grouping: {missing}")
            df.set_index(required_cols, inplace=True)

        grouped_filtered_df = group_and_filter_data(df)
        grouped_filtered_file = Path(args.out) / f"{args.experiment}_imputed_grouped_filtered.csv"
        grouped_filtered_df.to_csv(grouped_filtered_file)
        logger.info(f"Grouped and filtered data saved to {grouped_filtered_file}")

    except Exception as e:
        logger.error(f"Error during grouping and filtering: {e}")
        grouped_filtered_df = df.copy()

    # === Feature Selection ===
    df_selected = correlation_filter(grouped_filtered_df, threshold=args.correlation_threshold)
    df_selected = variance_threshold_selector(df_selected)
    logger.info(f"Feature selection complete. Final shape: {df_selected.shape}")

    # === Save Final Cleaned Output ===
    output_path = Path(args.out) / f"{args.experiment}_imputed_grouped_filtered_feature_selected.csv"
    df_selected.to_csv(output_path)
    logger.info(f"Final cleaned data saved to {output_path}")
