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

    # block of code to handle inconsistant naming of cols (name, Name or cpd_id?)
    # Handle flexible cpd_id assignment
    cpd_id_series = df.get("cpd_id")
    name_series = None
    if "name" in df.columns:
        name_series = df["name"]
    elif "Name" in df.columns:
        name_series = df["Name"]


    # Standardise both if they exist
    if cpd_id_series is not None:
        cpd_id_series = cpd_id_series.astype(str).str.strip().replace("nan", np.nan)
    if name_series is not None:
        name_series = name_series.astype(str).str.strip().replace("nan", np.nan)

    # Case 1: cpd_id column missing or completely blank — fall back to name
    if cpd_id_series is None or cpd_id_series.isnull().all():
        if name_series is not None and name_series.notnull().any():
            df["cpd_id"] = name_series
            logger.info("Using 'name' column as fallback for missing 'cpd_id'.")
        else:
            df["cpd_id"] = "unknown"
            logger.warning("Both 'cpd_id' and 'name' are missing or empty. All cpd_id values set to 'unknown'.")
    else:
        df["cpd_id"] = cpd_id_series.copy()

        # Fill in blank cpd_id values with name if possible
        missing_mask = df["cpd_id"].isnull() | (df["cpd_id"] == "")
        if name_series is not None:
            fallback_mask = missing_mask & name_series.notnull()
            df.loc[fallback_mask, "cpd_id"] = name_series[fallback_mask]
            missing_mask = df["cpd_id"].isnull() | (df["cpd_id"] == "")

        # Final fill of any blanks
        if missing_mask.any():
            logger.warning(f"{args.experiment}: {missing_mask.sum()} rows have missing or blank 'cpd_id' — filling with 'unknown'.")
            df.loc[missing_mask, "cpd_id"] = "unknown"

    # Final clean
    df["cpd_id"] = df["cpd_id"].astype(str).str.strip().replace("nan", "unknown")

    # logging
    unique_cpd_count = df["cpd_id"].nunique(dropna=True)
    logger.info(f"{args.experiment}: Assigned 'cpd_id' to {unique_cpd_count} unique compounds.")


    # Normalise/ consistant column naming
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
    required_metadata = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    for col in required_metadata:
        if col not in df.columns:
            logger.warning(f"Column '{col}' is missing. Injecting placeholder value.")
            df[col] = "unknown"

    df = standardise_metadata_columns(df, logger=logger, dataset_name=args.experiment)
    df = ensure_multiindex(df, required_levels=("cpd_id", "Library", "cpd_type", "Plate_Metadata", "Well_Metadata"), logger=logger, dataset_name=args.experiment)


    if isinstance(df.index, pd.MultiIndex):
        index_df = df.index.to_frame(index=False)
        if index_df.isnull().any(axis=1).any():
            logger.warning(f"{args.experiment}: MultiIndex contains null values — these rows will be removed.")
            mask = index_df.isnull().any(axis=1)
            df = df.loc[~mask.values]



    # Replace infinities and drop NaN columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(axis=1, how='all', inplace=True)


    # === Imputation ===
    logger.info("preparing data for imputation")
    # logger.debug(f"Columns after reset_index: {df.columns.tolist()}")


    # Backup MultiIndex if present
    index_backup = df.index.to_frame(index=False) if isinstance(df.index, pd.MultiIndex) else None
    df = df.reset_index(drop=False)
    logger.debug(f"Columns after reset_index: {df.columns.tolist()}")


    # Define metadata columns that must always be excluded from imputation
    metadata_cols = [
        "cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"
    ]

    # Log and validate presence
    for col in metadata_cols:
        if col not in df.columns:
            logger.warning(f"Metadata column '{col}' is missing before imputation.")

    # Select numeric feature columns *only*, excluding metadata
    numeric_cols = [
        col for col in df.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    logger.debug(f"Columns selected for imputation: {numeric_cols}")

    # Imputation
    imputer = KNNImputer(n_neighbors=args.knn_neighbors) if args.impute == "knn" else SimpleImputer(strategy="median")
    imputed_numeric_df = imputer.fit_transform(df[numeric_cols])
    numeric_df = pd.DataFrame(imputed_numeric_df, columns=numeric_cols)

    # Merge imputed features with preserved non-numeric data
    non_numeric_df = df.drop(columns=numeric_cols)
    df = pd.concat([non_numeric_df, numeric_df], axis=1)

    logger.debug(f"Columns after imputation: {df.columns.tolist()}")
    # Double-check metadata not dropped during reset/impute
    for col in ["Plate_Metadata", "Well_Metadata"]:
        if col not in df.columns:
            logger.warning(f"Metadata column '{col}' is missing post-imputation — adding as 'unknown'.")
            df[col] = "unknown"


    # Attempt to restore MultiIndex
    if index_backup is not None:
        logger.debug(f"Index backup columns before restore: {index_backup.columns.tolist()}")
        try:
            df = restore_multiindex(df, index_backup=index_backup, dataset_name=args.experiment)
        except Exception as e:
            logger.error(f"{args.experiment}: Cannot restore MultiIndex, likely due to missing metadata columns. {e}")


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
            logger.debug(f"Columns after reset_index: {df.columns.tolist()}")

        
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

        ungrouped_filtered_path = Path(args.out) / f"{args.experiment}_imputed_ungrouped_filtered.tsv"
        df_ungrouped_filtered.to_csv(ungrouped_filtered_path, index=False, sep='\t')
        logger.info(f"Ungrouped, correlation- and variance-filtered data (with metadata) saved to {ungrouped_filtered_path}")
    except Exception as e:
        logger.warning(f"Could not process ungrouped correlation- and variance-filtered output: {e}")


    # === Grouping and Filtering ===
    logger.info("Grouping and filtering data by 'cpd_id' and 'Library'.")
    grouped_filtered_df = None
    grouped_filtered_file = Path(args.out) / f"{args.experiment}_imputed_grouped_filtered.tsv"

    try:
        required_cols = ["cpd_id", "Library", "cpd_type"]

        if isinstance(df.index, pd.MultiIndex):
            logger.debug("Resetting MultiIndex before grouping.")
            df = df.reset_index()

        logger.debug(f"Columns before grouping: {df.columns.tolist()}")
        logger.debug(f"Index before grouping: {df.index.names}")

        # Remove duplicated or conflicting columns
        for col in required_cols:
            if col in df.columns and col in df.index.names:
                logger.debug(f"Dropping column '{col}' to avoid index conflict.")
                df.drop(columns=col, inplace=True)

        # Ensure all required grouping columns exist
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column for grouping: {col}")

        # Set index for grouping
        df.set_index(required_cols, inplace=True, drop=False)
        logger.debug(f"Index set to: {df.index.names}")

        # Perform grouping
        grouped_filtered_df = group_and_filter_data(df)

        if grouped_filtered_df is None or grouped_filtered_df.empty:
            raise ValueError("Grouped dataframe is empty after filtering.")

        grouped_filtered_df.to_csv(grouped_filtered_file, sep='\t', index=False)
        logger.info(f"Grouped and filtered data saved to {grouped_filtered_file}")

    except Exception as e:
        logger.error(f"Error during grouping and filtering: {e}")
        grouped_filtered_df = None


    # === Feature Selection ===
    logger.info("Starting feature selection from grouped and filtered data.")
    df_selected = None  # Predefine so we can safely check later

    try:
        if grouped_filtered_df is None:
            raise ValueError("No valid grouped data available for feature selection.")

        # Identify metadata columns present in grouped data
        possible_metadata = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
        available_metadata = [col for col in possible_metadata if col in grouped_filtered_df.columns]

        if len(available_metadata) < 2:
            raise ValueError(f"Not enough metadata columns available for feature selection: {available_metadata}")

        metadata_df = grouped_filtered_df[available_metadata].copy()

        # Drop metadata to isolate features
        feature_df = grouped_filtered_df.drop(columns=available_metadata, errors='ignore')
        feature_df = feature_df.select_dtypes(include=[np.number])

        # Apply filters
        filtered_features = correlation_filter(feature_df, threshold=args.correlation_threshold)
        filtered_features = variance_threshold_selector(filtered_features)

        # Log number of features before and after feature selection
        n_features_before = filtered_features.shape[1]
        filtered_features = run_feature_selection(filtered_features, experiment=experiment_name)
        n_features_after = filtered_features.shape[1]

        logger.info(
            "%s: Feature selection reduced dimensionality from %d to %d features",
            experiment_name,
            n_features_before,
            n_features_after
)

        df_selected = pd.concat([metadata_df, filtered_features], axis=1)
        logger.info(f"Feature selection complete. Final shape: {df_selected.shape}")
        logger.debug(f"Metadata columns retained: {available_metadata}")
        logger.debug(f"Number of features retained: {filtered_features.shape[1]}")

    except Exception as e:
        logger.error(f"Feature selection skipped due to error: {e}")
        df_selected = None


    # === Save Final Cleaned Output ===
    if df_selected is not None:
        output_path = Path(args.out) / f"{args.experiment}_imputed_grouped_filtered_feature_selected.tsv"
        df_selected.to_csv(output_path, index=False, sep='\t')
        logger.info(f"Final cleaned data saved to {output_path}")
    else:
        logger.warning("No feature-selected data to save.")
