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

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
# so we keep the index .. fingers crossed!
from sklearn import set_config
set_config(transform_output="pandas")

from cell_painting.process_data import (
    group_and_filter_data, 
    standardise_metadata_columns,
    variance_threshold_selector,
    correlation_filter,
    load_annotation,
    standardise_annotation_columns
)


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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



    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {args.experiment}")

    os.makedirs(args.out, exist_ok=True)


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

    logger.info(f"Initial data shape: {df.shape}")

    df = standardise_metadata_columns(df)

    # Replace infinities and drop NaN columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(axis=1, how='all', inplace=True)

    # === Imputation ===
    imputer = KNNImputer(n_neighbors=args.knn_neighbors) if args.impute == "knn" else SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    logger.info(f"Imputation ({args.impute}) completed.")

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


    # === Grouping and Filtering ===
    logger.info("Grouping and filtering data by 'cpd_id' and 'Library'.")
    try:
        from cell_painting.process_data import group_and_filter_data

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
    output_path = Path(args.out) / f"{args.experiment}_imputed_grouped_filtered_fselected.csv"
    df_selected.to_csv(output_path)
    logger.info(f"Final cleaned data saved to {output_path}")



