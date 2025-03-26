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
    --output_dir               : Directory where outputs will be saved.
    --experiment_name          : Name of the experiment.
    --impute_method            : Method for imputing missing values ('median' or 'knn').
    --knn_neighbors            : Number of neighbors for KNN imputation.
    --correlation_threshold    : Threshold for removing highly correlated features.
    --annotation_file          : (Optional) Path to the annotation CSV file.

Example Usage:
--------------
    python impute_missing_data.py \
        --input_dir ./data/raw \
        --output_dir ./data/processed \
        --experiment_name MyExperiment \
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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def variance_threshold_selector(data, threshold=0.05):
    """Select features based on variance threshold."""
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data.iloc[:, selector.get_support(indices=True)]


def correlation_filter(data, threshold=0.99):
    """Remove highly correlated features based on threshold."""
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    return data.drop(columns=drop_cols)


def load_annotation(annotation_path):
    """Load annotation file safely."""
    try:
        annotation_df = pd.read_csv(annotation_path)
        annotation_df.columns = annotation_df.columns.str.strip().str.replace(" ", "_")
        annotation_df.set_index(['Plate_Metadata', 'Well_Metadata'], inplace=True)
        logger.info(f"Annotation file loaded with shape {annotation_df.shape}")
        return annotation_df
    except Exception as e:
        logger.warning(f"Annotation file could not be loaded: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_name", default="Experiment")
    parser.add_argument("--impute_method", choices=["median", "knn"], default="knn")
    parser.add_argument("--knn_neighbors", type=int, default=5)
    parser.add_argument("--correlation_threshold", type=float, default=0.99)
    parser.add_argument("--annotation_file", required=False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {args.experiment_name}")

    # Load data
    input_files = list(Path(args.input_dir).glob("*.csv"))
    if not input_files:
        logger.error("No input CSV files found.")
        sys.exit(1)

    dataframes = [pd.read_csv(f, index_col=0) for f in input_files]
    df = pd.concat(dataframes, axis=0)
    logger.info(f"Initial data shape: {df.shape}")

    # Replace infinities and drop NaN columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(axis=1, how='all', inplace=True)

    # Imputation
    imputer = KNNImputer(n_neighbors=args.knn_neighbors) if args.impute_method == "knn" else SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    logger.info(f"Imputation ({args.impute_method}) completed.")

    # Optional Annotation Merge
    if args.annotation_file:
        annotation_df = load_annotation(args.annotation_file)
        if annotation_df is not None:
            df = df.join(annotation_df, how='inner')
            logger.info(f"Data merged with annotation. New shape: {df.shape}")

    # Feature selection
    df = correlation_filter(df, threshold=args.correlation_threshold)
    df = variance_threshold_selector(df)
    logger.info(f"Feature selection complete. Final shape: {df.shape}")

    # Save cleaned data
    output_path = Path(args.output_dir) / f"{args.experiment_name}_cleaned.csv"
    df.to_csv(output_path)
    logger.info(f"Data saved to {output_path}")
