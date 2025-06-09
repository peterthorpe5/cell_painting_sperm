#!/usr/bin/env python3
# coding: utf-8

"""
prepare_anomaly_detection_data.py

Prepare Cell Painting well-level features for Anomaly Detection Screening (ZaritskyLab, 2024).

- Imputes missing values by default (median or KNN, or none).
- Supports per-plate scaling to reduce batch effects (StandardScaler, RobustScaler, or auto).
- Auto-selects scaler based on feature normality (Shapiro-Wilk), unless overridden.
- Preserves Cell Painting metadata: cpd_id, cpd_type, Library, Plate_Metadata, Well_Metadata.
- Handles index restoration and column standardisation.
- Splits controls into train/val/test per plate (default 40/10/50).
- Runs pycytominer feature selection.
- Ensures all data splits/treatments have identical columns for downstream compatibility.
- Outputs all splits and treatments as tab-separated files.

Requires
--------
pandas, numpy, pycytominer, scikit-learn, scipy

Example
-------
python prepare_anomaly_detection_data.py \
    --input_file cellprofiler_well_profiles.tsv \
    --output_dir prepped/ \
    --control_label DMSO \
     --experiment STB1  \
    --zscore_method mean \
    --impute knn \
    --scale_per_plate \
    --scale_method auto \
    --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
    --na_cutoff 0.05 --corr_threshold 0.9 --unique_cutoff 0.01 --freq_cut 0.05
"""

import argparse
import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pycytominer import feature_select
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import shapiro
from anomaly_detection.library import  (harmonise_columns, 
                                        check_normality,
                                        scale_per_plate,
                                        feature_selection,
                                        align_columns,
                                        impute_missing,
                                        standardise_features,
                                        split_controls)


def parse_args():
    """
    Parse command-line arguments for anomaly detection data preparation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare data for Anomaly Detection Screening.")
    parser.add_argument('--input_file', required=True, help='Input file (TSV or CSV) with well-level features.')
    parser.add_argument('--output_dir', required=True, help='Output directory.')
    parser.add_argument('--control_label', required=True, help='Value in "cpd_type" that defines negative controls (e.g. DMSO).')
    parser.add_argument('--plate_col', default='Plate_Metadata', help='Plate metadata column name.')
    parser.add_argument('--well_col', default='Well_Metadata', help='Well metadata column name.')
    parser.add_argument("--experiment", required=True, type=str, help="Experiment name or prefix to add to all output files")
    parser.add_argument('--cpd_type_col', default='cpd_type', help='Compound type column.')
    parser.add_argument('--zscore_method', choices=['mean', 'median'], default='median',
                        help='How to perform z-scoring: "mean" (standard z-score) or "median" (robust, median/MAD).')
    parser.add_argument('--impute', choices=['none', 'median', 'knn'], default='knn',
                        help='Impute missing values before scaling: "none", "median", or "knn".')
    parser.add_argument('--knn_neighbours', type=int, default=5,
                        help='Number of neighbours for KNN imputation (only if --impute knn).')
    parser.add_argument('--scale_per_plate', action="store_true", default=True,
                        help='Apply scaling per plate to reduce batch effects (default: True).')
    parser.add_argument('--scale_method', choices=['standard', 'robust', 'auto', 'none'], default='auto',
                        help='Scaling method for per-plate scaling: "standard", "robust", "auto", or "none" (default: auto).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--train_frac', type=float, default=0.4, help='Fraction for training set (default: 0.4)')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction for validation set (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.5, help='Fraction for test set (default: 0.5)')
    parser.add_argument('--na_cutoff', type=float, default=0.05,
                        help='Maximum allowed NA fraction per feature (default: 0.05)')
    parser.add_argument('--corr_threshold', type=float, default=0.9,
                        help='Correlation threshold for dropping features (default: 0.9)')
    parser.add_argument('--unique_cutoff', type=float, default=0.01,
                        help='Threshold for dropping features with low unique value ratio (default: 0.01)')
    parser.add_argument('--freq_cut', type=float, default=0.05, help='Frequency cutoff for variance thresholding (default: 0.05)')
    return parser.parse_args()

def setup_logging(out_dir):
    """
    Set up logging to file and console.

    Parameters
    ----------
    out_dir : str
        Output directory for log file.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "prepare_anomaly_detection_data.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("prep_logger")


def main():
    """
    Main execution workflow for preparing anomaly detection data.
    Loads input, imputes missing data if requested, harmonises metadata columns, splits controls,
    applies feature selection, standardises features, and saves tab-separated outputs.
    """

    args = parse_args()
    logger = setup_logging(args.output_dir)
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {args.experiment}")

    # Load and harmonise
    if args.input_file.endswith(".csv"):
        df = pd.read_csv(args.input_file)
    else:
        df = pd.read_csv(args.input_file, sep="\t")
    logger.info(f"Loaded input with shape {df.shape}")

    df = harmonise_columns(df, logger=logger)

    required_metadata = ["cpd_id", args.cpd_type_col, "Plate_Metadata", "Well_Metadata"]
    if "Library" in df.columns:
        required_metadata.append("Library")
    for col in required_metadata:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing — will fill with 'unknown'.")
            df[col] = "unknown"
    metadata_cols = required_metadata.copy()
    logger.info(f"Using metadata columns: {metadata_cols}")

    index_backup = df.index.to_frame(index=False) if isinstance(df.index, pd.MultiIndex) else None
    df = df.reset_index(drop=True)

    # ========== IMPUTATION (now BEFORE scaling) ==========
    logger.info("Checking for inf/-inf values in features... these break things ...")
    # Only check numeric columns for inf/-inf/huge values
    numeric_df = df.select_dtypes(include=[np.number])

    logger.info("Checking for inf/-inf values in numeric features...")
    n_inf = np.isinf(numeric_df).sum().sum()
    n_neg_inf = np.isneginf(numeric_df).sum().sum()
    logger.info(f"Number of inf values: {n_inf}")
    logger.info(f"Number of -inf values: {n_neg_inf}")

    logger.info("Checking for very large values (>|1e10|) in numeric features...")
    n_large = (numeric_df.abs() > 1e10).sum().sum()
    logger.info(f"Number of very large values: {n_large}")

    # Replace in the original DataFrame (numeric columns only)
    df[numeric_df.columns] = numeric_df.replace([np.inf, -np.inf], np.nan)
    df[numeric_df.columns] = df[numeric_df.columns].applymap(lambda x: np.nan if isinstance(x, float) and abs(x) > 1e10 else x)

    if n_inf or n_neg_inf or n_large:
        logger.warning(f"Replaced {n_inf} inf, {n_neg_inf} -inf, {n_large} very large values with NaN before imputation.")



    if args.impute != "none":
        logger.info(f"Imputing missing values with method '{args.impute}' (KNN neighbours: {args.knn_neighbours})")
        df = impute_missing(df, method=args.impute, knn_neighbours=args.knn_neighbours, metadata_cols=metadata_cols, logger=logger)
    else:
        logger.info("Skipping imputation (impute=none).")

    # ========== PER-PLATE SCALING ==========
    scaler_used = None
    if getattr(args, "scale_per_plate", True) and args.scale_method != "none":
        metadata_cols = required_metadata.copy()
        feature_cols = [c for c in df.columns if c not in metadata_cols]

        if args.scale_method == "auto":
            logger.info("Auto-selecting scaling method per data distribution (Shapiro-Wilk normality test)...")
            normality_tsv = os.path.join(args.output_dir, "feature_normality.tsv")
            is_normal, normality_stats = check_normality(df, feature_cols, alpha=0.05, 
                                                         logger=logger, out_tsv=normality_tsv)
            scaler_used = "standard" if is_normal else "robust"
            logger.info(f"Auto-selected scaler: {scaler_used}.")
        else:
            scaler_used = args.scale_method
            logger.info(f"Using user-specified scaler: {scaler_used}")
        df = scale_per_plate(df, metadata_cols, plate_col=args.plate_col, method=scaler_used, logger=logger)
    else:
        logger.info("Skipping per-plate scaling.")

    # ========== SPLIT CONTROLS ==========
    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-5:
        raise ValueError("Split fractions must sum to 1.0 (got %.2f)" %
                        (args.train_frac + args.val_frac + args.test_frac))

    # Split controls into train/val/test
    train_ctrl, val_ctrl, test_ctrl = split_controls(
        df,
        plate_col="Plate_Metadata",
        cpd_type_col=args.cpd_type_col,
        control_label=args.control_label,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )

    # Record indices for each split before feature selection
    train_ctrl_idx = train_ctrl.index
    val_ctrl_idx = val_ctrl.index
    test_ctrl_idx = test_ctrl.index

    treatments = df[df[args.cpd_type_col] != args.control_label]
    logger.info(f"Train controls: {train_ctrl.shape}, Validation controls: {val_ctrl.shape}, Test controls: {test_ctrl.shape}, Treatments: {treatments.shape}")


    # ========== Feature selection ==========
    # Feature selection on ALL controls, not each split separately
    meta_cols = metadata_cols
    # Save metadata for each split
    train_ctrl_meta = train_ctrl[meta_cols].copy()
    val_ctrl_meta = val_ctrl[meta_cols].copy()
    test_ctrl_meta = test_ctrl[meta_cols].copy()

    logger.info("Applying pycytominer feature selection to controls and treatments...")
    all_ctrls = pd.concat([train_ctrl, val_ctrl, test_ctrl])
    selected_ctrls = feature_selection(
        all_ctrls, metadata_cols,
        na_cutoff=args.na_cutoff,
        corr_threshold=args.corr_threshold,
        unique_cutoff=args.unique_cutoff,
        freq_cut=args.freq_cut,
        logger=logger)

    # For each split, get rows in selected_ctrls that match the split metadata
    train_ctrl = pd.merge(train_ctrl_meta, selected_ctrls, on=meta_cols, how="inner")
    val_ctrl = pd.merge(val_ctrl_meta, selected_ctrls, on=meta_cols, how="inner")
    test_ctrl = pd.merge(test_ctrl_meta, selected_ctrls, on=meta_cols, how="inner")


    selected_treatments = feature_selection(
        treatments, metadata_cols,
        na_cutoff=args.na_cutoff,
        corr_threshold=args.corr_threshold,
        unique_cutoff=args.unique_cutoff,
        freq_cut=args.freq_cut,
        logger=logger
    )



    # === Write out the feature-selected, harmonised, imputed, and scaled dataset ===
    combined_selected = pd.concat([selected_ctrls, selected_treatments], axis=0, ignore_index=True)
    # Put metadata columns first
    meta_cols = metadata_cols
    other_cols = [c for c in combined_selected.columns if c not in meta_cols]
    ordered_cols = meta_cols + other_cols
    combined_selected = combined_selected[ordered_cols]
    full_selected_outfile = os.path.join(args.output_dir, 
                                         f"{args.experiment}_full_dataset_feature_selected.tsv")
    combined_selected.to_csv(full_selected_outfile, sep="\t", index=False)
    logger.info(f"Full dataset (feature-selected, not split/z-scored) written to: {full_selected_outfile}")


    # ========== Z-SCORING ==========
    # Z-scoring is a normalisation method that transforms the values of each 
    # feature so that they have a mean of zero and a standard deviation of one 
    # (when using the standard z-score). This process removes differences in 
    # scale and units across features, allowing them to be compared and modelled 
    # together more effectively.  Mathematically, each value is transformed by subtracting 
    # the mean and dividing by the standard deviation for that feature:   can also be 
    # done using median .. median/MAD) z-score: Subtracts the median and 
    # divides by the Median Absolute Deviation (MAD), which is more robust to outliers and works 
    # better for data that are not normally distributed.

    logger.info(f"Standardising features (z-score method={args.zscore_method}) using training controls...")

    train_ctrl_std = standardise_features(
        train_ctrl, train_ctrl, metadata_cols, method=args.zscore_method, 
        logger=logger
    )
    val_ctrl_std = standardise_features(
        val_ctrl, train_ctrl, metadata_cols, method=args.zscore_method, 
        logger=logger
    )
    test_ctrl_std = standardise_features(
        test_ctrl, train_ctrl, metadata_cols, method=args.zscore_method, 
        logger=logger
    )
    treatments_std = standardise_features(
        selected_treatments, train_ctrl, metadata_cols, method=args.zscore_method, 
        logger=logger
    )



    # ========== ALIGN OUTPUT COLUMNS ==========
    aligned = align_columns(
        [train_ctrl_std, val_ctrl_std, test_ctrl_std, treatments_std],
        metadata_cols=metadata_cols, logger=logger
    )
    train_ctrl_std, val_ctrl_std, test_ctrl_std, treatments_std = aligned

    # ========== OUTPUT ==========
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ctrl_std.to_csv(os.path.join(args.output_dir, f"{args.experiment}_train_controls.tsv"), sep="\t", index=False)
    val_ctrl_std.to_csv(os.path.join(args.output_dir, f"{args.experiment}_val_controls.tsv"), sep="\t", index=False)
    test_ctrl_std.to_csv(os.path.join(args.output_dir, f"{args.experiment}_test_controls.tsv"), sep="\t", index=False)
    treatments_std.to_csv(os.path.join(args.output_dir, f"{args.experiment}_treatments.tsv"), sep="\t", index=False)
    logger.info("All splits saved.")

    logger.info(f"train_controls.tsv shape: {train_ctrl_std.shape}")
    logger.info(f"val_controls.tsv shape: {val_ctrl_std.shape}")
    logger.info(f"test_controls.tsv shape: {test_ctrl_std.shape}")
    logger.info(f"treatments.tsv shape: {treatments_std.shape}")

    if index_backup is not None:
        logger.info("MultiIndex restoration not implemented here, but index is preserved in output columns for downstream merging.")

    logger.info("Completed data preparation for anomaly detection.")

if __name__ == "__main__":
    main()
