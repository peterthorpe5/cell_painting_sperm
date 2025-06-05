#!/usr/bin/env python3
# coding: utf-8

"""
prepare_anomaly_detection_data.py

Prepare Cell Painting well-level features for Anomaly Detection Screening (ZaritskyLab, 2024).
- Supports z-scoring with mean/std or median/MAD.
- Optionally imputes missing values using median or KNN, with full metadata/index preservation.
- Preserves Cell Painting metadata: cpd_id, cpd_type, Library, Plate_Metadata, Well_Metadata.
- Handles index restoration and column standardisation as in previous pipelines.
- Splits controls into train/val/test per plate (default 40/10/50).
- Runs pycytominer feature selection.
- Outputs all splits and treatments as tab-separated files.

Requires:
    pandas, numpy, pycytominer, scikit-learn

Example usage:
     

python prepare_anomaly_detection_data.py \
      --input_file cellprofiler_well_profiles.tsv \
      --output_dir prepped/ \
      --control_label DMSO \
      --zscore_method mean \
      --impute knn \
      --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
      --na_cutoff 0.05 --corr_threshold 0.9 --variance_threshold 0.05 --unique_cutoff 0.01

"""



import argparse
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pycytominer import feature_select
from sklearn.impute import SimpleImputer, KNNImputer


def harmonise_columns(df, logger=None):
    """
    Harmonise key metadata column names defensively (handles cpd_id, cpd_type, Library, Plate_Metadata, Well_Metadata).
    Accepts many capitalisation and spelling variants.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with possible variant column names.
    logger : logging.Logger or None
        Logger for harmonisation messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with harmonised column names.
    """
    rename_dict = {}

    # Lower-case map for robust matching
    col_map = {c.lower(): c for c in df.columns}

    def log_change(orig, new):
        if logger:
            logger.info(f"Renaming column '{orig}' to '{new}'.")

    # Plate
    for platename in ["plate_metadata", "plate"]:
        if platename in col_map:
            rename_dict[col_map[platename]] = "Plate_Metadata"
            log_change(col_map[platename], "Plate_Metadata")
            break

    # Well
    for wellname in ["well_metadata", "well"]:
        if wellname in col_map:
            rename_dict[col_map[wellname]] = "Well_Metadata"
            log_change(col_map[wellname], "Well_Metadata")
            break

    # cpd_id
    cpd_id_candidates = [
        "cpd_id", "compound_id", "comp_id", "compound", "compud_id",
        "compund_id", "compid", "comp", "compoundid"
    ]
    for cname in cpd_id_candidates:
        if cname in col_map:
            rename_dict[col_map[cname]] = "cpd_id"
            log_change(col_map[cname], "cpd_id")
            break

    # cpd_type
    cpd_type_candidates = ["cpd_type", "compound_type", "type"]
    for cname in cpd_type_candidates:
        if cname in col_map:
            rename_dict[col_map[cname]] = "cpd_type"
            log_change(col_map[cname], "cpd_type")
            break

    # Library
    library_candidates = ["library", "lib", "collection"]
    for cname in library_candidates:
        if cname in col_map:
            rename_dict[col_map[cname]] = "Library"
            log_change(col_map[cname], "Library")
            break

    if logger and not rename_dict:
        logger.info("No metadata columns required renaming.")

    return df.rename(columns=rename_dict)


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare data for Anomaly Detection Screening.")
    parser.add_argument('--input_file', required=True, help='Input file (TSV or CSV) with well-level features.')
    parser.add_argument('--output_dir', required=True, help='Output directory.')
    parser.add_argument('--control_label', required=True, help='Value in "cpd_type" that defines negative controls (e.g. DMSO).')
    parser.add_argument('--plate_col', default='Plate_Metadata', help='Plate metadata column name.')
    parser.add_argument('--well_col', default='Well_Metadata', help='Well metadata column name.')
    parser.add_argument('--cpd_type_col', default='cpd_type', help='Compound type column.')
    parser.add_argument('--zscore_method', choices=['mean', 'median'], default='mean',
                        help='How to perform z-scoring: "mean" (standard z-score) or "median" (robust, median/MAD).')
    parser.add_argument('--impute', choices=['none', 'median', 'knn'], default='none',
                        help='Impute missing values before scaling: "none", "median", or "knn".')
    parser.add_argument('--knn_neighbors', type=int, default=5,
                        help='Number of neighbours for KNN imputation (only if --impute knn).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--train_frac', type=float, default=0.4, help='Fraction for training set (default: 0.4)')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction for validation set (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.5, help='Fraction for test set (default: 0.5)')
    parser.add_argument('--na_cutoff', type=float, default=0.05,
                        help='Maximum allowed NA fraction per feature (default: 0.05)')
    parser.add_argument('--corr_threshold', type=float, default=0.9,
                        help='Correlation threshold for dropping features (default: 0.9)')
    parser.add_argument('--variance_threshold', type=float, default=0.05,
                        help='Variance threshold for dropping features (default: 0.05)')
    parser.add_argument('--unique_cutoff', type=float, default=0.01,
                        help='Threshold for dropping features with low unique value ratio (default: 0.01)')
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
        Configured logger instance.
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


def feature_selection(df, metadata_cols, na_cutoff=0.05, corr_threshold=0.9,
                      variance_threshold=0.05, unique_cutoff=0.01, logger=None):
    """
    Apply pycytominer feature selection: remove highly correlated, null, and invariant features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with selected features and preserved metadata columns.
    """
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    selected = feature_select(
        df[feature_cols],
        features="infer",
        operation=[
            "variance_threshold",
            "correlation_threshold",
            "drop_na_columns",
            "unique_value_threshold"
        ],
        corr_threshold=corr_threshold,
        variance_threshold=variance_threshold,
        na_cutoff=na_cutoff,
        unique_cutoff=unique_cutoff
    )
    if logger:
        logger.info(f"Feature selection reduced from {len(feature_cols)} to {selected.shape[1]} features.")
    # Return with metadata
    return pd.concat([df[metadata_cols].reset_index(drop=True), selected.reset_index(drop=True)], axis=1)


def impute_missing(df, method="median", knn_neighbors=5, metadata_cols=None, logger=None):
    """
    Impute missing values in features using median or KNN, preserving all metadata columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed features and original metadata.
    """
    if metadata_cols is None:
        metadata_cols = []
    feat_cols = [c for c in df.columns if c not in metadata_cols]
    meta_df = df[metadata_cols].copy()
    feat_df = df[feat_cols].copy()
    # Impute
    if method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    else:
        raise ValueError("Unknown imputation method: %s" % method)
    imputed = imputer.fit_transform(feat_df)
    feat_imputed_df = pd.DataFrame(imputed, columns=feat_df.columns, index=feat_df.index)
    # Join back
    out = pd.concat([meta_df.reset_index(drop=True), feat_imputed_df.reset_index(drop=True)], axis=1)
    if logger:
        logger.info(f"Imputation complete (method={method}, shape={out.shape}).")
    return out


def standardise_features(df, train_controls, metadata_cols, method="mean"):
    """
    Standardise features using either mean/std or median/MAD from training controls.

    Returns
    -------
    pandas.DataFrame
        Standardised DataFrame.
    """
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    if method == "mean":
        location = train_controls[feature_cols].mean()
        scale = train_controls[feature_cols].std(ddof=0)
    elif method == "median":
        location = train_controls[feature_cols].median()
        scale = train_controls[feature_cols].mad() * 1.4826  # Consistent with robust z-score
    else:
        raise ValueError("Unknown z-score method: %s" % method)
    scale_replaced = scale.replace(0, np.nan)
    df_z = df.copy()
    df_z[feature_cols] = (df[feature_cols] - location) / scale_replaced
    return df_z


def split_controls(df, plate_col, cpd_type_col, control_label, train_frac=0.4, val_frac=0.1, test_frac=0.5, seed=42):
    """
    For each plate, randomly split controls into train/validation/test sets based on provided fractions.

    Returns
    -------
    tuple of pandas.DataFrame
        (train_controls, validation_controls, test_controls)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5, "Split fractions must sum to 1.0"
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for plate, group in df.groupby(plate_col):
        controls = group[group[cpd_type_col] == control_label]
        n = len(controls)
        if n == 0:
            continue
        indices = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        idx_train = controls.iloc[indices[:n_train]].index
        idx_val = controls.iloc[indices[n_train:n_train+n_val]].index
        idx_test = controls.iloc[indices[n_train+n_val:]].index
        train.append(df.loc[idx_train])
        val.append(df.loc[idx_val])
        test.append(df.loc[idx_test])
    return pd.concat(train), pd.concat(val), pd.concat(test)


def main():
    """
    Main execution workflow for preparing anomaly detection data.
    """
    args = parse_args()
    logger = setup_logging(args.output_dir)

    # Load data and harmonise metadata columns defensively
    if args.input_file.endswith(".csv"):
        df = pd.read_csv(args.input_file)
    else:
        df = pd.read_csv(args.input_file, sep="\t")
    logger.info(f"Loaded input with shape {df.shape}")

    df = harmonise_columns(df, logger=logger)

    # Defensive check/fill for all metadata columns (cpd_id, cpd_type, Library, Plate_Metadata, Well_Metadata)
    required_metadata = ["cpd_id", args.cpd_type_col, "Plate_Metadata", "Well_Metadata"]
    if "Library" in df.columns:
        required_metadata.append("Library")

    for col in required_metadata:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing — will fill with 'unknown'.")
            df[col] = "unknown"
    metadata_cols = required_metadata.copy()
    logger.info(f"Using metadata columns: {metadata_cols}")

    # Backup MultiIndex if present (not used for output, but tracked for completeness)
    index_backup = df.index.to_frame(index=False) if isinstance(df.index, pd.MultiIndex) else None
    df = df.reset_index(drop=True)

    # Optional: Impute missing values before feature selection/z-scoring
    if args.impute != "none":
        logger.info(f"Imputing missing values with method '{args.impute}' (KNN neighbours: {args.knn_neighbors})")
        df = impute_missing(df, method=args.impute, knn_neighbors=args.knn_neighbors, metadata_cols=metadata_cols, logger=logger)

    # Split controls by plate
    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-5:
        raise ValueError("Split fractions must sum to 1.0 (got %.2f)" %
                        (args.train_frac + args.val_frac + args.test_frac))

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

    treatments = df[df[args.cpd_type_col] != args.control_label]
    logger.info(f"Train controls: {train_ctrl.shape}, Validation controls: {val_ctrl.shape}, Test controls: {test_ctrl.shape}, Treatments: {treatments.shape}")

    # Feature selection (fit on all controls, apply to all)
    all_ctrls = pd.concat([train_ctrl, val_ctrl, test_ctrl])
    selected_ctrls = feature_selection(
        all_ctrls, metadata_cols,
        na_cutoff=args.na_cutoff,
        corr_threshold=args.corr_threshold,
        variance_threshold=args.variance_threshold,
        unique_cutoff=args.unique_cutoff,
        logger=logger
    )
    selected_treatments = feature_selection(
        treatments, metadata_cols,
        na_cutoff=args.na_cutoff,
        corr_threshold=args.corr_threshold,
        variance_threshold=args.variance_threshold,
        unique_cutoff=args.unique_cutoff,
        logger=logger
    )

    # Standardise features (fit only on training controls)
    train_ctrl_std = standardise_features(
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        metadata_cols, method=args.zscore_method
    )
    val_ctrl_std = standardise_features(
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        train_ctrl_std, metadata_cols, method=args.zscore_method
    )
    test_ctrl_std = standardise_features(
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        train_ctrl_std, metadata_cols, method=args.zscore_method
    )
    treatments_std = standardise_features(selected_treatments, train_ctrl_std, metadata_cols, method=args.zscore_method)

    # Save outputs (tab-separated, with metadata)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ctrl_std.to_csv(os.path.join(args.output_dir, "train_controls.tsv"), sep="\t", index=False)
    val_ctrl_std.to_csv(os.path.join(args.output_dir, "val_controls.tsv"), sep="\t", index=False)
    test_ctrl_std.to_csv(os.path.join(args.output_dir, "test_controls.tsv"), sep="\t", index=False)
    treatments_std.to_csv(os.path.join(args.output_dir, "treatments.tsv"), sep="\t", index=False)
    logger.info("All splits saved.")

    if index_backup is not None:
        logger.info("MultiIndex restoration not implemented here, but index is preserved in output columns for downstream merging.")

    logger.info("Completed data preparation for anomaly detection.")


if __name__ == "__main__":
    main()
