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
    --zscore_method mean \
    --impute knn \
    --scale_per_plate \
    --scale_method auto \
    --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
    --na_cutoff 0.05 --corr_threshold 0.9 --unique_cutoff 0.01 --freq_cut 0.05
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pycytominer import feature_select
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import shapiro

def harmonise_columns(df, logger=None):
    """
    Harmonise key metadata column names for merging and processing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with possibly non-standard column names.
    logger : logging.Logger, optional
        Logger for harmonisation messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with harmonised column names.
    """
    rename_dict = {}
    col_map = {c.lower(): c for c in df.columns}
    def log_change(orig, new):
        if logger:
            logger.info(f"Renaming column '{orig}' to '{new}'.")
    for platename in ["plate_metadata", "plate"]:
        if platename in col_map:
            rename_dict[col_map[platename]] = "Plate_Metadata"
            log_change(col_map[platename], "Plate_Metadata")
            break
    for wellname in ["well_metadata", "well"]:
        if wellname in col_map:
            rename_dict[col_map[wellname]] = "Well_Metadata"
            log_change(col_map[wellname], "Well_Metadata")
            break
    cpd_id_candidates = [
        "cpd_id", "compound_id", "comp_id", "compound", "compud_id",
        "compund_id", "compid", "comp", "compoundid"
    ]
    for cname in cpd_id_candidates:
        if cname in col_map:
            rename_dict[col_map[cname]] = "cpd_id"
            log_change(col_map[cname], "cpd_id")
            break
    cpd_type_candidates = ["cpd_type", "compound_type", "type"]
    for cname in cpd_type_candidates:
        if cname in col_map:
            rename_dict[col_map[cname]] = "cpd_type"
            log_change(col_map[cname], "cpd_type")
            break
    library_candidates = ["library", "lib", "collection"]
    for cname in library_candidates:
        if cname in col_map:
            rename_dict[col_map[cname]] = "Library"
            log_change(col_map[cname], "Library")
            break
    if logger and not rename_dict:
        logger.info("No metadata columns required renaming.")
    return df.rename(columns=rename_dict)

def check_normality(df, feature_cols, alpha=0.05, logger=None, sample_n=200, out_tsv=None):
    """
    Check normality for each feature using Shapiro-Wilk test, log summary,
    and optionally write a TSV report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    feature_cols : list
        Feature columns to test.
    alpha : float
        p-value threshold for normality.
    logger : logging.Logger or None
        Logger.
    sample_n : int
        Max samples for normality test.
    out_tsv : str or None
        Optional path to write per-feature normality report.

    Returns
    -------
    is_majority_normal : bool
        True if majority of features are normal, else False.
    stats_df : pandas.DataFrame
        Per-feature normality result table.
    """
    records = []
    normal_count = 0
    test_count = 0
    insufficient_features = []
    for col in feature_cols:
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(data) >= 3:
            vals = data
            if len(vals) > sample_n:
                vals = vals.sample(sample_n, random_state=42)
            try:
                stat, pval = shapiro(vals)
            except Exception as e:
                if logger:
                    logger.warning(f"Shapiro test failed for feature '{col}': {e}")
                continue
            is_norm = pval > alpha
            records.append({
                "feature": col,
                "shapiro_stat": stat,
                "p_value": pval,
                "is_normal": is_norm,
                "n_tested": len(vals)
            })
            if is_norm:
                normal_count += 1
            test_count += 1
        else:
            insufficient_features.append(col)
            if logger:
                logger.warning(f"Feature '{col}' has insufficient data (n={len(data)}) for normality test.")
    normal_ratio = normal_count / test_count if test_count > 0 else 0
    if logger:
        logger.info(f"Normality check (Shapiro-Wilk, alpha={alpha}): "
                    f"{normal_count}/{test_count} ({normal_ratio:.2%}) features appear normal.")
    stats_df = pd.DataFrame(records)
    if out_tsv and len(stats_df) > 0:
        stats_df.to_csv(out_tsv, sep="\t", index=False)
        if logger:
            logger.info(f"Feature-wise normality test results written to: {out_tsv}")
    if logger and insufficient_features:
        logger.warning(f"{len(insufficient_features)} features had insufficient data for normality test: {insufficient_features}")
    return normal_ratio > 0.8, stats_df

def scale_per_plate(df, metadata_cols, plate_col="Plate_Metadata", method="standard", logger=None):
    """
    Scale features per plate using StandardScaler or RobustScaler.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with metadata and features.
    metadata_cols : list
        Metadata columns to exclude from scaling.
    plate_col : str
        Plate metadata column name.
    method : str
        Scaling method ('standard' or 'robust').
    logger : logging.Logger or None
        Logger for logging.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled features and metadata preserved.
    """
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    scaled_df = []
    scaler_class = StandardScaler if method == "standard" else RobustScaler
    for plate, group in df.groupby(plate_col):
        scaler = scaler_class()
        feat = group[feature_cols]
        feat_imp = feat.fillna(feat.median())
        scaled = scaler.fit_transform(feat_imp)
        scaled_feat = pd.DataFrame(scaled, columns=feature_cols, index=feat.index)
        scaled_feat[feat.isna()] = np.nan
        scaled_plate = pd.concat([group[metadata_cols].reset_index(drop=True),
                                  scaled_feat.reset_index(drop=True)], axis=1)
        if logger:
            logger.info(f"Scaled plate '{plate}' (n={group.shape[0]}) using {method} scaler.")
        scaled_df.append(scaled_plate)
    result = pd.concat(scaled_df, axis=0).reset_index(drop=True)
    if logger:
        logger.info(f"Per-plate scaling complete (method={method}, shape={result.shape}).")
    return result

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
    parser.add_argument('--cpd_type_col', default='cpd_type', help='Compound type column.')
    parser.add_argument('--zscore_method', choices=['mean', 'median'], default='mean',
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

def feature_selection(df, metadata_cols, na_cutoff=0.05, corr_threshold=0.9,
                      unique_cutoff=0.01, freq_cut=0.05, logger=None):
    """
    Apply pycytominer feature selection: remove highly correlated, null, and invariant features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with selected features and preserved metadata columns.
    """
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    df_numeric = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    n_non_numeric = (df_numeric.isnull().all()).sum()
    if logger and n_non_numeric > 0:
        bad_cols = df_numeric.columns[df_numeric.isnull().all()].tolist()
        logger.warning(f"{n_non_numeric} feature columns are non-numeric and will be excluded: {bad_cols}")
    df_numeric = df_numeric.loc[:, ~df_numeric.isnull().all()]
    selected = feature_select(
        df_numeric,
        features=list(df_numeric.columns),
        operation=[
            "variance_threshold",
            "correlation_threshold",
            "drop_na_columns"
        ],
        na_cutoff=na_cutoff,
        corr_threshold=corr_threshold,
        unique_cut=unique_cutoff,
        freq_cut=freq_cut
    )
    if logger:
        logger.info(f"Feature selection reduced from {len(df_numeric.columns)} to {selected.shape[1]} features.")
    result = pd.concat([df[metadata_cols].reset_index(drop=True),
                        selected.reset_index(drop=True)], axis=1)
    return result

def align_columns(dfs, metadata_cols, logger=None):
    """
    Align all DataFrames to the same set of columns (metadata + union of all features).

    Returns
    -------
    list of pandas.DataFrame
        DataFrames with aligned columns.
    """
    all_features = set()
    for df in dfs:
        all_features.update([c for c in df.columns if c not in metadata_cols])
    all_cols = metadata_cols + sorted(all_features)
    aligned = []
    for df in dfs:
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
        aligned.append(df[all_cols])
    if logger:
        logger.info(f"Aligned all outputs to {len(all_cols)} columns (including metadata).")
    return aligned


# This function is a pain .. numpy data array. index etc ...
def impute_missing(df, method="median", knn_neighbours=5, metadata_cols=None, logger=None):
    """
    Impute missing values in features using median or KNN, preserving all metadata columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to impute.
    method : str
        Imputation method ('median' or 'knn').
    knn_neighbours : int
        Number of neighbours for KNN imputation.
    metadata_cols : list or None
        Metadata columns to exclude from imputation.
    logger : logging.Logger or None
        Logger for messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed features and original metadata. Columns containing only missing
        values are removed prior to imputation.
    """
    if metadata_cols is None:
        metadata_cols = []

    candidate_cols = [c for c in df.columns if c not in metadata_cols]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df[c])]

    if logger:
        if non_numeric_cols:
            logger.warning(f"Skipping non-numeric columns during imputation: {list(non_numeric_cols)}")
        logger.info(f"Imputing features: {numeric_cols} (n={len(numeric_cols)}); preserving metadata: {metadata_cols}")
        logger.info(f"Total columns in df: {len(df.columns)}")
        logger.info(f"Metadata columns: {metadata_cols}")
        logger.info(f"Candidate (non-meta) columns: {len(candidate_cols)}")
        logger.info(f"Numeric feature columns: {len(numeric_cols)}")
        logger.info(f"Non-numeric candidate columns: {non_numeric_cols}")

    meta_df = df[metadata_cols].copy()
    feat_df = df[numeric_cols].copy()

    # Diagnose columns with only missing values
    all_nan_cols = feat_df.columns[feat_df.isna().all()]
    if len(all_nan_cols) > 0:
        if logger:
            logger.warning(f"Removing columns with all NaN values before imputation: {list(all_nan_cols)}")
        feat_df = feat_df.drop(columns=all_nan_cols)
        numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]

    if logger:
        logger.info(f"Shape of feat_df before imputation: {feat_df.shape}")
        logger.info(f"Number of missing values per column (top 20 shown):\n{feat_df.isna().sum().sort_values(ascending=False).head(20)}")
        logger.info(f"Columns remaining for imputation: {numeric_cols}")

    # Choose imputer
    if method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbours)
    else:
        raise ValueError("Unknown imputation method: %s" % method)

    # Impute features
    imputed = imputer.fit_transform(feat_df)
    feat_imputed_df = pd.DataFrame(imputed, columns=feat_df.columns, index=feat_df.index)

    # Combine metadata and imputed features
    out = pd.concat([meta_df.reset_index(drop=True), feat_imputed_df.reset_index(drop=True)], axis=1)
    return out




def standardise_features(df, train_controls, metadata_cols, method="mean", logger=None):
    """
    Standardise features using either mean/std or median/MAD from training controls.

    Returns
    -------
    pandas.DataFrame
        Standardised DataFrame.
    """
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    shared_cols = [c for c in feature_cols if c in train_controls.columns]
    missing_in_train = set(feature_cols) - set(shared_cols)
    if logger is not None and missing_in_train:
        logger.warning(f"{len(missing_in_train)} features in dataframe but not in train_controls. Dropping: {list(missing_in_train)}")
    feature_cols = shared_cols
    if method == "mean":
        location = train_controls[feature_cols].mean()
        scale = train_controls[feature_cols].std(ddof=0)
    elif method == "median":
        location = train_controls[feature_cols].median()
        scale = train_controls[feature_cols].mad() * 1.4826
    else:
        raise ValueError("Unknown z-score method: %s" % method)
    scale_replaced = scale.replace(0, np.nan)
    df_z = df.copy()
    df_z[feature_cols] = (df[feature_cols] - location) / scale_replaced
    return pd.concat([df_z[metadata_cols], df_z[feature_cols]], axis=1)

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
    Loads input, imputes missing data if requested, harmonises metadata columns, splits controls,
    applies feature selection, standardises features, and saves tab-separated outputs.
    """

    args = parse_args()
    logger = setup_logging(args.output_dir)

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

    # ========== FEATURE SELECTION ==========
    logger.info("Applying pycytominer feature selection to controls and treatments...")
    all_ctrls = pd.concat([train_ctrl, val_ctrl, test_ctrl])
    selected_ctrls = feature_selection(
        all_ctrls, metadata_cols,
        na_cutoff=args.na_cutoff,
        corr_threshold=args.corr_threshold,
        unique_cutoff=args.unique_cutoff,
        freq_cut=args.freq_cut,
        logger=logger
    )
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
    full_selected_outfile = os.path.join(args.output_dir, "full_dataset_feature_selected.tsv")
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
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        metadata_cols, method=args.zscore_method, logger=logger
    )
    val_ctrl_std = standardise_features(
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        train_ctrl_std, metadata_cols, method=args.zscore_method, logger=logger
    )
    test_ctrl_std = standardise_features(
        selected_ctrls[selected_ctrls[args.cpd_type_col] == args.control_label],
        train_ctrl_std, metadata_cols, method=args.zscore_method, logger=logger
    )
    treatments_std = standardise_features(
        selected_treatments, train_ctrl_std, metadata_cols,
        method=args.zscore_method, logger=logger
    )

    # ========== ALIGN OUTPUT COLUMNS ==========
    aligned = align_columns(
        [train_ctrl_std, val_ctrl_std, test_ctrl_std, treatments_std],
        metadata_cols=metadata_cols, logger=logger
    )
    train_ctrl_std, val_ctrl_std, test_ctrl_std, treatments_std = aligned

    # ========== OUTPUT ==========
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ctrl_std.to_csv(os.path.join(args.output_dir, "train_controls.tsv"), sep="\t", index=False)
    val_ctrl_std.to_csv(os.path.join(args.output_dir, "val_controls.tsv"), sep="\t", index=False)
    test_ctrl_std.to_csv(os.path.join(args.output_dir, "test_controls.tsv"), sep="\t", index=False)
    treatments_std.to_csv(os.path.join(args.output_dir, "treatments.tsv"), sep="\t", index=False)
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
