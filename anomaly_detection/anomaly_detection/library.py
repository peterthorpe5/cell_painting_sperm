#!/usr/bin/env python3
# coding: utf-8

"""
module lib for scripts. 
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
from pycytominer.aggregate import aggregate
from scipy.stats import shapiro



def harmonise_metadata_columns(df, logger=None):
    """
    Harmonise column names in a metadata DataFrame to ensure compatibility
    with downstream merging, regardless of their original names.
    """
    rename_dict = {}
    col_map = {col.lower(): col for col in df.columns}

    # Plate harmonisation
    plate_candidates = ["plate_metadata", "plate"]
    for platename in plate_candidates:
        if platename in col_map:
            rename_dict[col_map[platename]] = "Plate_Metadata"
            if logger:
                logger.info(f"Renaming '{col_map[platename]}' to 'Plate_Metadata'")
            break

    # Well harmonisation
    well_candidates = ["well_metadata", "well"]
    for wellname in well_candidates:
        if wellname in col_map:
            rename_dict[col_map[wellname]] = "Well_Metadata"
            if logger:
                logger.info(f"Renaming '{col_map[wellname]}' to 'Well_Metadata'")
            break

    # cpd_id harmonisation (including many possible variants)
    cpd_candidates = [
        "cpd_id", "compound_id", "comp_id", "compound", "compud_id",
        "compund_id", "compid", "comp", "compoundid"
    ]
    for cpdname in cpd_candidates:
        if cpdname in col_map:
            rename_dict[col_map[cpdname]] = "cpd_id"
            if logger:
                logger.info(f"Renaming '{col_map[cpdname]}' to 'cpd_id'")
            break

    # cpd_type harmonisation (similar variants possible)
    cpd_type_candidates = ["cpd_type", "compound_type", "type"]
    for cpdtype in cpd_type_candidates:
        if cpdtype in col_map:
            rename_dict[col_map[cpdtype]] = "cpd_type"
            if logger:
                logger.info(f"Renaming '{col_map[cpdtype]}' to 'cpd_type'")
            break

    # Library harmonisation (optional)
    library_candidates = ["library", "lib", "collection"]
    for lib in library_candidates:
        if lib in col_map:
            rename_dict[col_map[lib]] = "Library"
            if logger:
                logger.info(f"Renaming '{col_map[lib]}' to 'Library'")
            break

    if logger and not rename_dict:
        logger.info("No metadata columns required renaming.")

    return df.rename(columns=rename_dict)



def harmonise_cpd_type_column(df, id_col="cpd_id", cpd_type_col="cpd_type", logger=None):
    """
    Harmonise compound type labelling, renaming the original and creating a cleaned 'cpd_type' column.
    """
    # Rename original cpd_type column
    if cpd_type_col in df.columns:
        df = df.rename(columns={cpd_type_col: "cpd_type_raw"})
        if logger:
            logger.info(f"Renamed original '{cpd_type_col}' column to 'cpd_type_raw'.")

    # Fill with blanks if missing
    cpd_type_vals = df.get("cpd_type_raw", pd.Series([""]*len(df))).fillna("").str.lower()
    cpd_id_vals = df.get(id_col, pd.Series([""]*len(df))).fillna("").str.strip().str.upper()

    new_types = []
    for orig, cid in zip(cpd_type_vals, cpd_id_vals):
        if "positive control" in orig:
            new_types.append("positive control")
        elif "negative control" in orig or cid == "DMSO":
            new_types.append("DMSO")
        elif "compound" in orig:
            new_types.append("compound")
        else:
            new_types.append("compound")
    df["cpd_type"] = new_types

    if logger:
        logger.info("Harmonised compound type column added as 'cpd_type'.")

    return df


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


if __name__ == "__main__":
    main()
