#!/usr/bin/env python3
# coding: utf-8

"""
merge_cellprofiler_with_metadata_featureselect.py

Merge raw CellProfiler per-object output with metadata (no aggregation), (no average per well!!)
impute missing data, apply optional per-plate scaling, and perform feature selection
(variance/correlation filter) using process_data.py functions.

Each object/cell is a row. All metadata is preserved, and a unique 'row_number' column can be added.

Requires:
    pandas, numpy, scikit-learn, scipy
    process_data.py in the same directory or PYTHONPATH

Example usage:
--------------
python impute...py \
    --input_dir ./raw_cp/ \
    --output_file combined_raw_profiles.tsv \
    --metadata_file plate_map.csv \
    --merge_keys Plate,Well \
    --impute knn \
    --knn_neighbours 5 \
    --scale_per_plate \
    --scale_method auto \
    --correlation_threshold 0.99 \
    --variance_threshold 0.05 \
    --sep '\t'
"""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import shapiro
from cell_painting.process_data import (
    standardise_metadata_columns,
    variance_threshold_selector,
    correlation_filter
)

def parse_args():
    """
    Parse command-line arguments for merging, imputing, scaling, and feature selection.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Merge CellProfiler per-object data with metadata, impute, scale, and feature-select (no aggregation).")
    parser.add_argument('--input_dir', required=True, help='Directory with CellProfiler CSV files.')
    parser.add_argument('--output_file', required=True, help='Output file (TSV/CSV).')
    parser.add_argument('--metadata_file', required=True, help='Metadata file (e.g. plate map).')
    parser.add_argument('--merge_keys', default='Plate,Well', help='Comma-separated metadata columns for merging (default: Plate,Well).')
    parser.add_argument('--impute', choices=['none', 'median', 'knn'], default='knn',
                        help='Impute missing values: "none", "median", or "knn" (default: knn).')
    parser.add_argument('--knn_neighbours', type=int, default=5,
                        help='Neighbours for KNN imputation (default: 5).')
    parser.add_argument('--scale_per_plate', action='store_true', help='Apply scaling per plate (default: False).')
    parser.add_argument('--scale_method', choices=['standard', 'robust', 'auto', 'none'], default='robust',
                        help='Scaling method: "standard", "robust", "auto", or "none" (default: robust).')
    parser.add_argument('--correlation_threshold', type=float, default=0.99,
                        help='Correlation threshold for filtering features (default: 0.99).')
    parser.add_argument('--variance_threshold', type=float, default=0.05,
                        help='Variance threshold for filtering features (default: 0.05).  Low-variance features are almost constant across all samples—they do not help distinguish between classes or clusters.')
    parser.add_argument('--sep', default='\t', help='Delimiter for output file (default: tab).')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set logging level.')
    return parser.parse_args()


def setup_logging(log_level="INFO"):
    """
    Set up logging to console.

    Parameters
    ----------
    log_level : str
        Logging level as a string ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns
    -------
    logging.Logger
        Configured logger object.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("merge_logger")


def harmonise_column_names(df, candidates, target, logger):
    """
    Harmonise column names in a DataFrame, renaming any candidate to the target if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    candidates : list of str
        Possible column names to look for.
    target : str
        Desired column name.
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    tuple
        (pandas.DataFrame with harmonised column name, str name of the harmonised column)
    """
    found = [c for c in candidates if c in df.columns]
    if not found:
        logger.warning(f"None of the candidate columns {candidates} found in DataFrame.")
        return df, None
    if target in df.columns and target not in found:
        logger.info(f"Target column '{target}' already present.")
        return df, target
    chosen = found[0]
    if chosen != target:
        logger.info(f"Renaming column '{chosen}' to '{target}'.")
        df = df.rename(columns={chosen: target})
    return df, target


def impute_missing(df, method="knn", knn_neighbours=5, logger=None):
    """
    Impute missing values for all numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numeric columns to impute.
    method : str
        Imputation method: 'median' or 'knn'.
    knn_neighbours : int
        Number of neighbours for KNN imputation.
    logger : logging.Logger or None
        Logger for status messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed numeric columns.
    """
    logger = logger or logging.getLogger("impute_logger")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.warning("No numeric columns found for imputation.")
        return df
    
    # Drop columns that are all-NaN
    nan_all = [col for col in numeric_cols if df[col].isnull().all()]
    if nan_all:
        logger.warning(f"Dropping {len(nan_all)} all-NaN columns before imputation: {nan_all}")
        df = df.drop(columns=nan_all)
        numeric_cols = [col for col in numeric_cols if col not in nan_all]

    before_na = df[numeric_cols].isna().sum().sum()
    logger.info(f"Imputing missing values in {len(numeric_cols)} numeric columns (nans before: {before_na})...")
    if method == "median":
        imp = SimpleImputer(strategy="median")
    elif method == "knn":
        imp = KNNImputer(n_neighbors=knn_neighbours)
    else:
        logger.info("Imputation skipped.")
        return df
    df[numeric_cols] = imp.fit_transform(df[numeric_cols])
    after_na = df[numeric_cols].isna().sum().sum()
    logger.info(f"Imputation complete (nans after: {after_na}).")
    return df


def auto_select_scaler(df, feature_cols, logger):
    """
    Choose robust or standard scaler based on Shapiro normality test.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features.
    feature_cols : list of str
        List of feature columns to test.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    str
        'standard' if most features are normal, else 'robust'.
    """
    stats = []
    for col in feature_cols:
        vals = df[col].dropna().values
        if len(vals) < 20:
            continue
        try:
            p = shapiro(vals)[1]
        except Exception:
            continue
        stats.append(p)
    normal_fraction = np.mean([p > 0.05 for p in stats]) if stats else 0
    scaler = "standard" if normal_fraction > 0.5 else "robust"
    logger.info(f"Normality fraction={normal_fraction:.2f}. Using scaler: {scaler}.")
    return scaler


def scale_per_plate(df, plate_col, method="auto", logger=None):
    """
    Scale numeric features per plate using the chosen method.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numeric features and plate metadata.
    plate_col : str
        Name of the plate metadata column.
    method : str
        Scaling method: 'standard', 'robust', or 'auto'.
    logger : logging.Logger or None
        Logger for status messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled numeric features.
    """
    logger = logger or logging.getLogger("scale_logger")
    plates = df[plate_col].unique()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Scaling per plate: {plates}. Method: {method}.")
    df_scaled = df.copy()
    for plate in plates:
        idx = df[plate_col] == plate
        plate_data = df.loc[idx, numeric_cols]
        if method == "auto":
            scaler_type = auto_select_scaler(plate_data, numeric_cols, logger)
        else:
            scaler_type = method
        scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        try:
            df_scaled.loc[idx, numeric_cols] = scaler.fit_transform(plate_data)
        except Exception as e:
            logger.warning(f"Scaling failed for plate {plate}: {e}")
    logger.info("Scaling complete.")
    return df_scaled


def main():
    """
    Main workflow to merge CellProfiler per-object data with metadata, impute, scale,
    and apply feature selection, outputting a final tab-separated table.

    Steps:
        1. Load and concatenate CellProfiler CSVs (per-object, no aggregation).
        2. Harmonise plate/well column names for merging.
        3. Load metadata file and harmonise column names.
        4. Merge metadata into per-object DataFrame.
        5. Add 'row_number' for row uniqueness.
        6. Impute missing values if requested.
        7. Optionally scale features per plate.
        8. Standardise metadata columns.
        9. Perform feature selection (correlation filter then variance threshold).
        10. Output feature-selected, metadata-enriched table.

    Returns
    -------
    None
    """
    args = parse_args()
    logger = setup_logging(args.log_level)
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Arguments: {' '.join(sys.argv)}")

    # 1. Load and concatenate CellProfiler CSVs
    input_dir = Path(args.input_dir)
   
    csv_files = [f for f in sorted(input_dir.glob("*.csv"))
             if f.name.lower() != "normalised.csv"]

    if not csv_files:
        logger.error(f"No CSV files found in {args.input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(csv_files)} CellProfiler CSV files: {[f.name for f in csv_files]}")
    dataframes = [pd.read_csv(f) for f in csv_files]
    cp_df = pd.concat(dataframes, axis=0, ignore_index=True)
    logger.info(f"Concatenated DataFrame shape: {cp_df.shape}")
    
    # Clean problematic values
    bad_strings = ['#NAME?', '#VALUE!', '#DIV/0!', 'N/A', 'NA', '', ' ']
    cp_df.replace(bad_strings, np.nan, inplace=True)
    logger.info("Replaced known bad strings with NaN.")

    # Drop columns that are entirely NaN
    na_cols = cp_df.columns[cp_df.isna().all()].tolist()
    if na_cols:
        logger.warning(f"Dropping {len(na_cols)} all-NaN columns: {na_cols}")
        cp_df = cp_df.drop(columns=na_cols)
    logger.info(f"DataFrame shape after dropping all-NaN columns: {cp_df.shape}")


    # 2. Harmonise plate/well columns
    merge_keys = [k.strip() for k in args.merge_keys.split(",")]
    plate_candidates = [merge_keys[0], "Plate", "plate", "Plate_Metadata", "Metadata_Plate", "Image_Metadata_Plate"]
    well_candidates = [merge_keys[1], "Well", "well", "Well_Metadata", "Metadata_Well", "Image_Metadata_Well"]
    cp_df, plate_col = harmonise_column_names(cp_df, plate_candidates, "Plate_Metadata", logger)
    cp_df, well_col = harmonise_column_names(cp_df, well_candidates, "Well_Metadata", logger)
    if plate_col is None or well_col is None:
        logger.error("Could not harmonise plate/well column names in CellProfiler data.")
        sys.exit(1)
    logger.info(f"CellProfiler: using plate column '{plate_col}', well column '{well_col}'.")

    # 3. Load and harmonise metadata
    meta_df = pd.read_csv(args.metadata_file)
    logger.info(f"Shape meta_df: {meta_df.shape}")
    meta_df, meta_plate_col = harmonise_column_names(meta_df, plate_candidates, plate_col, logger)
    meta_df, meta_well_col = harmonise_column_names(meta_df, well_candidates, well_col, logger)
    if meta_plate_col is None or meta_well_col is None:
        logger.error("Could not harmonise plate/well column names in metadata.")
        sys.exit(1)
    logger.info(f"Metadata: using plate column '{meta_plate_col}', well column '{meta_well_col}'.")

    # 4. Merge metadata
    merged_df = cp_df.merge(meta_df, how="left", left_on=[plate_col, well_col], right_on=[meta_plate_col, meta_well_col], suffixes=('', '_meta'))
    logger.info(f"Shape after metadata merge: {merged_df.shape}")
    # Drop columns that are all NA after merging metadata
    na_cols_postmerge = merged_df.columns[merged_df.isna().all()].tolist()
    if na_cols_postmerge:
        logger.warning(f"Dropping {len(na_cols_postmerge)} all-NaN columns after metadata merge: {na_cols_postmerge}")
        merged_df = merged_df.drop(columns=na_cols_postmerge)
    logger.info(f"Shape after dropping all-NaN columns post-merge: {merged_df.shape}")

    # Warn for missing metadata
    missing_meta = merged_df[[col for col in meta_df.columns if col not in [meta_plate_col, meta_well_col]]].isnull().all(axis=1).sum()
    if missing_meta > 0:
        logger.warning(f"{missing_meta} rows have missing metadata after merge.")

    # 5. Add row_number  - currently commented out. add if required, plus then alter the metadata_cols = [   ... too
    # merged_df.insert(0, "row_number", range(1, len(merged_df) + 1))

    # 6. Impute missing data (if requested)
    # Replace inf, -inf, and very large values with NaN before imputation
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(merged_df[numeric_cols]).sum().sum()
    n_neg_inf = np.isneginf(merged_df[numeric_cols]).sum().sum()
    n_large = (np.abs(merged_df[numeric_cols]) > 1e10).sum().sum()

    if n_inf or n_neg_inf or n_large:
        logger.warning(f"Replacing {n_inf} inf, {n_neg_inf} -inf, {n_large} very large values (>1e10) with NaN before imputation.")
        merged_df[numeric_cols] = merged_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        merged_df[numeric_cols] = merged_df[numeric_cols].applymap(lambda x: np.nan if isinstance(x, float) and abs(x) > 1e10 else x)

    # Drop columns that are entirely NaN
    na_cols = merged_df.columns[merged_df.isna().all()].tolist()

    if na_cols:
        logger.warning(f"Dropping {len(na_cols)} all-NaN columns: {na_cols}")
        merged_df = merged_df.drop(columns=na_cols)
    logger.info(f"DataFrame shape after dropping all-NaN columns: {cp_df.shape}")


    if args.impute != "none":
        merged_df = impute_missing(merged_df, method=args.impute, knn_neighbours=args.knn_neighbours, logger=logger)
    else:
        logger.info("Imputation skipped (impute=none).")

    # 7. Per-plate scaling (if requested)
    if args.scale_per_plate and args.scale_method != "none":
        feature_cols = [c for c in merged_df.select_dtypes(include=[np.number]).columns if c != "row_number"]
        merged_df = scale_per_plate(merged_df, plate_col=plate_col, method=args.scale_method, logger=logger)
    else:
        logger.info("Per-plate scaling not performed.")

    # 8. Standardise metadata columns for downstream compatibility
    merged_df = standardise_metadata_columns(merged_df, logger=logger, dataset_name="merged_raw")

    # 9. Preserve metadata columns 
    # row number removed for now, will add if needed downstream
    # metadata_cols = ["row_number", "cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]

    metadata_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    present_metadata = [col for col in metadata_cols if col in merged_df.columns]
    metadata_df = merged_df[present_metadata].copy()

    # 10. Feature selection (correlation and variance filtering)
    # Identify starting features
    feature_cols = [c for c in merged_df.columns if c not in present_metadata and pd.api.types.is_numeric_dtype(merged_df[c])]
    n_start_features = len(feature_cols)
    logger.info(f"Feature selection on {n_start_features} numeric columns.")
    logger.info(f"Shape before correlation_filter: {merged_df[feature_cols].shape}")

    # Correlation filter
    filtered_corr = correlation_filter(merged_df[feature_cols], threshold=args.correlation_threshold)
    n_after_corr = filtered_corr.shape[1]
    logger.info(f"Shape after correlation_filter: {filtered_corr.shape}")
    logger.info(f"Dropped {n_start_features - n_after_corr} features due to correlation threshold.")

    # Variance threshold filter
    filtered_final = variance_threshold_selector(filtered_corr, threshold=args.variance_threshold)
    n_after_var = filtered_final.shape[1]
    logger.info(f"Shape after variance_threshold_selector: {filtered_final.shape}")
    logger.info(f"Dropped {n_after_corr - n_after_var} features due to low variance.")
    logger.info(f"Feature selection retained {n_after_var} features.")


    # 11. Combine metadata and filtered features
    final_df = pd.concat([metadata_df, filtered_final], axis=1)


    # 12. Output
    # Drop duplicate columns if present (keep first occurrence)
    dupe_cols = final_df.columns.duplicated()
    if any(dupe_cols):
        dup_names = final_df.columns[dupe_cols].tolist()
        logger.warning(f"Found and dropping duplicate columns in output: {dup_names}")
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    final_df.to_csv(args.output_file, sep=args.sep, index=False)
    logger.info(f"Saved feature-selected data to {args.output_file} (shape: {final_df.shape})")
    logger.info("Merge complete.")

if __name__ == "__main__":
    main()
