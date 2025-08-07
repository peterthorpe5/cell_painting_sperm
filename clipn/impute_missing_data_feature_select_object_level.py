#!/usr/bin/env python3
# coding: utf-8

"""
merge_cellprofiler_with_metadata_featureselect.py

Prepare CellProfiler per-object data for analysis with optional well-level aggregation.

This script merges raw CellProfiler per-object (single cell) output files with metadata (e.g., plate map),
imputes missing data, applies optional per-plate feature scaling, and performs feature selection using
variance and correlation filters. Optionally, it can aggregate features to the well level by taking the median
of all objects within each well.

Workflow:
---------
1. Load all CellProfiler per-object CSV or CSV.GZ files from a specified directory,
   excluding files with 'image' or 'normalised' in the filename.
2. Merge with metadata (e.g., plate map) using harmonised plate and well columns.
3. Clean and standardise metadata columns (cpd_id, cpd_type, Library).
4. Impute missing data using KNN or median (or skip, as specified).
5. Optionally normalise features to the DMSO control (robust z-score).
6. Optionally scale features per plate.
7. Perform feature selection by correlation and variance thresholding.
8. Output the final table:
   - By default, one row per object (cell).
   - If --aggregate_per_well is set, aggregate features to the median per well.

Inputs:
-------
- --input_dir:        Directory containing CellProfiler per-object CSV or CSV.GZ files.
- --metadata_file:    CSV/TSV file with plate and well metadata (e.g., plate map).
- --output_file:      Path to output file (tab- or comma-separated).
- --merge_keys:       Comma-separated list of plate and well column names for merging (default: Plate,Well).
- --impute:           Missing value imputation method: none, median, or knn.
- --knn_neighbours:   Number of neighbours for KNN imputation (default: 5).
- --scale_per_plate:  Flag to apply scaling per plate (default: off).
- --scale_method:     Scaling method: standard, robust, auto, or none.
- --no_dmso_normalisation:  Flag to skip DMSO normalisation (default: ON).
- --correlation_threshold:  Correlation threshold for feature filtering (default: 0.99).
- --variance_threshold:     Variance threshold for feature filtering (default: 0.05).
- --aggregate_per_well:     If set, output one row per well (median of all objects in the well).
- --sep:              Output file delimiter (default: tab).
- --log_level:        Logging level.

Requirements:
-------------
- Python 3.7+
- pandas, numpy, scikit-learn, scipy, psutil
- cell_painting.process_data.py with required helper functions (in PYTHONPATH)

Example usage:
--------------
# Per-object output (default)
python merge_cellprofiler_with_metadata_featureselect.py \
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

# Median-aggregated per-well output
python merge_cellprofiler_with_metadata_featureselect.py \
    --input_dir ./raw_cp/ \
    --output_file median_per_well.tsv \
    --metadata_file plate_map.csv \
    --aggregate_per_well \
    --sep '\t'

Author: Pete Thorpe, 2025

"""

import argparse
import logging
import os
import psutil
import time
from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import shapiro
import gc
from cell_painting.process_data import (
    standardise_metadata_columns,
    variance_threshold_selector,
    correlation_filter)

_script_start_time = time.time()

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
    parser.add_argument('--no_dmso_normalisation',
                        action='store_true',
                        help='If set, do not normalise each feature to the median of DMSO wells (default: normalisation ON).')

    parser.add_argument('--correlation_threshold', type=float, default=0.9,
                        help='Correlation threshold for filtering features (default: 0.9).')
    parser.add_argument('--variance_threshold', type=float, default=0.1,
                        help='Variance threshold for filtering features (default: 0.1).  Low-variance features are almost constant across all samples—they do not help distinguish between classes or clusters.')
    parser.add_argument('--library', type=str, default=None,
                    help="Value to add as 'Library' column if not present in the metadata. "
                         "If omitted and column missing, script will error.")
    parser.add_argument('--sep', default='\t', help='Delimiter for output file (default: tab).')
    parser.add_argument('--aggregate_per_well',
                        action='store_true',
                        help='If set, output one row per well (median of all objects in the well). Default: off (per-object output).'
                        )
    parser.add_argument('--per_well_output_file', type=str, default=None,
                        help="Optional output file for per-well aggregated data (default: auto-named from --output_file)")

    parser.add_argument('--no_compress_output',
                        action='store_true',
                        help='If set, do NOT compress the output file. Default: output will be compressed (.gz if filename ends with .gz).')

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


def robust_read_csv(path, logger=None, n_check_lines=2):
    """
    Robustly read a CSV/TSV file, auto-detecting comma or tab delimiter.
    Tries comma first; if fails or tab detected in header, tries tab.
    If both fail, raises the last error.

    Parameters
    ----------
    path : str
        File path to read.
    logger : logging.Logger, optional
        Logger for status messages.
    n_check_lines : int
        Number of lines to check for tab presence.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    Exception
        If neither CSV nor TSV parsing succeeds.
    """
    logger = logger or logging.getLogger("robust_csv")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [next(f) for _ in range(n_check_lines)]
    # Check for tab character in the first few lines
    if any('\t' in line for line in lines):
        logger.info("Detected tab character in header; reading as TSV.")
        return pd.read_csv(path, sep='\t')
    # Try comma first, fall back to tab if any failure
    try:
        df = pd.read_csv(path)
        # Extra check: If only one column and there are tabs in header, treat as TSV
        if df.shape[1] == 1 and '\t' in lines[0]:
            logger.info("File appears to be TSV but read as CSV; retrying as TSV.")
            return pd.read_csv(path, sep='\t')
        logger.info("Read file as comma-separated.")
        return df
    except Exception as e:
        logger.warning(f"Reading as CSV failed ({e}); retrying as TSV.")
        try:
            df = pd.read_csv(path, sep='\t')
            logger.info("Successfully read file as tab-separated after CSV failed.")
            return df
        except Exception as e2:
            logger.error(f"Reading as TSV also failed: {e2}")
            raise e2  # Reraise the last exception



def log_memory_usage(logger, prefix="", extra_msg=None):
    """
    Log the current and peak memory usage (RAM) of the running process.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    prefix : str
        Optional prefix for the log message.
    extra_msg : str or None
        Optional extra string to log.
    """
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    peak_gb = None
    # Try to get max RSS if possible (platform-dependent)
    try:
        # On Linux, 'peak_wset' or 'peak_rss' in memory_info(). Not always available.
        if hasattr(process, "memory_info"):
            # peak is not available via psutil directly, only via resource module
            import resource
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.uname().sysname == "Linux":
                peak_gb = peak_rss / (1024 ** 2)  # kilobytes -> GB
            else:
                peak_gb = peak_rss / (1024 ** 3)  # bytes -> GB (Mac/others)
    except Exception:
        pass
    elapsed = time.time() - _script_start_time
    msg = f"{prefix} Memory usage: {mem_gb:.2f} GB (resident set size)"
    if peak_gb:
        msg += f", Peak: {peak_gb:.2f} GB"
    msg += f", Elapsed: {elapsed/60:.1f} min"
    if extra_msg:
        msg += " | " + extra_msg
    logger.info(msg)


def add_plate_well_metadata_if_missing(obj_df, input_dir, obj_filename, logger):
    """
    Add plate/well metadata to object-level DataFrame by merging with matching image-level file if needed.
    
    Parameters
    ----------
    obj_df : pd.DataFrame
        The loaded object-level DataFrame.
    input_dir : pathlib.Path
        Path object of the directory containing the files.
    obj_filename : str
        Name of the object-level file (e.g., HepG2CP_Cell.csv).
    logger : logging.Logger
        Logger object.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with plate/well columns if possible.
    """
    plate_candidates = ["Plate", "Plate_Metadata", "Metadata_Plate", "Image_Metadata_Plate"]
    well_candidates = ["Well", "Well_Metadata", "Metadata_Well", "Image_Metadata_Well"]

    # Check if plate/well present
    has_plate = any(col in obj_df.columns for col in plate_candidates)
    has_well = any(col in obj_df.columns for col in well_candidates)
    if has_plate and has_well:
        logger.info(f"Plate/Well metadata already present in {obj_filename}")
        return obj_df

    # Try to find image-level file
    base = obj_filename
    for suffix in ["_Cell", "_Cytoplasm", "_Nuclei"]:
        base = base.replace(suffix, "")
    for suffix in [".csv", ".csv.gz"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    image_file_candidates = [f"{base}_Image.csv", f"{base}_Image.csv.gz"]
    image_file = None
    for candidate in image_file_candidates:
        path = input_dir / candidate
        if path.exists():
            image_file = path
            break
    if image_file is None:
        logger.error(
            f"Plate/Well metadata missing and no matching image-level file found for {obj_filename}. "
            f"Tried: {image_file_candidates}"
        )
        return obj_df  # Let downstream checks error out, or you can choose to sys.exit(1)

    logger.info(f"Merging {obj_filename} with {image_file.name} to add plate/well metadata.")
    img_df = pd.read_csv(image_file)
    # Defensive: Check for ImageNumber in both
    if "ImageNumber" not in obj_df.columns or "ImageNumber" not in img_df.columns:
        logger.error(f"Cannot merge {obj_filename} and {image_file.name}: 'ImageNumber' not present in both files.")
        return obj_df

    # Merge on ImageNumber
    merged = obj_df.merge(img_df, on="ImageNumber", suffixes=("", "_img"), how="left")
    logger.info(f"Merged {obj_filename} with {image_file.name}; resulting shape: {merged.shape}")

    # Check again for plate/well
    has_plate = any(col in merged.columns for col in plate_candidates)
    has_well = any(col in merged.columns for col in well_candidates)
    if not (has_plate and has_well):
        logger.error(
            f"Even after merging with image-level file, could not find both plate and well columns in {obj_filename}."
        )
    return merged


def standardise_well_name(well):
    """
    Convert well names to zero-padded format (e.g., A1 -> A01, B9 -> B09, H12 -> H12).
    
    Parameters
    ----------
    well : str
        Well name (e.g., 'A1', 'B09', 'H12').

    Returns
    -------
    str
        Zero-padded well name (e.g., 'A01', 'B09', 'H12'). 
        If input does not match expected pattern, original value is returned.
    """
    if isinstance(well, str):
        match = re.match(r"^([A-Ha-h])(\d{1,2})$", well.strip())
        if match:
            row = match.group(1).upper()
            col = int(match.group(2))
            return f"{row}{col:02d}"
    return well


def normalise_to_dmso(df, feature_cols, metadata_col='cpd_type', dmso_label='dmso', logger=None):
    """
    Robust Z-score normalisation using DMSO controls:

    
    robust-z =  (feature value - DMSO median) / DMSO MAD

    where MAD is:  abs(x - median_DMSO), then the median of all of these. 
    

    --no_dmso_normalisation : bool, optional  
    If set, do not normalise features to the median of DMSO wells per plate (default: False, normalisation ON).


    For each plate and feature:
        - Subtract the median of DMSO wells (per plate, per feature).
        - Divide by the median absolute deviation (MAD) of DMSO wells.
        - If MAD is zero, skip scaling for that feature/plate.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and metadata.
    feature_cols : list of str
        Names of feature columns to normalise.
    metadata_col : str, optional
        Metadata column indicating compound type (default: 'cpd_type').
    dmso_label : str, optional
        Value indicating DMSO wells (default: 'dmso', case-insensitive).
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with features robustly normalised to DMSO (per plate).
    """
    logger = logger or logging.getLogger("dmso_norm")
    plate_col = 'Plate_Metadata'
    if plate_col not in df.columns or metadata_col not in df.columns:
        logger.error(f"Required columns '{plate_col}' or '{metadata_col}' missing.")
        raise ValueError(f"Required columns '{plate_col}' or '{metadata_col}' missing.")
    plates = df[plate_col].unique()
    df_norm = df.copy()
    for plate in plates:
        idx_plate = df[plate_col] == plate
        idx_dmso = idx_plate & (df[metadata_col].str.lower() == dmso_label.lower())
        if idx_dmso.sum() == 0:
            logger.warning(f"No DMSO wells found for plate {plate}. Skipping DMSO normalisation for this plate.")
            continue
        dmso_median = df.loc[idx_dmso, feature_cols].median()
        # dmso_mad = df.loc[idx_dmso, feature_cols].mad() -  not in this version of pandas
        dmso_mad = df.loc[idx_dmso, feature_cols].apply(lambda x: np.median(np.abs(x - np.median(x))), axis=0)

        # Avoid division by zero or near-zero MAD: only scale if MAD > 0
        mad_zero = dmso_mad == 0
        if mad_zero.any():
            zero_cols = dmso_mad.index[mad_zero].tolist()
            logger.warning(f"MAD=0 for plate {plate}, features: {zero_cols}. Skipping scaling for these features (will only centre).")
        # Robust z-score: (value - DMSO_median) / DMSO_mad
        for feature in feature_cols:
            vals = df.loc[idx_plate, feature]
            # Subtract median
            vals = vals - dmso_median[feature]
            # Divide by MAD (if not zero)
            if dmso_mad[feature] > 0:
                vals = vals / dmso_mad[feature]
            if not np.issubdtype(df_norm[feature].dtype, np.floating):
                df_norm[feature] = df_norm[feature].astype(np.float32)
            # Assign back
            df_norm.loc[idx_plate, feature] = vals.astype(np.float32)
        logger.info(f"Robust DMSO normalisation complete for plate {plate}.")
    return df_norm



def harmonise_column_names(df, candidates, target, logger):
    """
    Harmonise column names in a DataFrame, renaming a single unambiguous candidate to the target if needed.
    If multiple candidates are found, abort and force user to fix input.

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
        (pandas.DataFrame with harmonised column name, str name of the harmonised column or None)

    Raises
    ------
    ValueError
        If multiple candidates for the column are found, or if a conflict exists with the target.
    """
    found = [c for c in candidates if c in df.columns]
    if len(found) == 0:
        logger.warning(f"No candidate columns for '{target}' found in DataFrame: {candidates}")
        return df, None
    if len(found) > 1:
        logger.error(
            f"Ambiguous candidates for '{target}' found: {found}. Please ensure only one exists and rerun."
        )
        raise ValueError(
            f"Ambiguous candidates for '{target}' found: {found}. "
            "Please ensure only one exists in your input file and rerun the script."
        )
    chosen = found[0]
    if chosen == target:
        logger.info(f"Target column '{target}' already present.")
        return df, target
    # Do not rename if target already exists
    if target in df.columns:
        logger.error(
            f"Target column '{target}' already exists alongside candidate '{chosen}'. "
            "No renaming performed. Please resolve this conflict."
        )
        raise ValueError(
            f"Target column '{target}' already exists alongside candidate '{chosen}'. "
            "Please resolve this conflict in your input file."
        )
    logger.info(f"Renaming column '{chosen}' to '{target}'.")
    df = df.rename(columns={chosen: target})
    return df, target



def impute_missing(df, method="knn", knn_neighbours=5, logger=None,
                   max_cells=1000000, max_features=3000):
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
    max_cells : int, optional
        Maximum number of rows for which to attempt imputation.
    max_features : int, optional
        Maximum number of columns for which to attempt imputation.
    logger : logging.Logger or None
        Logger for status messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed numeric columns.
    """
    logger = logger or logging.getLogger("impute_logger")
    n_cells, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if n_cells > max_cells or len(numeric_cols) > max_features:
        logger.warning(
            f"Imputation skipped: dataset too large for safe in-memory imputation "
            f"({n_cells:,} rows, {len(numeric_cols)} numeric columns; "
            f"thresholds: {max_cells:,} rows, {max_features} columns)."
        )
        return df

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
        imputer = SimpleImputer(strategy="median")
    elif method == "knn":
        logger.info("Running KNN imputation (may be memory-intensive).")
        imputer = KNNImputer(n_neighbors=knn_neighbours)
    else:
        logger.info("Imputation skipped.")
        return df
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols].astype(np.float32))
    after_na = df[numeric_cols].isna().sum().sum()
    gc.collect()
    logger.info(f"Imputation complete (nans after: {after_na}).")
    return df


def clean_metadata_columns(df, logger=None):
    """
    Standardise Cell Painting metadata columns (cpd_id, cpd_type, Library).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned metadata columns.
    """
    # Standardise cpd_type values
    cpd_type_map = {
        "positive controls (sperm painting)": "positive_control",
        "negative control (DMSO)": "DMSO",
    }

    for col in ["cpd_id", "cpd_type", "Library"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].eq("") | df[col].isna(), col] = pd.NA

    if "cpd_type" in df.columns:
        df["cpd_type"] = df["cpd_type"].replace(cpd_type_map)
        # Set all cpd_type that are not DMSO or positive_control (case insensitive) to 'compound'
        mask_not_control = ~df["cpd_type"].isin(["DMSO", "positive_control"])
        mask_has_cpd_id = df["cpd_id"].notna() & (df["cpd_id"] != "DMSO")
        df.loc[mask_not_control & mask_has_cpd_id, "cpd_type"] = "compound"
        if logger:
            logger.info("Standardised cpd_type values.")

    # Set Library column
    if "Library" in df.columns and "cpd_id" in df.columns:
        mask_dmso = (df["cpd_id"] == "DMSO") | (df["cpd_type"] == "DMSO")
        mask_has_cpd_id = df["cpd_id"].notna() & (df["cpd_id"] != "DMSO")
        df.loc[mask_dmso, "Library"] = "control"
        df.loc[mask_has_cpd_id & ~mask_dmso, "Library"] = "compound"
        if logger:
            n_dmso = mask_dmso.sum()
            n_compound = (mask_has_cpd_id & ~mask_dmso).sum()
            logger.info(f"Set Library='control' for {n_dmso} rows and 'compound' for {n_compound} rows.")

    return df

import logging

def find_merge_columns(feature_cols, meta_cols, candidates=None):
    """
    Auto-detects common merge columns (flexible to case/underscore variations).

    Args:
        feature_cols (list): Columns from CellProfiler features file.
        meta_cols (list): Columns from metadata file.
        candidates (list): List of preferred merge column names (case/underscore insensitive).

    Returns:
        dict: Mapping of {standard_name: (feature_col, meta_col)} for merging.
    """
    # Standard candidates
    if candidates is None:
        candidates = ['ImageNumber', 'Plate', 'Well', 'Field', 'Image_Metadata_Well', 'Image_Metadata_Plate']

    found = {}
    for cand in candidates:
        for fcol in feature_cols:
            if cand.replace('_','').lower() == fcol.replace('_','').lower():
                for mcol in meta_cols:
                    if cand.replace('_','').lower() == mcol.replace('_','').lower():
                        found[cand] = (fcol, mcol)
    return found

def robust_merge_features_and_metadata(features_df, meta_df, logger=None):
    """
    Merge CellProfiler features with metadata, logging all steps and mismatches.

    Args:
        features_df (pd.DataFrame): CellProfiler per-object or per-image data.
        meta_df (pd.DataFrame): Metadata DataFrame.
        logger (logging.Logger): Optional logger.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    fcols = list(features_df.columns)
    mcols = list(meta_df.columns)
    colmap = find_merge_columns(fcols, mcols)

    if logger:
        logger.info(f"Candidate merge columns: {colmap}")

    if not colmap:
        msg = ("No matching merge columns found! "
               f"Features cols: {fcols[:10]}..., Metadata cols: {mcols[:10]}...")
        if logger:
            logger.error(msg)
        else:
            print(msg)
        # Return unmerged, but user can debug further
        return features_df

    # Pick the *first* matching merge column for now
    merge_on = list(colmap.values())[0]
    fcol, mcol = merge_on

    if logger:
        logger.info(f"Merging on: Feature col '{fcol}', Metadata col '{mcol}'")

    merged = features_df.merge(meta_df, left_on=fcol, right_on=mcol, how='left')

    if logger:
        logger.info(f"Merged shape: {merged.shape}, Unmatched rows: {(merged[mcol].isna()).sum()}")
    return merged


def harmonise_metadata_columns(df, logger=None, is_metadata_file=False):
    """
    Harmonise plate and well column names in the provided DataFrame to 'Plate_Metadata' and 'Well_Metadata'.

    Args:
        df (pd.DataFrame): Input DataFrame (annotation or features).
        logger (logging.Logger, optional): Logger for information and warnings.
        is_metadata_file (bool): If True, use broader candidate set for annotation file.

    Returns:
        pd.DataFrame: DataFrame with harmonised column names.
    """
    plate_candidates = [
        "Plate_Metadata", "Plate", "Barcode", "plate", "Metadata_Plate", "Image_Metadata_Plate"
    ] if is_metadata_file else [
        "Plate_Metadata", "Plate", "plate", "Metadata_Plate", "Image_Metadata_Plate"
    ]
    well_candidates = [
        "Well_Metadata", "Well", "well", "Metadata_Well", "Image_Metadata_Well"
    ]
    # Plate harmonisation
    plate_col = next((col for col in plate_candidates if col in df.columns), None)
    if plate_col and plate_col != "Plate_Metadata":
        df = df.rename(columns={plate_col: "Plate_Metadata"})
        if logger:
            logger.info(f"Renamed column '{plate_col}' to 'Plate_Metadata'.")
    elif not plate_col:
        if logger:
            logger.warning(f"None of the candidate plate columns {plate_candidates} found in DataFrame.")
    # Well harmonisation
    well_col = next((col for col in well_candidates if col in df.columns), None)
    if well_col and well_col != "Well_Metadata":
        df = df.rename(columns={well_col: "Well_Metadata"})
        if logger:
            logger.info(f"Renamed column '{well_col}' to 'Well_Metadata'.")
    elif not well_col:
        if logger:
            logger.warning(f"None of the candidate well columns {well_candidates} found in DataFrame.")
    # Final check
    if "Plate_Metadata" not in df.columns or "Well_Metadata" not in df.columns:
        if logger:
            logger.error("Could not harmonise plate/well column names in metadata.")
        raise ValueError("Could not harmonise plate/well column names in metadata.")
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

def infer_dtypes(filename, nrows=1000):
    """
    Infer optimal dtypes for reading a CellProfiler CSV.
    Floats become float32, metadata to 'category'.

    Parameters
    ----------
    filename : str or Path
        CSV file path.
    nrows : int
        Number of rows to peek for type guessing.

    Returns
    -------
    dict
        Dictionary of column:dtype for use with pd.read_csv(dtype=...).
    """
    tmp = pd.read_csv(filename, nrows=nrows)
    dtypes = {}
    for col in tmp.columns:
        if tmp[col].dtype == float:
            dtypes[col] = 'float32'
        elif any(x in col.lower() for x in ['plate', 'well', 'cpd', 'type', 'library']):
            dtypes[col] = 'category'
    return dtypes


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
            df_scaled.loc[idx, numeric_cols] = scaler.fit_transform(plate_data).astype(np.float32)

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
    log_memory_usage(logger, prefix=" START ")

    # 1. Load and concatenate CellProfiler CSVs
    input_dir = Path(args.input_dir)

   # 1. Load and concatenate CellProfiler CSVs (.csv and .csv.gz), exclude "image" and "normalised" files
    all_files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.csv.gz")))

    # Exclude files with 'image' or 'normalised' in filename (case-insensitive)
    csv_files = [
        f for f in all_files
        if "image" not in f.name.lower() and "normalised" not in f.name.lower()
    ]
    excluded_files = [f.name for f in all_files if f not in csv_files]
    logger.info(f"Looking for CellProfiler CSV files in {input_dir}...")
    logger.info("Excluding files with 'image' or 'normalised' in their names.")

    

    if not csv_files:
        logger.error(f"No CellProfiler CSV files found in {args.input_dir} after exclusions.")
        sys.exit(1)
    if excluded_files:
        logger.info(f"Excluded files (contain 'image' or 'normalised'): {excluded_files}")
    logger.info(f"Files to load: {[f.name for f in csv_files]}")

    # Read each file (compressed or not), and add plate/well metadata if missing
    dataframes = []

    for f in csv_files:
        logger.info(f"Reading file: {f.name}")
        dtype_map = infer_dtypes(f)
        df = pd.read_csv(f, dtype=dtype_map)
        df = add_plate_well_metadata_if_missing(df, input_dir, f.name, logger)
        log_memory_usage(logger, prefix=f" [After {f.name} loaded] ")
        dataframes.append(df)
        # Free temp DataFrame -  too much RAM being used
        del df
        gc.collect()
        log_memory_usage(logger, prefix=f" [After {f.name} loaded garbage cleaned] ")



    cp_df = pd.concat(dataframes, axis=0, ignore_index=True)
    log_memory_usage(logger, prefix=" [After datasets loaded ]")
    logger.info(f"Concatenated DataFrame shape: {cp_df.shape}")
    # Free up memory from the per-file DataFrames list
    del dataframes
    gc.collect()
    log_memory_usage(logger, prefix=" [After datasets loaded garbage cleaned]")

    # Downcast floats to float32 after concat to ensure no upcasting
    float_cols = cp_df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        cp_df[float_cols] = cp_df[float_cols].astype('float32')
        logger.info(f"Downcast {len(float_cols)} float64 columns to float32 after concat.")
    gc.collect()


    # Convert suitable object columns to category
    object_cols = cp_df.select_dtypes(include=['object']).columns
    cat_candidates = [col for col in object_cols if any(x in col.lower() for x in ['plate', 'well', 'cpd', 'type', 'library'])]
    for col in cat_candidates:
        cp_df[col] = cp_df[col].astype('category')
    logger.info(f"Converted {len(cat_candidates)} object columns to category.")

    gc.collect()


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
    meta_df = robust_read_csv(args.metadata_file, logger=logger)
    meta_df = harmonise_metadata_columns(meta_df, logger, is_metadata_file=True)
    meta_df = clean_metadata_columns(meta_df, logger)
    logger.info(f"Shape meta_df: {meta_df.shape}")

    # Downcast float columns to float32 in meta_df
    meta_float_cols = meta_df.select_dtypes(include=['float64']).columns
    if len(meta_float_cols) > 0:
        meta_df[meta_float_cols] = meta_df[meta_float_cols].astype('float32')
        logger.info(f"Downcast {len(meta_float_cols)} float64 columns to float32 in meta_df.")

    # Convert suitable object columns to category in meta_df
    meta_object_cols = meta_df.select_dtypes(include=['object']).columns
    meta_cat_candidates = [col for col in meta_object_cols if any(x in col.lower() for x in ['plate', 'well', 'cpd', 'type', 'library'])]
    for col in meta_cat_candidates:
        meta_df[col] = meta_df[col].astype('category')
    logger.info(f"Converted {len(meta_cat_candidates)} object columns to category in meta_df.")

    gc.collect()


    if "Library" not in meta_df.columns:
        if args.library is not None:
            meta_df["Library"] = args.library
            logger.info(f"'Library' column not found in metadata. Added with value: {args.library}")
        else:
            logger.error(
                "'Library' column missing from metadata and --library not provided. "
                "Please specify --library if you want to add this column.")
            sys.exit(1)
    else:
        logger.info("'Library' column found in metadata.")


    meta_df, meta_plate_col = harmonise_column_names(meta_df, plate_candidates, plate_col, logger)
    meta_df, meta_well_col = harmonise_column_names(meta_df, well_candidates, well_col, logger)
    if meta_plate_col is None or meta_well_col is None:
        logger.error("Could not harmonise plate/well column names in metadata.")
        sys.exit(1)
    logger.info(f"Metadata: using plate column '{meta_plate_col}', well column '{meta_well_col}'.")

    logger.info(f"CellProfiler unique plate values: {sorted(cp_df[plate_col].unique())[:10]}")
    logger.info(f"Metadata unique plate values: {sorted(meta_df[meta_plate_col].unique())[:10]}")
    logger.info(f"CellProfiler unique well values: {sorted(cp_df[well_col].unique())[:10]}")
    logger.info(f"Metadata unique well values: {sorted(meta_df[meta_well_col].unique())[:10]}")


    # Standardise Well_Metadata in both main and metadata DataFrames
    cp_df['Well_Metadata'] = cp_df['Well_Metadata'].apply(standardise_well_name)
    meta_df['Well_Metadata'] = meta_df['Well_Metadata'].apply(standardise_well_name)
    logger.info("Standardised Well_Metadata in both CellProfiler and metadata DataFrames.")

    # Standardise Well_Metadata (or 'Well', if not renamed yet) in metadata
    if 'Well_Metadata' in meta_df.columns:
        meta_df['Well_Metadata'] = meta_df['Well_Metadata'].apply(standardise_well_name)
    elif 'Well' in meta_df.columns:
        meta_df['Well'] = meta_df['Well'].apply(standardise_well_name)

    logger.info(f"CellProfiler unique plate values: {sorted(cp_df[plate_col].unique())[:10]}")
    logger.info(f"Metadata unique plate values: {sorted(meta_df[meta_plate_col].unique())[:10]}")
    logger.info(f"CellProfiler unique well values: {sorted(cp_df[well_col].unique())[:10]}")
    logger.info(f"Metadata unique well values: {sorted(meta_df[meta_well_col].unique())[:10]}")

    # 4. Merge metadata
    merged_df = cp_df.merge(meta_df, how="left", 
                            left_on=[plate_col, well_col], 
                            right_on=[meta_plate_col, meta_well_col], suffixes=('', '_meta'))
    logger.info(f"Shape after metadata merge: {merged_df.shape}")


    # Drop columns that are all NA after merging metadata
    na_cols_postmerge = merged_df.columns[merged_df.isna().all()].tolist()
    if na_cols_postmerge:
        logger.warning(f"Dropping {len(na_cols_postmerge)} all-NaN columns after metadata merge: {na_cols_postmerge}")
        merged_df = merged_df.drop(columns=na_cols_postmerge)
    logger.info(f"Shape after dropping all-NaN columns post-merge: {merged_df.shape}")

    # Warn for missing metadata
    meta_cols_to_check = [col for col in meta_df.columns if col not in [meta_plate_col, meta_well_col] and col in merged_df.columns]
    if meta_cols_to_check:
        missing_meta = merged_df[meta_cols_to_check].isnull().all(axis=1).sum()
        if missing_meta > 0:
            logger.warning(f"{missing_meta} rows have missing metadata after merge.")
    else:
        logger.warning("No additional metadata columns found in merged_df for missing metadata check.")

    # Free memory from cp_df and meta_df
    del cp_df
    del meta_df
    gc.collect()
    log_memory_usage(logger, prefix=" After merge ")

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
        # Convert to float first to ensure .abs() works as expected (if needed)
        merged_df[numeric_cols] = merged_df[numeric_cols].astype(np.float32)
        mask = merged_df[numeric_cols].abs() > 1e10
        merged_df[numeric_cols] = merged_df[numeric_cols].where(~mask, np.nan)


    # Drop columns that are entirely NaN
    na_cols = merged_df.columns[merged_df.isna().all()].tolist()

    if na_cols:
        logger.warning(f"Dropping {len(na_cols)} all-NaN columns: {na_cols}")
        merged_df = merged_df.drop(columns=na_cols)
    logger.info(f"DataFrame shape after dropping all-NaN columns: {merged_df.shape}")



    if args.impute != "none":
        merged_df = impute_missing(merged_df, method=args.impute, knn_neighbours=args.knn_neighbours, logger=logger)
    else:
        logger.info("Imputation skipped (impute=none).")


    # 7a. default DMSO normalisation (after scaling, before feature selection)

    if not args.no_dmso_normalisation:
        feature_cols = [c for c in merged_df.select_dtypes(include=[np.number]).columns if c != "row_number"]
        merged_df = normalise_to_dmso(
            merged_df,
            feature_cols,
            metadata_col="cpd_type",
            dmso_label="dmso",
            logger=logger
        )
        logger.info("DMSO normalisation complete.")
        log_memory_usage(logger, prefix=" [After DMSO normalisation ]")
    else:
        logger.info("DMSO normalisation disabled by user (--no_dmso_normalisation set).")


    # 7. Per-plate scaling (if requested)
    if args.scale_per_plate and args.scale_method != "none":
        feature_cols = [c for c in merged_df.select_dtypes(include=[np.number]).columns if c != "row_number"]
        merged_df = scale_per_plate(merged_df, plate_col=plate_col, method=args.scale_method, logger=logger)
    else:
        logger.info("Per-plate scaling not performed.")

    log_memory_usage(logger, prefix=" [After scaling ]")
    # 8. Standardise metadata columns for downstream compatibility
    merged_df = standardise_metadata_columns(merged_df, logger=logger, dataset_name="merged_raw")

    # 9. Preserve metadata columns 
    # row number removed for now, will add if needed downstream
    # metadata_cols = ["row_number", "cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]

    metadata_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    present_metadata = [col for col in metadata_cols if col in merged_df.columns]
    metadata_df = merged_df[present_metadata].copy()


    # 10. Feature selection (variance, then correlation filtering for efficiency)
    feature_cols = [c for c in merged_df.columns if c not in present_metadata and pd.api.types.is_numeric_dtype(merged_df[c])]
    n_start_features = len(feature_cols)
    logger.info(f"Feature selection on {n_start_features} numeric columns.")

    # Convert features to float32 to save memory
    merged_df[feature_cols] = merged_df[feature_cols].astype(np.float32)
    gc.collect()
    logger.info(f"Features cast to float32. Shape: {merged_df[feature_cols].shape}")

    # Variance threshold filtering (first for RAM efficiency)
    logger.info(f"variance threshold: {args.variance_threshold}")
    if args.variance_threshold and args.variance_threshold > 0.0:
        logger.info(f"Applying variance threshold: {args.variance_threshold}")
        filtered_var = variance_threshold_selector(merged_df[feature_cols], threshold=args.variance_threshold)
        n_after_var = filtered_var.shape[1]
        logger.info(f"After variance filtering: {n_after_var} features remain (removed {n_start_features - n_after_var}).")
        log_memory_usage(logger, prefix="[After variance filtering] ")
        gc.collect()
    else:
        logger.info("Variance threshold filter disabled (None or zero). Skipping variance filtering.")
        filtered_var = merged_df[feature_cols]

    # Correlation filter (greedy/efficient if possible)
    logger.info(f"correlation threshold: {args.correlation_threshold}")
    if args.correlation_threshold and args.correlation_threshold > 0.0:
        logger.info(f"Applying correlation threshold: {args.correlation_threshold}")
        filtered_corr = correlation_filter(filtered_var, threshold=args.correlation_threshold)
        n_after_corr = filtered_corr.shape[1]
        logger.info(f"After correlation filtering: {n_after_corr} features remain (removed {n_after_var - n_after_corr}).")
    else:
        logger.info("Correlation threshold filter disabled (None or zero). Skipping correlation filtering.")
        filtered_corr = filtered_var
        logger.info("No correlation filtering applied, using all variance-filtered features.")


    log_memory_usage(logger, prefix="[After correlation filtering] ")
    gc.collect()
    logger.info(f"Feature selection retained {n_after_corr} features.")


    # 11. Combine metadata and filtered features
    # Only keep essential metadata columns and the final filtered features
    metadata_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    present_metadata = [col for col in metadata_cols if col in merged_df.columns]

    # After feature selection:
    final_cols = present_metadata + list(filtered_corr.columns)
    final_df = merged_df[final_cols].copy()



    # 12. OUTPUT SECTION: always write both per-object and per-well median output

    # Drop duplicate columns (keep first occurrence)
    dupe_cols = final_df.columns.duplicated()
    if any(dupe_cols):
        dup_names = final_df.columns[dupe_cols].tolist()
        logger.warning(f"Found and dropping duplicate columns in output: {dup_names}")
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Remove columns not needed in output
    cols_to_drop = ['ImageNumber', 'ObjectNumber']
    final_df = final_df.drop(columns=[col for col in cols_to_drop if col in final_df.columns])

    # --- PER-OBJECT OUTPUT ---
    per_object_output_file = args.output_file
    per_object_output = final_df

    # Write per-object output
    if not args.no_compress_output and per_object_output_file.endswith(".gz"):
        compression = "gzip"
        logger.info("Per-object output will be gzip-compressed (.gz).")
    else:
        compression = None
        if args.no_compress_output:
            logger.info("Per-object output compression disabled by user (--no_compress_output set).")
        elif not per_object_output_file.endswith(".gz"):
            logger.info("Per-object output file does not end with .gz; writing uncompressed.")

    per_object_output.to_csv(per_object_output_file, sep=args.sep, index=False, compression=compression)
    logger.info(f"Saved per-object data to {per_object_output_file} (shape: {per_object_output.shape})")

    # --- PER-WELL AGGREGATION ---
    logger.info("Aggregating to per-well median. Each row will represent a single well (median of all objects per well).")
    group_cols = ["Plate_Metadata", "Well_Metadata"]
    meta_cols = ["cpd_id", "cpd_type", "Library"]
    keep_cols = group_cols + meta_cols + [c for c in final_df.columns if c not in group_cols + meta_cols]
    output_df = final_df.loc[:, [c for c in keep_cols if c in final_df.columns]].copy()
    # Median aggregation for features (excluding meta/group)
    feature_cols = [c for c in output_df.columns if c not in group_cols + meta_cols]
    agg_df = output_df.groupby(group_cols, as_index=False)[feature_cols].median()
    # Merge constant metadata back per well
    for col in meta_cols:
        if col in output_df.columns:
            unique_meta = output_df.groupby(group_cols, as_index=False)[col].agg(
                lambda x: x.dropna().unique()[0] if len(x.dropna().unique()) == 1 else pd.NA
            )
            agg_df = agg_df.merge(unique_meta, on=group_cols, how="left")
    # Reorder columns: meta, group, features
    ordered_cols = [c for c in meta_cols if c in agg_df.columns] + group_cols + [c for c in agg_df.columns if c not in meta_cols + group_cols]
    agg_df = agg_df.loc[:, ordered_cols]

    # Name per-well output file: default to same as output_file, but with '_per_well' before extension
    if hasattr(args, 'per_well_output_file') and args.per_well_output_file is not None:
        per_well_output_file = args.per_well_output_file
    else:
        # Insert '_per_well' before extension
        file_parts = os.path.splitext(per_object_output_file)
        if file_parts[1] == ".gz":
            file_base, ext2 = os.path.splitext(file_parts[0])
            per_well_output_file = f"{file_base}_per_well{ext2}.gz"
        else:
            per_well_output_file = f"{file_parts[0]}_per_well{file_parts[1]}"
    # Write per-well output
    agg_df.to_csv(per_well_output_file, sep=args.sep, index=False, compression=compression)
    logger.info(f"Saved per-well aggregated data to {per_well_output_file} (shape: {agg_df.shape})")

    logger.info("Merge complete. Note: not all metadata is preserved. Re-attach later if required.")
    log_memory_usage(logger, prefix=" END ")





if __name__ == "__main__":
    main()
