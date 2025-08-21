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
import gzip 
import pandas as pd
import numpy as np
from itertools import islice
from typing import Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import shapiro
import gc
from typing import Optional, Iterable
from cell_painting.process_data import (
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
                        help='If set, skip DMSO normalisation. Default: OFF (normalisation ON).')
    parser.add_argument('--correlation_threshold', type=float, default=0.99,
                        help='Correlation threshold for filtering features (default: 0.99).')
    parser.add_argument('--variance_threshold', type=float, default=0.05,
                        help='Variance threshold for filtering features (default: 0.05).  Low-variance features are almost constant across all samples—they do not help distinguish between classes or clusters.')
    parser.add_argument('--library', type=str, default=None,
                    help="Value to add as 'Library' column if not present in the metadata. "
                         "If omitted and column missing, script will error.")
    parser.add_argument('--sep', default='\t', help='Delimiter for output file (default: tab).')
    parser.add_argument('--aggregate_per_well',
                        action='store_true',
                        help='If set, output one row per well (median of all objects in the well). Default: off (per-object output).'
                        )
    parser.add_argument('--corr_strategy',
                        choices=['variance', 'min_redundancy'],
                        default='variance',
                        help='How to pick a representative within correlated groups.'
                    )
    parser.add_argument('--protect_features',
                        type=lambda s: [x.strip() for x in s.split(',')] if s else None,
                        default=None,
                        help='Comma-separated list of features to always keep.'
                    )

    parser.add_argument('--per_well_output_file', type=str, default=None,
                        help="Optional output file for per-well aggregated data (default: auto-named from --output_file)")
    parser.add_argument('--per_object_output',
                        action='store_true',
                        help='If set, output per-object (single cell) data as well as per-well aggregated. Default: False (only per-well output).')

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
    Robustly read a CSV/TSV (optionally gzipped), auto-detecting the delimiter,
    decoding with BOM-awareness, and normalising header labels.

    Parameters
    ----------
    path : str
        Path to the file to read.
    logger : logging.Logger, optional
        Logger instance.
    n_check_lines : int, optional
        Number of lines to inspect when guessing the delimiter.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame with normalised column labels.

    Raises
    ------
    ValueError
        If the file cannot be parsed as either CSV or TSV.
    """
    logger = logger or logging.getLogger(__name__)

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8", errors="replace") as fh:
        lines = list(islice(fh, n_check_lines))

    header_has_tabs = any("\t" in line for line in lines)
    try_first, try_second = ("\t", ",") if header_has_tabs else (",", "\t")

    for sep, label in ((try_first, "first guess"), (try_second, "fallback")):
        try:
            # Use explicit sep and BOM-aware decoding; python engine is tolerant.
            df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
            df.columns = normalise_column_labels(df.columns)
            logger.info(
                "Read file as %s-separated (%s).",
                "tab" if sep == "\t" else "comma",
                label,
            )
            return df
        except Exception as err:
            logger.warning(
                "Reading as %s-separated failed (%s).",
                "tab" if sep == "\t" else "comma",
                err,
            )

    raise ValueError(f"Failed to parse file '{path}' as CSV or TSV.")



def normalise_column_labels(cols):
    """
    Clean a sequence of column labels by removing BOM/zero-width characters,
    converting NBSP to a space, and stripping outer whitespace.

    Parameters
    ----------
    cols : Iterable[str]
        Original column labels.

    Returns
    -------
    pandas.Index
        Cleaned column labels as a pandas Index.
    """
    return pd.Index(
        [
            str(c)
            .replace("\ufeff", "")   # BOM
            .replace("\u200b", "")   # zero-width space
            .replace("\xa0", " ")    # non-breaking space -> space
            .strip()
            for c in cols
        ]
    )


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


def _rank_features_for_corr_filter(
    X: pd.DataFrame,
    strategy: str = "variance",
    protect: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> list[str]:
    """
    Produce an ordered list of feature names indicating the priority to KEEP.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric feature matrix (rows = samples, cols = features).
    strategy : {"variance", "min_redundancy"}
        Ranking heuristic:
            - "variance": sort by variance (desc) — keep most variable first.
            - "min_redundancy": sort by mean absolute correlation (asc) —
              keep least redundant first.
    protect : iterable of str, optional
        Features that must be kept. They are placed at the front of the ranking
        (original order preserved).
    logger : logging.Logger, optional
        Logger for diagnostics.

    Returns
    -------
    list[str]
        Feature names sorted by priority to keep.
    """
    logger = logger or logging.getLogger("corr_filter")

    cols = list(X.columns)
    protect = [c for c in (protect or []) if c in cols]
    unprotected = [c for c in cols if c not in protect]

    if strategy == "variance":
        variances = X[unprotected].var(ddof=1).astype(float)
        ranked = list(variances.sort_values(ascending=False).index)
        logger.info(
            "Correlation filter ranking: strategy=variance; "
            "top-5 most variable: %s",
            ranked[:5],
        )

    elif strategy == "min_redundancy":
        # Compute absolute correlation (fill NaN->0 to avoid propagation)
        corr = X[unprotected].corr().abs().fillna(0.0)
        # Mean absolute correlation to others (exclude self by adjusting diagonal to 0)
        np.fill_diagonal(corr.values, 0.0)
        mean_abs_corr = corr.mean(axis=0)
        ranked = list(mean_abs_corr.sort_values(ascending=True).index)
        logger.info(
            "Correlation filter ranking: strategy=min_redundancy; "
            "top-5 least redundant (lowest mean |corr|): %s",
            ranked[:5],
        )

    else:
        logger.warning(
            "Unknown ranking strategy '%s'; falling back to original column order.",
            strategy,
        )
        ranked = unprotected

    # Protected features first (original order), then ranked others
    return protect + ranked


def correlation_filter_smart(
    X: pd.DataFrame,
    threshold: float = 0.99,
    strategy: str = "variance",
    protect: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Drop highly correlated features, keeping the “best” one per correlated group.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric feature matrix (rows = samples, cols = features).
    threshold : float, default 0.99
        Absolute Pearson correlation threshold above which a later feature is dropped.
    strategy : {"variance", "min_redundancy"}
        How to decide which feature to keep first:
          - "variance": keep highest-variance features.
          - "min_redundancy": keep features with lowest mean absolute correlation first.
    protect : iterable of str, optional
        Features that must be retained even if correlated.
    logger : logging.Logger, optional
        Logger for progress and summary.

    Returns
    -------
    pd.DataFrame
        X with correlated columns removed according to the strategy.
    """
    logger = logger or logging.getLogger("corr_filter")

    if X.empty or X.shape[1] == 1:
        logger.info("Correlation filter skipped (matrix has ≤1 column).")
        return X

    # Ensure numeric; guard against non-numeric dtypes sneaking in
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] != X.shape[1]:
        missing = set(X.columns) - set(X_num.columns)
        logger.warning(
            "Non-numeric columns ignored in correlation filter: %s",
            sorted(missing)[:10],
        )

    logger.info(
        "Applying correlation filter: n_features=%d, threshold=%.3f, strategy=%s",
        X_num.shape[1],
        threshold,
        strategy,
    )

    # Ranking (priority to KEEP)
    keep_priority = _rank_features_for_corr_filter(
        X_num, strategy=strategy, protect=protect, logger=logger
    )

    # Precompute absolute correlation matrix for speed
    corr = X_num.corr().abs().fillna(0.0)
    kept: list[str] = []
    dropped: list[str] = []

    # Greedy selection: iterate in keep-priority order;
    # if candidate is highly correlated with any kept feature, drop it.
    kept_set = set()
    for col in keep_priority:
        if col in kept_set:
            continue
        if not kept:
            kept.append(col)
            kept_set.add(col)
            continue

        # Check correlation with already kept features
        # If any |r| >= threshold, drop; else keep.
        if corr.loc[col, kept].max() >= threshold:
            dropped.append(col)
        else:
            kept.append(col)
            kept_set.add(col)

    n_before = X_num.shape[1]
    n_after = len(kept)
    logger.info(
        "Correlation filter result: kept=%d, dropped=%d (%.1f%% removed).",
        n_after,
        n_before - n_after,
        100.0 * (n_before - n_after) / max(n_before, 1),
    )

    # Helpful debug preview
    if dropped:
        logger.debug("First 10 dropped due to high correlation: %s", dropped[:10])
    logger.debug("First 10 kept: %s", kept[:10])

    # Return DataFrame with kept columns only (preserve original column order among kept)
    kept_in_original_order = [c for c in X.columns if c in kept_set]
    return X.loc[:, kept_in_original_order]



def add_plate_well_metadata_if_missing(obj_df: pd.DataFrame,
                                        input_dir: Path,
                                        obj_filename: str,
                                        logger: logging.Logger
                                    ) -> pd.DataFrame:
    """
    Attach plate/well metadata to an object-level table by merging its matching image file.

    The function:
      - Checks if plate/well already exist; if yes, returns the input unchanged.
      - Locates the sibling *_Image.csv[.gz] file based on the object filename.
      - Reads ONLY the columns needed from the image file (ImageNumber + plate/well).
      - Left-merges on 'ImageNumber'.
      - Returns a DataFrame and does not leave duplicate suffixed columns behind.

    Args:
        obj_df: Object-level DataFrame (e.g., Cell, Cytoplasm, Nuclei table).
        input_dir: Directory containing the object and image tables.
        obj_filename: The object-level filename (used to find the image file).
        logger: Logger for status/warning messages.

    Returns:
        A DataFrame guaranteed to have, if possible, plate/well columns (possibly
        via merge with the image table). If the image table cannot be found or
        lacks the required keys, the original DataFrame is returned.
    """
    plate_candidates = ["Plate", "Plate_Metadata", "Metadata_Plate", "Image_Metadata_Plate"]
    well_candidates = ["Well", "Well_Metadata", "Metadata_Well", "Image_Metadata_Well"]

    # Already have plate/well? Done.
    has_plate = any(col in obj_df.columns for col in plate_candidates)
    has_well = any(col in obj_df.columns for col in well_candidates)
    if has_plate and has_well:
        logger.info(f"Plate/Well metadata already present in {obj_filename}")
        return obj_df

    # Derive base and look for *_Image
    base = obj_filename
    for suffix in ["_Cell", "_Cytoplasm", "_Nuclei"]:
        base = base.replace(suffix, "")
    for suffix in [".csv.gz", ".csv"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]

    candidates = [f"{base}_Image.csv.gz", f"{base}_Image.csv"]
    image_path = None
    for cand in candidates:
        p = input_dir / cand
        if p.exists():
            image_path = p
            break

    if image_path is None:
        logger.error(
            f"Plate/Well metadata missing and no matching image-level file found for {obj_filename}. "
            f"Tried: {candidates}"
        )
        return obj_df

    # Read as little as we can from the image table
    try:
        # Peek to find actual plate/well col names
        img_head = pd.read_csv(image_path, nrows=5)
        img_plate = next((c for c in plate_candidates if c in img_head.columns), None)
        img_well = next((c for c in well_candidates if c in img_head.columns), None)
        usecols = ["ImageNumber"] + [c for c in [img_plate, img_well] if c is not None]
        if "ImageNumber" not in obj_df.columns:
            logger.error(f"Cannot merge {obj_filename} and {image_path.name}: 'ImageNumber' not present in object file.")
            return obj_df

        if not usecols or len(usecols) == 1:  # only ImageNumber found or none
            logger.error(f"No plate/well columns found in {image_path.name}.")
            return obj_df

        img_df = pd.read_csv(image_path, usecols=usecols)
    except Exception as exc:
        logger.error(f"Failed reading image file '{image_path.name}': {exc}")
        return obj_df

    if "ImageNumber" not in img_df.columns:
        logger.error(f"Image file '{image_path.name}' lacks 'ImageNumber'; cannot merge.")
        return obj_df

    # Merge and clean
    merged = obj_df.merge(img_df, on="ImageNumber", how="left", suffixes=("", "_img"))
    logger.info(f"Merged {obj_filename} with {image_path.name}; resulting shape: {merged.shape}")

    # Standardise/clean to a single pair of columns
    merged = select_single_plate_well(merged, logger=logger)
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
    Harmonise a column name to a canonical target, with defensive handling.

    Steps
    -----
    1) De-duplicate the provided `candidates` list (order preserved).
    2) Detect duplicate column *labels* in `df` and de-duplicate by keeping the
       first occurrence (logs which labels were collapsed).
    3) If `target` already exists, return immediately.
    4) Find which candidate (if any) exists in `df` after de-duplication.
       - If none: log a warning and return (df, None).
       - If more than one *distinct* candidate exists: raise a clear error.
       - If exactly one: rename it to `target`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame whose columns may contain several variants of a name.
    candidates : list of str
        Possible column names to look for (case-sensitive, order matters).
        Duplicates in this list are ignored.
    target : str
        Desired canonical column name to enforce.
    logger : logging.Logger
        Logger for diagnostic messages.

    Returns
    -------
    tuple[pandas.DataFrame, str | None]
        The (possibly modified) DataFrame and the resolved column name
        (the canonical `target`) or None if nothing matched.

    Raises
    ------
    ValueError
        If multiple *distinct* candidate columns are present in `df`.
    """
    # 1) De-duplicate candidate strings while preserving order
    cand_unique = list(dict.fromkeys([str(c) for c in candidates]))
    if len(cand_unique) != len(candidates):
        dup_in_candidates = [c for i, c in enumerate(candidates) if c in candidates[:i]]
        logger.debug(
            "Candidate list contained duplicates; collapsed these (kept first): %s",
            sorted(set(dup_in_candidates))
        )
    logger.debug("Candidates (unique, ordered): %s", cand_unique)
    logger.debug("Target: %s", target)

    # 2) De-duplicate duplicate column labels in df (keep first)
    dup_mask = df.columns.duplicated()
    if dup_mask.any():
        dup_labels = df.columns[dup_mask].unique().tolist()
        before_cols = df.shape[1]
        logger.warning(
            "Duplicate column labels detected and will be de-duplicated by keeping the first occurrence: %s",
            dup_labels
        )
        df = df.loc[:, ~df.columns.duplicated()]
        logger.info("Column de-duplication: %d → %d columns.",
                    before_cols, df.shape[1])
    else:
        logger.debug("No duplicate column labels found in DataFrame.")

    # 3) If the canonical target already exists, prefer it outright
    if target in df.columns:
        logger.info("Target column '%s' already present; no rename needed.", target)
        return df, target

    # 4) Identify which (unique) candidates exist in the DataFrame
    found = [c for c in cand_unique if c in df.columns]
    logger.debug("Candidates present in DataFrame: %s", found)

    if len(found) == 0:
        logger.warning(
            "No candidate columns for '%s' found in DataFrame. Tried: %s",
            target, cand_unique
        )
        return df, None

    if len(found) > 1:
        # More than one *distinct* name present → genuine ambiguity
        logger.error(
            "Ambiguous candidates for '%s' found in DataFrame: %s. "
            "Please ensure only one exists and re-run.",
            target, found
        )
        raise ValueError(
            f"Ambiguous candidates for '{target}' found: {found}. "
            "Please ensure only one exists in your input file and re-run the script."
        )

    # Exactly one candidate present: rename if needed
    chosen = found[0]
    if chosen == target:
        logger.info("Chosen candidate equals target '%s'; no rename performed.", target)
        return df, target

    if target in df.columns:
        # This branch is defensive; we already returned earlier if target existed.
        logger.error(
            "Target '%s' already exists alongside candidate '%s'. "
            "No renaming performed. Please resolve this conflict.",
            target, chosen
        )
        raise ValueError(
            f"Target column '{target}' already exists alongside candidate '{chosen}'. "
            "Please resolve this conflict in your input file."
        )

    logger.info("Renaming column '%s' to canonical '%s'.", chosen, target)
    df = df.rename(columns={chosen: target})
    return df, target




def impute_missing(df, method="knn", knn_neighbours=5, logger=None,
                   max_cells=100000000, max_features=300000):
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


def harmonise_metadata_columns(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
    is_metadata_file: bool = False,
    preferred: Optional[tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Harmonise plate and well column names to 'Plate_Metadata' and 'Well_Metadata'.

    Resolution order:
      1) Use user-preferred names from --merge_keys (case/underscore-insensitive).
      2) Try broad synonym lists for plate and well headers.
      3) If Row/Column exist, construct Well as A01-style and use that.

    Parameters
    ----------
    df : pd.DataFrame
        Input metadata DataFrame.
    logger : logging.Logger, optional
        Logger for messages.
    is_metadata_file : bool
        Kept for compatibility (not used).
    preferred : tuple[str, str] or None
        (plate_name, well_name) as passed by the user.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Plate_Metadata' and 'Well_Metadata' present.

    Raises
    ------
    ValueError
        If required columns cannot be resolved.
    """
    logger = logger or logging.getLogger("meta_harmonise")
    df = df.copy()
    df.columns = normalise_column_labels(df.columns)

    def _norm(s: str) -> str:
        return s.replace("_", "").replace(" ", "").lower()

    def _find_col(candidates: list[str]) -> Optional[str]:
        norm_map = {_norm(c): c for c in df.columns}
        for cand in candidates:
            hit = norm_map.get(_norm(cand))
            if hit:
                return hit
        return None

    plate_col = None
    well_col = None

    # 1) Preferred keys from --merge_keys
    if preferred and len(preferred) == 2:
        plate_try, well_try = preferred
        plate_col = _find_col([plate_try])
        well_col = _find_col([well_try])

    # 2) Synonyms
    if plate_col is None:
        plate_syns = [
            "Plate_Metadata", "Plate", "plate", "Barcode", "Plate_Barcode", "PlateBarcode",
            "PlateID", "Plate_Id", "PlateName", "Assay_Plate_Barcode", "AssayPlate_Barcode",
            "Metadata_Plate", "Image_Metadata_Plate", "original_Pt_Mt"
        ]
        plate_col = _find_col(plate_syns)

    if well_col is None:
        well_syns = [
            "Well_Metadata", "Well", "well", "Well Name", "WellName", "Well_ID", "WellID",
            "Metadata_Well", "Image_Metadata_Well"
        ]
        well_col = _find_col(well_syns)

    # 3) Fallback from Row/Column
    if well_col is None:
        row_col = _find_col(["Row", "WellRow", "row", "Image_Metadata_Row"])
        col_col = _find_col(["Column", "Col", "WellColumn", "column", "Image_Metadata_Column", "Image_Metadata_Col"])
        if row_col and col_col:
            tmp = df[[row_col, col_col]].copy()
            tmp["Well_Metadata"] = (
                tmp[row_col].astype(str).str.strip().str.upper().str[0]
                + tmp[col_col].astype(str)
                  .str.extract(r"(\d+)", expand=False).astype(float).fillna(0).astype(int)
                  .astype(str).str.zfill(2)
            )
            df["Well_Metadata"] = tmp["Well_Metadata"]
            well_col = "Well_Metadata"
            logger.info(f"Constructed 'Well_Metadata' from '{row_col}' + '{col_col}'.")

    # Apply final renames to canonical labels
    if plate_col and plate_col != "Plate_Metadata":
        df = df.rename(columns={plate_col: "Plate_Metadata"})
        plate_col = "Plate_Metadata"
        logger.info("Renamed plate column to 'Plate_Metadata'.")

    if well_col and well_col != "Well_Metadata":
        df = df.rename(columns={well_col: "Well_Metadata"})
        well_col = "Well_Metadata"
        logger.info("Renamed well column to 'Well_Metadata'.")

    missing = [c for c in ("Plate_Metadata", "Well_Metadata") if c not in df.columns]
    if missing:
        cols_preview = ", ".join(df.columns.tolist())
        logger.error(f"Could not harmonise: missing {missing}. Available columns: {cols_preview}")
        raise ValueError(
            f"Could not harmonise plate/well names in metadata. Missing {missing}. "
            f"Available columns: {cols_preview}"
        )

    return df



def select_single_plate_well(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Ensure exactly one pair of plate/well metadata columns exists with canonical names.

    The function:
      - Detects possible plate and well column variants.
      - Prefers to keep/rename to 'Plate_Metadata' and 'Well_Metadata'.
      - Drops any alternative/suffixed duplicates (e.g. *_img, *_x, *_y).
      - Leaves the rest of the DataFrame intact.

    Args:
        df: Input DataFrame that may contain multiple plate/well columns.
        logger: Optional logger for informational messages.

    Returns:
        A DataFrame with a single, consistent pair of columns:
        'Plate_Metadata' and 'Well_Metadata'.
    """
    logger = logger or logging.getLogger("coerce_plate_well")

    plate_variants = [
        "Plate_Metadata", "Plate", "plate", "Metadata_Plate", "Image_Metadata_Plate",
        "Plate_Metadata_img", "Plate_img", "Metadata_Plate_img",
        "Plate_Metadata_x", "Plate_Metadata_y"
    ]
    well_variants = [
        "Well_Metadata", "Well", "well", "Metadata_Well", "Image_Metadata_Well",
        "Well_Metadata_img", "Well_img", "Metadata_Well_img",
        "Well_Metadata_x", "Well_Metadata_y"
    ]

    df_out = df.copy()

    # ---- Plate ----
    plate_present = [c for c in plate_variants if c in df_out.columns]
    if not plate_present:
        if logger:
            logger.debug("No plate column candidates found; leaving as-is.")
    else:
        keep_plate = "Plate_Metadata" if "Plate_Metadata" in plate_present else plate_present[0]
        if keep_plate != "Plate_Metadata":
            if "Plate_Metadata" in df_out.columns and keep_plate != "Plate_Metadata":
                # already have a Plate_Metadata; do nothing
                pass
            else:
                df_out = df_out.rename(columns={keep_plate: "Plate_Metadata"})
        # Drop all other plate variants except the canonical one
        drop_plate = [c for c in plate_present if c != "Plate_Metadata"]
        if drop_plate and logger:
            logger.info(f"Dropping alternative plate columns: {drop_plate}")
        df_out = df_out.drop(columns=drop_plate, errors="ignore")

    # ---- Well ----
    well_present = [c for c in well_variants if c in df_out.columns]
    if not well_present:
        if logger:
            logger.debug("No well column candidates found; leaving as-is.")
    else:
        keep_well = "Well_Metadata" if "Well_Metadata" in well_present else well_present[0]
        if keep_well != "Well_Metadata":
            if "Well_Metadata" in df_out.columns and keep_well != "Well_Metadata":
                # already have a Well_Metadata; do nothing
                pass
            else:
                df_out = df_out.rename(columns={keep_well: "Well_Metadata"})
        # Drop all other well variants except the canonical one
        drop_well = [c for c in well_present if c != "Well_Metadata"]
        if drop_well and logger:
            logger.info(f"Dropping alternative well columns: {drop_well}")
        df_out = df_out.drop(columns=drop_well, errors="ignore")

    # Final tidy: any generic suffixed junk that slipped through
    junk_like = [c for c in df_out.columns if c.endswith(("_img", "_x", "_y"))]
    # Keep if they are not the canonical Plate/Well names
    junk_like = [c for c in junk_like if c not in ("Plate_Metadata", "Well_Metadata")]
    if junk_like and logger:
        logger.info(f"Dropping leftover suffixed columns: {junk_like[:10]}{'...' if len(junk_like) > 10 else ''}")
    df_out = df_out.drop(columns=junk_like, errors="ignore")

    return df_out


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


def _uniq(seq):
    """
    Return items from `seq` with order preserved and duplicates removed.

    Parameters
    ----------
    seq : Iterable
        Input sequence.

    Returns
    -------
    list
        De-duplicated list preserving first occurrence order.
    """
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def standardise_metadata_columns(df, logger=None, dataset_name=None):
    """
    Standardise metadata columns (cpd_id, cpd_type, Library) in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (can be merged data).
    logger : logging.Logger, optional
        Logger for messages.
    dataset_name : str or None
        Kept for backward compatibility; not used.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardised metadata columns.
    """
    return clean_metadata_columns(df, logger=logger)


def _norm_plate(s):
    return s.astype(str).str.strip()

def _norm_well(s):
    return (
        s.astype(str)
         .str.strip()
         .str.upper()
         .str.replace(r'^([A-H])(\d{1,2})$', lambda m: f"{m.group(1)}{int(m.group(2)):02d}", regex=True)
    )

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

        # Add Plate/Well from the matching *_Image file if needed,
        # and coerce to a single, clean pair of Plate_Metadata/Well_Metadata
        df = add_plate_well_metadata_if_missing(df, input_dir, f.name, logger)
        df = select_single_plate_well(df, logger=logger)

        log_memory_usage(logger, prefix=f" [After {f.name} loaded] ")
        dataframes.append(df)

        # Free temp DataFrame - too much RAM being used
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

    plate_candidates = _uniq([
        merge_keys[0],
        "Plate", "plate", "Plate_Metadata", "Metadata_Plate", "Image_Metadata_Plate"
    ])
    well_candidates = _uniq([
        merge_keys[1],
        "Well", "well", "Well_Metadata", "Metadata_Well", "Image_Metadata_Well"
    ])

    cp_df, plate_col = harmonise_column_names(cp_df, plate_candidates, "Plate_Metadata", logger)
    cp_df, well_col  = harmonise_column_names(cp_df, well_candidates,  "Well_Metadata",  logger)
    if plate_col is None or well_col is None:
        logger.error("Could not harmonise plate/well column names in CellProfiler data.")
        sys.exit(1)
    logger.info("CellProfiler: using plate column '%s', well column '%s'.", plate_col, well_col)



    # 3. Load and harmonise metadata
    meta_df = robust_read_csv(args.metadata_file, logger=logger)

    # Use the caller’s explicit merge keys as first preference
    preferred_keys = tuple([k.strip() for k in args.merge_keys.split(",")]) if args.merge_keys else None
    meta_df = harmonise_metadata_columns(meta_df, logger=logger, is_metadata_file=True, preferred=preferred_keys)

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

    logger.info(f"CellProfiler unique plate values: {sorted(cp_df[plate_col].unique())[:10]}")
    logger.info(f"Metadata unique plate values: {sorted(meta_df[meta_plate_col].unique())[:10]}")
    logger.info(f"CellProfiler unique well values: {sorted(cp_df[well_col].unique())[:10]}")
    logger.info(f"Metadata unique well values: {sorted(meta_df[meta_well_col].unique())[:10]}")

    # 4. Merge metadata
    for df_ in (cp_df, meta_df):
        for col in ('Plate_Metadata', 'Well_Metadata'):
            if col in df_.columns and pd.api.types.is_string_dtype(df_[col]):
                df_[col] = df_[col].str.strip()
    

    cp_df[plate_col] = _norm_plate(cp_df[plate_col])
    cp_df[well_col]  = _norm_well(cp_df[well_col])


    # --- 0) Make sure keys are normalized ---
    meta_df[meta_plate_col] = _norm_plate(meta_df[meta_plate_col])
    meta_df[meta_well_col]  = _norm_well(meta_df[meta_well_col])

    # --- 1) Audit duplicates by PlatexWell ---
    dup_mask = meta_df.duplicated([meta_plate_col, meta_well_col], keep=False)
    if dup_mask.any():
        dup_preview = meta_df.loc[
            dup_mask,
            [meta_plate_col, meta_well_col, "cpd_id", "cpd_type", "Library"]
            + [c for c in meta_df.columns if "concentration" in c.lower()]
        ].head(10)
        logger.warning(
            f"Metadata has duplicated PlatexWell keys (n={dup_mask.sum()}). "
            f"Example rows:\n{dup_preview}"
        )

    # --- 2) Assert single cpd_id per well (just to be safe) ---
    multi_cpd = (
        meta_df.groupby([meta_plate_col, meta_well_col])["cpd_id"]
            .nunique(dropna=True)
    )
    conflict_keys = multi_cpd[multi_cpd > 1]
    if len(conflict_keys):
        bad = (meta_df
            .merge(conflict_keys.reset_index().drop(columns=["cpd_id"]),
                    on=[meta_plate_col, meta_well_col], how="inner")
            .sort_values([meta_plate_col, meta_well_col]))
        logger.error(
            "Some wells map to multiple distinct cpd_id values. "
            "Fix the plate map before proceeding. Example:\n"
            f"{bad[[meta_plate_col, meta_well_col, 'cpd_id']].head(20)}"
        )
        raise ValueError("Metadata has wells with >1 cpd_id.")

    # --- 3) Drop concentration columns (you don't need them now) ---
    conc_cols = [c for c in meta_df.columns if "concentration" in c.lower()]
    if conc_cols:
        logger.info(f"Dropping concentration columns from metadata: {conc_cols}")
        # optional: keep an audit file
        # meta_df[ [meta_plate_col, meta_well_col] + conc_cols ].to_csv("dropped_concentrations.tsv", sep="\t", index=False)
        meta_df = meta_df.drop(columns=conc_cols)

    # --- 4) Collapse to exactly one row per Plate×Well ---
    before = meta_df.shape[0]
    meta_df = meta_df.drop_duplicates(subset=[meta_plate_col, meta_well_col], keep="first")
    after = meta_df.shape[0]
    if before != after:
        logger.info(f"Collapsed metadata to unique keys: {before} → {after} rows.")

    # (re)create key sets for diagnostics
    left_keys  = cp_df[[plate_col,  well_col ]].drop_duplicates()
    right_keys = meta_df[[meta_plate_col, meta_well_col]].drop_duplicates()

    logger.info(f"cp_df unique Plate×Well:  {left_keys.shape[0]:,}")
    logger.info(f"meta_df unique Plate×Well:{right_keys.shape[0]:,}")
    logger.info("Intersection size: {:,}".format(
        left_keys.merge(
            right_keys,
            left_on=[plate_col, well_col],
            right_on=[meta_plate_col, meta_well_col]
        ).shape[0]
    ))

    # --- 5) Safe merge: enforce many-to-one ---
    merged_df = cp_df.merge(
        meta_df,
        how="left",
        left_on=[plate_col, well_col],
        right_on=[meta_plate_col, meta_well_col],
        suffixes=('', '_meta'),
        validate="many_to_one"
    )
    logger.info(f"Shape after metadata merge: {merged_df.shape}")


    anti = left_keys.merge(
        right_keys,
        left_on=[plate_col, well_col],
        right_on=[meta_plate_col, meta_well_col],
        how='left',
        indicator=True
    ).query('_merge == "left_only"')[[plate_col, well_col]]

    n_missing = len(anti)
    logger.warning(f"Keys missing in metadata: {n_missing} Plate×Well pairs (unique).")

    if n_missing:
        top = (cp_df.merge(anti, on=[plate_col, well_col], how='inner')
                    [plate_col].value_counts().head(10))
        logger.warning(f"Top plates with missing metadata:\n{top.to_string()}")
        logger.info(f"Example missing keys:\n{anti.head(20).to_string(index=False)}")


    # After merge
    n_total = merged_df.shape[0]
    n_cpd = merged_df['cpd_id'].notna().sum() if 'cpd_id' in merged_df.columns else 0
    logger.info(f"Post-merge: cpd_id present for {n_cpd}/{n_total} rows ({100.0*n_cpd/n_total:.1f}%).")

    # If low assignment, show top key mismatches
    if n_cpd < 0.8 * n_total:
        logger.warning("Low cpd_id assignment. Dumping a small sample of unmatched keys...")
        missing = merged_df.loc[merged_df['cpd_id'].isna(), ['Plate_Metadata','Well_Metadata']].drop_duplicates().head(20)
        logger.warning(f"Example missing keys:\n{missing}")



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


    # 7a. default DMSO normalisation (then scaling, before feature selection)

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


    # ---- SNAPSHOT FOR PER-OBJECT SAVE (if requested) ----
    # Keep a lightweight copy only when user asked for it to avoid RAM blow-up.
    per_object_df = None
    if getattr(args, "per_object_output", False):
        # Keep just meta + numeric features (we will align feature set later)
        meta_keep = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
        meta_keep = [c for c in meta_keep if c in merged_df.columns]
        feat_keep = [c for c in merged_df.columns
                    if c not in meta_keep and pd.api.types.is_numeric_dtype(merged_df[c])]
        per_object_df = merged_df[meta_keep + feat_keep].copy()



    # -------------------------
    # Aggregate to per-well before feature selection
    # -------------------------
    # This is a shame, but too much RAM is used if we dont do this

    # Aggregate per-object data to median per well
    # -------------------------
    # Optional aggregation to per-well BEFORE feature selection
    # -------------------------
    agg_df = None
    if args.aggregate_per_well:
        logger.info("Aggregating per-object data to per-well median (because --aggregate_per_well is set).")
        group_cols = ["Plate_Metadata", "Well_Metadata"]
        meta_cols = ["cpd_id", "cpd_type", "Library"]
        keep_cols = group_cols + meta_cols + [c for c in merged_df.columns if c not in group_cols + meta_cols]
        output_df = merged_df.loc[:, [c for c in keep_cols if c in merged_df.columns]].copy()

        feature_cols = [c for c in output_df.columns
                        if c not in group_cols + meta_cols and pd.api.types.is_numeric_dtype(output_df[c])]

        non_numeric = [c for c in output_df.columns
                    if c not in group_cols + meta_cols and not pd.api.types.is_numeric_dtype(output_df[c])]
        if non_numeric:
            logger.warning(f"Skipped non-numeric columns from aggregation: {non_numeric}")

        agg_df = output_df.groupby(group_cols, as_index=False, observed=False)[feature_cols].median()

        # bring back invariant metadata per well
        for col in meta_cols:
            if col in output_df.columns:
                unique_meta = output_df.groupby(group_cols, as_index=False)[col].agg(
                    lambda x: x.dropna().unique()[0] if len(x.dropna().unique()) == 1 else pd.NA
                )
                agg_df = agg_df.merge(unique_meta, on=group_cols, how="left")

        # order columns
        ordered_cols = [c for c in meta_cols if c in agg_df.columns] + group_cols + \
                    [c for c in agg_df.columns if c not in meta_cols + group_cols]
        agg_df = agg_df.loc[:, ordered_cols]
        logger.info(f"Aggregated DataFrame shape (per-well): {agg_df.shape}")

    # -------------------------
    # Choose the frame to run feature selection on
    # -------------------------
    fs_df = agg_df if args.aggregate_per_well else merged_df

    log_memory_usage(logger, prefix=" [Before feature selection ]")
    present_metadata = [c for c in ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"] if c in fs_df.columns]
    feature_cols = [c for c in fs_df.columns if c not in present_metadata and pd.api.types.is_numeric_dtype(fs_df[c])]

    logger.info(f"Feature selection on {len(feature_cols)} numeric columns.")
    # fs_df[feature_cols] = fs_df[feature_cols].astype(np.float32)
    fs_df.loc[:, feature_cols] = fs_df[feature_cols].astype(np.float32)


    # 10) Variance filter
    if args.variance_threshold and args.variance_threshold > 0.0:
        logger.info(f"Applying variance threshold: {args.variance_threshold}")
        # Apply variance threshold filter
        # This will also generate a PDF with variance diagnostics

        filtered_var = variance_threshold_selector(
                                    data=fs_df[feature_cols],
                                    threshold=args.variance_threshold,
                                    pdf_path=os.path.splitext(args.output_file)[0] + "_variance_diagnostics.pdf",
                                    log_pdf_path=os.path.splitext(args.output_file)[0] + "_variance_diagnostics_log.pdf",
                                    title="Variance diagnostics (pre-filter, post-normalisation)",
                                    bins=None,
                                    bin_scale=3.0,
                                    max_bins=240,
                                    log_sorted_x=False,
                                    log_x_linear_pdf=False
                                )
        logger.info("Feature selection summary: start=%d → after variance=%d",
                    len(feature_cols), filtered_var.shape[1])
        if filtered_var.empty:
            logger.error("Variance threshold filtering resulted in no features remaining. "
                         "Please check your threshold or input data.")

    else:
        logger.info("Variance threshold filter disabled.")
        filtered_var = fs_df[feature_cols]

    # Correlation filter
    if args.correlation_threshold and args.correlation_threshold > 0.0:
        logger.info(f"Applying correlation threshold: {args.correlation_threshold}")
        filtered_corr = correlation_filter_smart(filtered_var,
                                                threshold=args.correlation_threshold, 
                                                strategy="variance",
                                                logger=logger
                                            )
        logger.info("Feature selection summary: start=%d → after variance=%d → after correlation=%d",
                    len(feature_cols), filtered_var.shape[1], filtered_corr.shape[1])

        
    else:
        logger.info("Correlation threshold filter disabled.")
        filtered_corr = filtered_var

    selected_features = list(filtered_corr.columns)
    logger.info(f"Feature selection retained {len(selected_features)} features.")
    # Build the final per-well/per-object output frame for main write
    final_cols = present_metadata + selected_features
    merged_df = fs_df[final_cols].copy()



    log_memory_usage(logger, prefix="[After correlation filtering] ")
    gc.collect()
    # after you set filtered_corr (in both branches)
    n_after_corr = filtered_corr.shape[1]
    logger.info(f"Feature selection retained {n_after_corr} features.")


    # 12) OUTPUT SECTION 

    # Remove columns not needed in output
    cols_to_drop = ['ImageNumber', 'ObjectNumber']
    for col in cols_to_drop:
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=col)

    # Drop duplicate columns (keep first occurrence)
    dupe_cols = merged_df.columns.duplicated()
    if any(dupe_cols):
        dup_names = merged_df.columns[dupe_cols].tolist()
        logger.warning(f"Found and dropping duplicate columns in output: {dup_names}")
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Per-well output (feature-selected, median-aggregated)
    per_well_output_file = args.per_well_output_file or args.output_file
    if not args.no_compress_output and per_well_output_file.endswith(".gz"):
        compression = "gzip"
        logger.info("Per-well output will be gzip-compressed (.gz).")
    else:
        compression = None
        if args.no_compress_output:
            logger.info("Per-well output compression disabled by user (--no_compress_output set).")
        elif not per_well_output_file.endswith(".gz"):
            logger.info("Per-well output file does not end with .gz; writing uncompressed.")

    merged_df.to_csv(per_well_output_file, sep=args.sep, index=False, compression=compression)
    logger.info(f"Saved per-well aggregated, feature-selected data to {per_well_output_file} (shape: {merged_df.shape})")

    # --- OPTIONAL: Per-object output (if --per_object_output set) ---
    # 12. Optional per-object save
    if getattr(args, "per_object_output", False):
        if not args.aggregate_per_well:
            logger.info("--per_object_output requested, but --aggregate_per_well was not set. "
                        "Main output is already per-object; skipping extra per-object file.")
        else:
            if per_object_df is None:
                logger.warning("per_object_df snapshot missing; cannot write per-object output.")
            else:
                # Align columns to the selected feature set for consistency
                meta_keep = [c for c in ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
                            if c in per_object_df.columns]
                keep_cols = meta_keep + [c for c in selected_features if c in per_object_df.columns]
                missing_from_po = set(selected_features) - set(per_object_df.columns)
                if missing_from_po:
                    logger.warning(f"{len(missing_from_po)} selected features not present in per-object table; "
                                f"they will be absent from per-object output.")

                per_object_out = (
                    args.per_well_output_file
                    if args.per_well_output_file else args.output_file
                )
                per_object_out = per_object_out.replace(".tsv.gz", "_per_object.tsv.gz") \
                                            .replace(".tsv", "_per_object.tsv") \
                                            .replace(".csv.gz", "_per_object.csv.gz") \
                                            .replace(".csv", "_per_object.csv")

                po_compression = "gzip" if (not args.no_compress_output and per_object_out.endswith(".gz")) else None
                per_object_df[keep_cols].to_csv(per_object_out, sep=args.sep, index=False, compression=po_compression)
                logger.info(f"Wrote per-object output to {per_object_out} (shape: {per_object_df[keep_cols].shape})")

    mode_msg = "per-well aggregated" if args.aggregate_per_well else "per-object"
    logger.info(f"Saved {mode_msg}, feature-selected data to {per_well_output_file} (shape: {merged_df.shape})")

    logger.info("Output complete.")
    log_memory_usage(logger, prefix=" END ")



if __name__ == "__main__":
    main()