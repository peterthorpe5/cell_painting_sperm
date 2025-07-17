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
import re
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
    parser.add_argument(
    '--normalise_to_dmso',
    action='store_true',
    help='If set, normalise each feature to the median of DMSO wells (default: False).')

    parser.add_argument('--correlation_threshold', type=float, default=0.99,
                        help='Correlation threshold for filtering features (default: 0.99).')
    parser.add_argument('--variance_threshold', type=float, default=0.05,
                        help='Variance threshold for filtering features (default: 0.05).  Low-variance features are almost constant across all samples—they do not help distinguish between classes or clusters.')
    parser.add_argument('--library', type=str, default=None,
                    help="Value to add as 'Library' column if not present in the metadata. "
                         "If omitted and column missing, script will error.")
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
    Normalise features by subtracting the DMSO median per plate.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame, including features and metadata.
    feature_cols : list of str
        Columns to normalise.
    metadata_col : str
        Column name indicating compound type.
    dmso_label : str
        Value in metadata_col indicating DMSO wells.
    logger : logging.Logger, optional
        Logger for progress messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with features normalised to DMSO per plate.
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
        idx_dmso = idx_plate & (df[metadata_col].str.lower() == dmso_label)
        if idx_dmso.sum() == 0:
            logger.warning(f"No DMSO wells found for plate {plate}. Skipping DMSO normalisation for this plate.")
            continue
        dmso_median = df.loc[idx_dmso, feature_cols].median()
        df_norm.loc[idx_plate, feature_cols] = df.loc[idx_plate, feature_cols] - dmso_median
        logger.info(f"Normalised plate {plate} to DMSO median.")
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



import pandas as pd

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
    meta_df = robust_read_csv(args.metadata_file, logger=logger)
    meta_df = harmonise_metadata_columns(meta_df, logger, is_metadata_file=True)
    meta_df = clean_metadata_columns(meta_df, logger)
    logger.info(f"Shape meta_df: {meta_df.shape}")




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


    # Standardise Well_Metadata in CellProfiler data
    cp_df['Well_Metadata'] = cp_df['Well_Metadata'].apply(standardise_well_name)

    # Standardise Well_Metadata (or 'Well', if not renamed yet) in metadata
    if 'Well_Metadata' in meta_df.columns:
        meta_df['Well_Metadata'] = meta_df['Well_Metadata'].apply(standardise_well_name)
    elif 'Well' in meta_df.columns:
        meta_df['Well'] = meta_df['Well'].apply(standardise_well_name)


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
    meta_cols_to_check = [col for col in meta_df.columns if col not in [meta_plate_col, meta_well_col] and col in merged_df.columns]
    if meta_cols_to_check:
        missing_meta = merged_df[meta_cols_to_check].isnull().all(axis=1).sum()
        if missing_meta > 0:
            logger.warning(f"{missing_meta} rows have missing metadata after merge.")
    else:
        logger.warning("No additional metadata columns found in merged_df for missing metadata check.")



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
    logger.info(f"DataFrame shape after dropping all-NaN columns: {merged_df.shape}")



    if args.impute != "none":
        merged_df = impute_missing(merged_df, method=args.impute, knn_neighbours=args.knn_neighbours, logger=logger)
    else:
        logger.info("Imputation skipped (impute=none).")


    # 7a. Optional DMSO normalisation (after scaling, before feature selection)
    if args.normalise_to_dmso:
        feature_cols = [c for c in merged_df.select_dtypes(include=[np.number]).columns if c != "row_number"]
        merged_df = normalise_to_dmso(
            merged_df,
            feature_cols,
            metadata_col="cpd_type",
            dmso_label="dmso",
            logger=logger
        )
        logger.info("DMSO normalisation complete.")
    else:
        logger.info("DMSO normalisation not requested.")


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
    # Only keep essential metadata columns and the final filtered features
    metadata_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    present_metadata = [col for col in metadata_cols if col in merged_df.columns]

    # After feature selection:
    final_cols = present_metadata + list(filtered_final.columns)
    final_df = merged_df[final_cols].copy()



    # 12. Output
    # Drop duplicate columns if present (keep first occurrence)
    dupe_cols = final_df.columns.duplicated()
    if any(dupe_cols):
        dup_names = final_df.columns[dupe_cols].tolist()
        logger.warning(f"Found and dropping duplicate columns in output: {dup_names}")
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    # List columns to drop
    cols_to_drop = ['ImageNumber', 'ObjectNumber']

    # Only drop if present to avoid errors
    final_df = final_df.drop(columns=[col for col in cols_to_drop if col in final_df.columns])

    final_df.to_csv(args.output_file, sep=args.sep, index=False)
    logger.info(f"Saved feature-selected data to {args.output_file} (shape: {final_df.shape})")
    logger.info("Merge complete.")
    logger.info("note to user: I do not keep all metadata. Re-attach later if you need it in the results")

if __name__ == "__main__":
    main()
