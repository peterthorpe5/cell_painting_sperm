#!/usr/bin/env python3
# coding: utf-8

"""
merge_cellprofiler_with_metadata_featureselect.py

Prepare CellProfiler per-object (single-cell) profiles for analysis with optional
robust outlier trimming and well-level aggregation.

This script:
  • Loads CellProfiler per-object CSV/CSV.GZ files, auto-attaches plate/well metadata,
    merges in a plate map, cleans metadata, optionally trims per-object outliers (robust),
    imputes missing values, optionally normalises to DMSO, optionally scales per plate,
    performs feature selection (variance + correlation), and writes the final table.
  • Can output either per-object rows (default) or per-well medians (with --aggregate_per_well).
  • Avoids using image/normalised tables as input and tolerates missing ImageNumber/ObjectNumber.

Workflow (high level)
---------------------
1) Discover & load per-object tables from --input_dir (excluding files whose names match /image|normalis(e|ed)|NOT_USED/i).
2) Ensure Plate/Well columns exist:
   - If missing, pull them from a sibling *_Image.csv[.gz] (or a unique folder-wide Image table).
   - Harmonise to canonical names: Plate_Metadata, Well_Metadata.
3) Read --metadata_file (plate map), harmonise Plate/Well, clean metadata (cpd_id, cpd_type, Library).
4) Merge metadata many-to-one into the per-object table on Plate×Well.
5) (Optional) Robust per-object outlier trimming (default: before imputation), within each plate×well:
   - Compute per-feature median/MAD, convert to robust z-scores, aggregate |z| to one distance per object.
   - Keep the most central fraction (default: central 90%) per group; drop the rest.
   - Writes an optional per-group QC summary if --trim_qc_file is provided.
6) (Optional) Impute missing numeric values (median or KNN).
7) (Optional) DMSO normalisation per plate (robust z to DMSO med/MAD) unless --no_dmso_normalisation.
8) (Optional) Scale numeric features per plate (standard, robust, or auto).
9) Feature selection:
   - Variance threshold filter.
   - Correlation filter with selectable strategy and optional “protected” features.
10) (Optional) Aggregate to per-well medians (if --aggregate_per_well).
11) Write the final table (compressed if the filename ends with .gz). Optionally also write per-object output.

Key behaviours & notes
---------------------
• Outlier trimming ("robust z-distance"):
  - Scope (--trim_scope): per_well (default), per_plate, or global.
  - Metric (--trim_metric): 
      - "q95" (recommended): 95th qunatile of |z| across features.
      - "l2": sqrt(mean(z^2)).
      - "max": max(|z|).
  - Keep fraction (--keep_central_fraction): default 0.90 keeps the central 90% within each group.
  - Feature usability within a group is data-driven:
      - Drop features from distance calc if > --trim_nan_feature_frac are NaN (default 0.5).
      - Drop objects that have < --trim_min_features_per_cell fraction of the usable features (default 0.5).
  - If a group yields no usable features (e.g., all MAD==0), the group is returned unchanged.
  - Trimming stage (--trim_stage): "pre_impute" (default) or "post_impute".
  - QC: if --trim_qc_file is provided, a small TSV with per-group {n_rows, n_kept, frac_kept, cutoff_distance} is written.

• DMSO normalisation:
  - For each plate and feature: robust-z = (x − median_DMSO) / MAD_DMSO (MAD computed as median(|x − median|)).
  - Skips scaling for features with MAD_DMSO == 0 (centering only).
  - Disable with --no_dmso_normalisation.

• Per-plate scaling:
  - --scale_per_plate with --scale_method {standard, robust, auto}. 
  - "auto" chooses standard vs robust via simple normality heuristics.

• Feature selection:
  - Variance filter: drop low-variance features (threshold configurable).
  - Correlation filter: greedy removal above --correlation_threshold, with ranking strategy
    (--corr_strategy {variance|min_redundancy}) and optional --protect_features to always keep.

• Column handling:
  - Plate/Well names are harmonised and zero-padded wells (e.g., A1→A01).
  - Known bad strings (#VALUE!, #DIV/0!, NA, etc.) → NaN; all-NaN columns are dropped.
  - "ImageNumber" and "ObjectNumber" are safe to be absent; if present they are removed from the final output.

Inputs / Command-line arguments
-------------------------------
Core I/O
  --input_dir                Directory with CellProfiler per-object CSV/CSV.GZ files.
  --metadata_file            CSV/TSV plate map with plate/well metadata.
  --output_file              Output path (TSV/CSV; gz compression if name ends with .gz).
  --sep                      Output delimiter (default: tab).

Merging / Harmonisation
  --merge_keys               Comma-separated plate,well header names to prefer (default: Plate,Well).
  --library                  If metadata lacks 'Library', set all rows to this value (else error).

Trimming (optional; robust outlier removal)
  --trim_objects             Enable robust per-object trimming (default: off).
  --keep_central_fraction    Fraction to keep per group (default: 0.90).
  --trim_scope               per_well | per_plate | global (default: per_well).
  --trim_metric              q95 | l2 | max (default: q95).
  --trim_stage               pre_impute | post_impute (default: pre_impute).
  --trim_nan_feature_frac    Max NaN fraction per feature within a group (default: 0.7).
  --trim_min_features_per_cell  Min fraction of usable features required per object (default: 0.25).
  --trim_qc_file             Optional TSV path for trimming QC summary.

Imputation / Normalisation / Scaling
  --impute                   none | median | knn (default: knn).
  --knn_neighbours           Neighbours for KNN (default: 5).
  --no_dmso_normalisation    Skip DMSO normalisation (default: OFF → normalisation ON).
  --scale_per_plate          Apply per-plate scaling (default: off).
  --scale_method             standard | robust | auto | none (default: robust).

Feature Selection
  --variance_threshold       Drop features below variance threshold (default: 0.05).
  --correlation_threshold    Drop features correlated ≥ threshold (default: 0.99).
  --corr_strategy            variance | min_redundancy (default: variance).
  --protect_features         Comma-separated list of features to always keep.

Aggregation & Extra Outputs
  --aggregate_per_well       Output one row per well (median of objects).
  --per_well_output_file     Optional explicit path for the per-well output.
  --per_object_output        Additionally write a per-object table aligned to selected features.
  --no_compress_output       Force uncompressed output even if filename ends with .gz.

Misc
  --log_level                DEBUG | INFO | WARNING | ERROR (default: INFO).

Requirements
------------
• Python 3.7+
• pandas, numpy, scikit-learn, scipy, psutil
• cell_painting.process_data.variance_threshold_selector in PYTHONPATH

Example usage
-------------
# Default per-object pipeline with robust trimming (keep central 90%), DMSO normalisation, and feature selection
python merge_cellprofiler_with_metadata_featureselect.py \
  --input_dir ./raw_cp/ \
  --metadata_file plate_map.csv \
  --output_file per_object_selected.tsv.gz \
  --merge_keys Plate,Well \
  --trim_objects \
  --keep_central_fraction 0.90 \
  --trim_scope per_well \
  --trim_metric q95 \
  --impute knn \
  --scale_per_plate \
  --scale_method auto \
  --correlation_threshold 0.99 \
  --variance_threshold 0.05 \
  --sep '\t'

# Per-well output (median of objects), with trimming and a QC report
python merge_cellprofiler_with_metadata_featureselect.py \
  --input_dir ./raw_cp/ \
  --metadata_file plate_map.csv \
  --output_file per_well_selected.tsv.gz \
  --aggregate_per_well \
  --trim_objects \
  --keep_central_fraction 0.90 \
  --trim_qc_file trim_qc.tsv \
  --sep '\t'


"""
from __future__ import annotations
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
from typing import Optional, Tuple
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import shapiro
import gc
from typing import Optional, Iterable
from cell_painting.process_data import (
    variance_threshold_selector)

_script_start_time = time.time()

IMAGE_GLOB_PATTERNS = ("*Image.csv.gz", "*_Image.csv.gz", "*Image.csv", "*_Image.csv")
EXCLUDE_PAT = re.compile(r"(?i)(image|normalis(e|ed)|NOT_USED)")  # exclude in object discovery


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
    parser.add_argument('--impute_scope',
                        choices=['global', 'per_plate', 'per_well'],
                        default='per_plate',
                        help='Scope for imputation. Median is fully vectorised for all scopes; '
                            'KNN runs per group for memory/speed if a grouped scope is chosen.'
                    )

    parser.add_argument('--scale_per_plate', action='store_true', help='Apply scaling per plate (default: False).')
    parser.add_argument('--scale_method', choices=['standard', 'robust', 'auto', 'none'], default='robust',
                        help='Scaling method: "standard", "robust", "auto", or "none" (default: robust).')
    parser.add_argument('--no_dmso_normalisation',
                        action='store_true',
                        help='If set, skip DMSO normalisation. Default: OFF (normalisation ON).')
    parser.add_argument("--correlation_method",
                        choices=["pearson", "spearman", "kendall"],
                        default="spearman",
                        help="Correlation metric for redundancy filter. Default spearman due to data distributions"
                    )

    
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


    # --- Outlier trimming (per-object) ---
    parser.add_argument(
        '--trim_objects',
        action='store_true',
        help='Trim per-object outliers by keeping the most central fraction within groups.'
    )
    parser.add_argument(
        '--keep_central_fraction',
        type=float,
        default=0.90,
        help='Fraction of objects to keep within each group (e.g., 0.90 keeps central 90%%).'
    )
    parser.add_argument(
        '--trim_scope',
        choices=['per_well', 'per_plate', 'global'],
        default='per_well',
        help='Grouping for robust centre estimation.'
    )
    parser.add_argument(
        '--trim_metric',
        choices=['q', 'q95', 'l2', 'max'],
        default='q95',
        help='How to aggregate per-feature |z| into one distance per object. '
    )
    parser.add_argument(
        '--trim_quantile',
        type=float,
        default=0.90,
        help='Quantile in (0, 1] used when --trim_metric is q/q95 (default: 0.90). '
    )

    parser.add_argument(
        '--trim_stage',
        choices=['pre_impute', 'post_impute'],
        default='pre_impute',
        help='Whether to trim before or after imputation.'
    )
    parser.add_argument(
        '--trim_nan_feature_frac',
        type=float,
        default=0.7,
        help='Drop features from the distance calculation if > this fraction is NaN within the group.'
    )
    parser.add_argument(
        '--trim_min_features_per_cell',
        type=float,
        default=0.25,
        help='Require at least this fraction of the usable features present for a cell; else the cell is dropped.'
    )
    parser.add_argument(
        '--trim_qc_file',
        type=str,
        default=None,
        help='Optional path to write a small QC table (TSV) with per-group trimming stats.'
    )

    # Categorical-like feature dropping: default ON, with an opt-out flag
    cat_group = parser.add_mutually_exclusive_group()
    cat_group.add_argument(
        "--drop_categorical_like",
        dest="drop_categorical_like",
        action="store_true",
        help="Detect and drop numeric features that look categorical/encoded (binary, few levels, etc.). (default: on)"
    )
    cat_group.add_argument(
        "--no_drop_categorical_like",
        dest="drop_categorical_like",
        action="store_false",
        help="Disable categorical-like detection/drop."
    )
    parser.set_defaults(drop_categorical_like=True)


    parser.add_argument(
        "--categorical_max_levels",
        type=int,
        default=20,
        help="Absolute max distinct non-null values to consider a column categorical-like (default: 20)."
    )
    parser.add_argument(
        "--categorical_unique_ratio",
        type=float,
        default=0.005,
        help="If (unique_non_null / non_null) <= this and unique_non_null <= 100, flag as categorical-like (default: 0.005)."
    )
    parser.add_argument(
        "--categorical_integer_levels",
        type=int,
        default=15,
        help="If all values are (near-)integers and distinct levels <= this, flag (default: 15)."
    )
    parser.add_argument(
        "--categorical_topk",
        type=int,
        default=3,
        help="K for top-K mass heuristic (default: 3)."
    )
    parser.add_argument(
        "--categorical_topk_mass",
        type=float,
        default=0.99,
        help="If the top-K levels cover >= this fraction and levels <= categorical_integer_levels, flag (default: 0.99)."
    )
    parser.add_argument(
        "--categorical_protect",
        type=lambda s: [x.strip() for x in s.split(",")] if s else None,
        default=None,
        help="Comma-separated feature names to protect from categorical-like dropping."
    )
    parser.add_argument(
        "--categorical_report",
        type=str,
        default=None,
        help="Optional path to write diagnostics of categorical-like detection (TSV). "
            "Default: <output_file>_categorical_like.tsv"
    )
    parser.add_argument(
        '--no_prefix_from_filename',
        action='store_true',
        help='If set, do not prefix feature columns from the source filename (default: prefixing ON).'
    )
    parser.add_argument(
        '--prefix_separator',
        type=str,
        default='__',
        help='Separator between the derived prefix and the original feature name (default: "__").'
    )


    parser.add_argument(
    "--drop_by_name",
    action="store_true",
    help="If set, drop feature columns whose names match --drop_by_name_pat (case-insensitive). Default: off."
    )
    parser.add_argument(
        "--drop_by_name_pat",
        type=str,
        default=r"(?:^|_)(?:Treatment|Number|Child|Paren|Location_[XYZ]|ZernikePhase|Euler|Plate|Well|Field|Center_(?:Z|X|Y)|no_|fn_|Volume)(?:_|$)",
        help="Regex used to flag feature columns to drop when --drop_by_name is set (case-insensitive)."
    )


    parser.add_argument('--no_compress_output',
                        action='store_true',
                        help='If set, do NOT compress the output file. Default: output will be compressed (.gz if filename ends with .gz).')
    parser.add_argument(
                        '--force_library_tag',
                        action='store_true',
                        help='If set with --library, overwrite any existing Library values with the given tag.'
                    )

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


def detect_categorical_like_features(
    df: pd.DataFrame,
    *,
    candidates: Optional[list[str]] = None,
    max_levels: int = 20,
    unique_ratio: float = 0.005,
    integer_levels_cap: int = 15,
    topk: int = 3,
    topk_mass_thr: float = 0.99,
    protect: Optional[Iterable[str]] = None,
    integer_tol: float = 1e-6,
    logger: Optional[logging.Logger] = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Identify numeric columns that behave like categorical/encoded features.

    Heuristics (flag a column if ANY is true):
      1) Low distinct count: nunique_non_null <= max_levels
      2) Low distinct ratio: (nunique_non_null / non_null) <= unique_ratio AND nunique_non_null <= 100
      3) Binary: values subset of {0,1} (allowing NaN)
      4) Small integer code: all values (within tol) are integers AND levels <= integer_levels_cap
      5) Mass concentration: top-K levels cover >= topk_mass_thr AND levels <= integer_levels_cap

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    candidates : list[str], optional
        Explicit numeric columns to check. If None, infer from df (numeric dtypes).
    max_levels : int
        Absolute distinct count threshold for categorical-like.
    unique_ratio : float
        Distinct / non-null ratio threshold.
    integer_levels_cap : int
        If integer-valued and #levels <= this, flag.
    topk : int
        K for the top-K mass heuristic.
    topk_mass_thr : float
        Threshold for the top-K mass heuristic (0-1).
    protect : Iterable[str], optional
        Feature names never to flag.
    integer_tol : float
        Tolerance for near-integer detection.
    logger : logging.Logger, optional
        Logger.

    Returns
    -------
    (list[str], pd.DataFrame)
        List of flagged column names and a diagnostics DataFrame with per-column metrics.
    """
    logger = logger or logging.getLogger("categorical_like")
    protect = set(protect or [])
    if candidates is None:
        candidates = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    flagged = []

    for col in candidates:
        s = df[col]
        non_null = s.notna().sum()
        if non_null == 0:
            # all NaN is handled elsewhere; skip here
            continue

        # fast metrics
        nunique = s.nunique(dropna=True)
        uniq_ratio = nunique / non_null

        # value distribution (for binary/topK)
        vc = s.value_counts(dropna=True)
        topk_mass = (vc.head(topk).sum() / non_null) if non_null else 0.0

        # integer-ish?
        v = s.dropna().to_numpy()
        all_integer = np.all(np.isclose(v, np.round(v), atol=integer_tol))

        # binary check (allow 0/1 or 1/0 with any floats close to them)
        unique_vals = sorted(np.unique(np.round(v, 6)))
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0})

        triggers = []
        if nunique <= max_levels:
            triggers.append(f"≤{max_levels} unique")
        if (uniq_ratio <= unique_ratio) and (nunique <= 100):
            triggers.append(f"unique_ratio≤{unique_ratio:g}")
        if is_binary:
            triggers.append("binary")
        if all_integer and nunique <= integer_levels_cap:
            triggers.append(f"integer_levels≤{integer_levels_cap}")
        if (topk_mass >= topk_mass_thr) and (nunique <= integer_levels_cap):
            triggers.append(f"top{topk}_mass≥{topk_mass_thr:.2f}")

        flagged_now = len(triggers) > 0 and (col not in protect)

        if flagged_now:
            flagged.append(col)

        # small sample of levels (stringified) for audit
        sample_levels = ", ".join(map(lambda x: str(x)[:12], vc.head(6).index.tolist()))

        rows.append({
            "feature": col,
            "non_null": int(non_null),
            "nunique": int(nunique),
            "unique_ratio": float(uniq_ratio),
            "is_binary": bool(is_binary),
            "all_integer": bool(all_integer),
            "topk_mass": float(topk_mass),
            "flagged": bool(flagged_now),
            "triggers": ";".join(triggers),
            "sample_levels": sample_levels,
        })

    diag = pd.DataFrame(rows).sort_values(["flagged", "nunique"], ascending=[False, True])
    logger.info(
        "Categorical-like detection: checked %d columns, flagged %d.",
        len(candidates), len(flagged)
    )
    return flagged, diag


def derive_object_prefix(filename: str) -> str:
    """
    Derive a column prefix from a CellProfiler object filename.

    Rules
    -----
    - Strip '.csv' or '.csv.gz'
    - Take the last '_' token (e.g., 'MyExpt_test_FilteredNuclei' → 'FilteredNuclei')
    - Drop a leading 'Filtered' (case-insensitive), so 'FilteredNuclei' → 'Nuclei'

    Parameters
    ----------
    filename : str
        Basename of the file being loaded.

    Returns
    -------
    str
        Clean prefix to prepend to feature columns.
    """
    base = filename
    if base.endswith(".csv.gz"):
        base = base[:-7]
    elif base.endswith(".csv"):
        base = base[:-4]
    last_token = base.split("_")[-1]
    # Remove leading 'Filtered' if present
    prefix = re.sub(r"(?i)^Filtered", "", last_token).strip()
    return prefix or last_token


def prefix_feature_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
    sep: str = "__",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Prefix feature columns with '<prefix><sep>' while protecting metadata-like columns.

    Protected (never prefixed)
    --------------------------
    - 'ImageNumber', 'ObjectNumber'
    - Plate/Well variants: 'Plate', 'Well', 'Plate_Metadata', 'Well_Metadata',
      'Metadata_Plate', 'Metadata_Well', 'Image_Metadata_Plate', 'Image_Metadata_Well'
    - Any column starting with 'Metadata_', 'FileName_', 'PathName_', or 'URL_'

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame just read from a CellProfiler object CSV.
    prefix : str
        Prefix derived from the filename (e.g., 'Nuclei', 'Acrosome').
    sep : str, optional
        Separator between prefix and original column name (default: '__').
    logger : logging.Logger, optional
        Logger for diagnostics.

    Returns
    -------
    pandas.DataFrame
        DataFrame with feature columns renamed.
    """
    logger = logger or logging.getLogger("prefix_cols")
    protected_exact = {
        "ImageNumber", "ObjectNumber",
        "Plate", "Well", "Plate_Metadata", "Well_Metadata",
        "Metadata_Plate", "Metadata_Well",
        "Image_Metadata_Plate", "Image_Metadata_Well",
    }
    protected_prefixes = ("Metadata_", "FileName_", "PathName_", "URL_")

    # If some columns are already prefixed with this prefix + sep, avoid double-prefixing.
    already_prefixed = f"{prefix}{sep}"
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if (
            (col in protected_exact)
            or col.startswith(protected_prefixes)
            or col.startswith(already_prefixed)
        ):
            continue
        rename_map[col] = f"{prefix}{sep}{col}"

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(
            "Prefixed %d columns with '%s%s' (protected %d).",
            len(rename_map),
            prefix,
            sep,
            len(df.columns) - len(rename_map),
        )
    else:
        logger.info("No columns needed prefixing for prefix '%s'.", prefix)

    return df



def attach_plate_well_with_fallback(
    *,
    obj_df: pd.DataFrame,
    input_dir: Path,
    obj_filename: str,
    image_cache: Optional[pd.DataFrame],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Attach plate/well metadata to an object-level table using an image table.

    Behaviour
    ---------
    1) If 'Plate_Metadata' and 'Well_Metadata' are already present in obj_df, return as-is.
    2) Try a sibling image table: <basename>_Image.csv[.gz].
    3) Otherwise, if exactly one image table exists in the folder (e.g. 'MyExpt_test_Image.csv.gz'),
       use it as a fallback for all object files (cached to avoid re-reading).
    4) Left-join on 'ImageNumber' and standardise column names to 'Plate_Metadata' and
       'Well_Metadata' if equivalents are found.

    Parameters
    ----------
    obj_df : pandas.DataFrame
        Object-level DataFrame (must contain 'ImageNumber' for merging).
    input_dir : pathlib.Path
        Directory containing object and image CSVs.
    obj_filename : str
        Filename of the object CSV (used to look for a sibling image file).
    image_cache : pandas.DataFrame or None
        Cached image table (subset of columns) to reuse across files. If None, this function
        may populate it when a single-folder image is discovered.
    logger : logging.Logger
        Logger for status and warnings.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame or None)
        Tuple of (possibly augmented) object DataFrame and the (possibly updated) image_cache.
    """
    # Already have plate/well? Nothing to do.
    if ("Plate_Metadata" in obj_df.columns) and ("Well_Metadata" in obj_df.columns):
        return obj_df, image_cache

    # We need ImageNumber to merge
    if "ImageNumber" not in obj_df.columns:
        logger.error("Object file '%s' has no 'ImageNumber' column; cannot attach image metadata.", obj_filename)
        return obj_df, image_cache

    # Candidate plate/well names within image table
    plate_cands = ["Plate_Metadata", "Metadata_Plate", "Plate", "plate", "Image_Metadata_Plate"]
    well_cands  = ["Well_Metadata", "Metadata_Well", "Well", "well", "Image_Metadata_Well"]

    # 1) Try sibling: <basename>_Image.csv[.gz]
    base = obj_filename
    for suff in (".csv.gz", ".csv"):
        if base.endswith(suff):
            base = base[: -len(suff)]
            break
    sibling_candidates = [input_dir / f"{base}_Image.csv.gz", input_dir / f"{base}_Image.csv"]

    image_path = next((p for p in sibling_candidates if p.exists()), None)

    # 2) Otherwise use (or build) folder-wide fallback cache if there is exactly one image file
    if image_path is None:
        if image_cache is None:
            # Find all image tables in the folder
            image_files = []
            for pat in ("*Image.csv.gz", "*_Image.csv.gz", "*Image.csv", "*_Image.csv"):
                image_files.extend(sorted(input_dir.glob(pat)))
            # Deduplicate
            seen = set()
            uniq = [p for p in image_files if not (p in seen or seen.add(p))]
            if len(uniq) == 1:
                logger.info("Using single-folder image table as fallback: %s", uniq[0].name)
                # Read a small peek to detect which plate/well columns exist
                head = pd.read_csv(uniq[0], nrows=5, low_memory=False)
                plate_col = next((c for c in plate_cands if c in head.columns), None)
                well_col  = next((c for c in well_cands  if c in head.columns), None)
                usecols = ["ImageNumber"] + [c for c in (plate_col, well_col) if c]
                if len(usecols) == 1:
                    logger.warning("Fallback image '%s' lacks plate/well columns; proceeding without attachment.", uniq[0].name)
                    return obj_df, image_cache
                image_cache = pd.read_csv(uniq[0], usecols=usecols, low_memory=False)
            elif len(uniq) == 0:
                logger.error("No image table found in %s and no sibling for '%s'.", str(input_dir), obj_filename)
                return obj_df, image_cache
            else:
                logger.warning("Multiple image tables found in %s; no unique fallback. Candidates: %s",
                               str(input_dir), ", ".join(p.name for p in uniq[:10]))
                return obj_df, image_cache
        # Use the cached fallback
        image_df = image_cache
    else:
        # Read only necessary columns from sibling file
        head = pd.read_csv(image_path, nrows=5, low_memory=False)
        plate_col = next((c for c in plate_cands if c in head.columns), None)
        well_col  = next((c for c in well_cands  if c in head.columns), None)
        usecols = ["ImageNumber"] + [c for c in (plate_col, well_col) if c]
        if len(usecols) == 1:
            logger.warning("Image file '%s' lacks plate/well columns; proceeding without attachment.", image_path.name)
            return obj_df, image_cache
        image_df = pd.read_csv(image_path, usecols=usecols, low_memory=False)

    # Merge and standardise names
    merged = obj_df.merge(image_df, how="left", on="ImageNumber", suffixes=("", "_img"))

    if "Plate_Metadata" not in merged.columns:
        cand = next((c for c in plate_cands if c in merged.columns), None)
        if cand:
            merged = merged.rename(columns={cand: "Plate_Metadata"})
    if "Well_Metadata" not in merged.columns:
        cand = next((c for c in well_cands if c in merged.columns), None)
        if cand:
            merged = merged.rename(columns={cand: "Well_Metadata"})

    return merged, image_cache

def trim_objects_by_robust_distance(
    *,
    df: pd.DataFrame,
    groupby: Optional[list[str]],
    feature_cols: Optional[list[str]],
    keep_central_fraction: float = 0.90,
    metric: str = "q95",
    nan_feature_frac: float = 0.5,
    min_features_per_cell: float = 0.5,
    trim_quantile: float = 0.90,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trim per-object outliers by keeping the most central fraction within groups.

    The method is robust z-distance trimming:
        - Within each group (e.g., per well), compute per-feature median and MAD.
        - Convert values to robust z-scores: (x - median) / MAD.
        - Aggregate |z| across features to one distance per object using:
            'q95' (recommended), 'l2', or 'max'.
        - Keep objects whose distance is at or below the group-specific
    
    quantile implied by `keep_central_fraction`.
              trim_quantile : float, optional
        Quantile in (0, 1] used when `metric == "q95"`. Defaults to 0.90.

    Parameters
    ----------
    df : pandas.DataFrame
        Input per-object table, already cleaned/merged with metadata.
    groupby : list[str] or None
        Column names to define groups (e.g., ['Plate_Metadata', 'Well_Metadata']),
        or None to treat all rows as one group.
    feature_cols : list[str] or None
        Numeric feature columns to consider. If None, they will be inferred
        as numeric columns excluding common metadata fields.
    keep_central_fraction : float, default 0.90
        Fraction of objects to keep within each group (0 < f ≤ 1).
    metric : {'q95', 'l2', 'max'}, default 'q95'
        Aggregation of per-feature |z| into a single per-object distance.
    nan_feature_frac : float, default 0.5
        Drop a feature from the distance calculation within a group if the
        fraction of NaNs in that group exceeds this threshold.
    min_features_per_cell : float, default 0.5
        Require that each object has at least this fraction of the usable
        features non-NaN for the distance calculation; otherwise it is dropped.
    logger : logging.Logger or None
        Logger for messages.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        A tuple of (trimmed_df, qc_df), where qc_df reports, per group:
        n_rows, n_kept, frac_kept, cutoff_distance.

    Notes
    -----
    - Features with MAD==0 within a group are skipped for distance computation.
    - If a group ends up with no usable features, the group is returned unchanged.
    - Ties at the cutoff are kept (stable behaviour).
    """
    logger = logger or logging.getLogger("trim_logger")
    # At the start of trim_objects_by_robust_distance(...)
    if groupby:
        missing = [g for g in groupby if g not in df.columns]
        if missing:
            (logger or logging.getLogger("trim_logger")
            ).warning("Trimming: missing groupby columns %s; falling back to global.", missing)
            groupby = None


    if not (0.0 < keep_central_fraction <= 1.0):
        raise ValueError("keep_central_fraction must be in (0, 1].")
    
    if not (0.0 < trim_quantile <= 1.0):
        raise ValueError("trim_quantile must be in (0, 1].")

    # Infer feature columns if not provided
    if feature_cols is None:
        meta_like = {"cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata", "ImageNumber", "ObjectNumber"}
        feature_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in meta_like
        ]
    if not feature_cols:
        logger.warning("No numeric feature columns to use for trimming; returning input unchanged.")
        return df, pd.DataFrame(columns=["group_key", "n_rows", "n_kept", "frac_kept", "cutoff_distance"])

    # Group iterator
    if groupby and len(groupby) > 0:
        groups = df.groupby(groupby, sort=False, observed=False, group_keys=False)
    else:
        # Single pseudo-group covering all rows
        groups = [((), df)]

    kept_masks = []
    qc_rows = []

    # Process each group independently
    for key, g in (groups if isinstance(groups, list) else groups):
        g_index = g.index

        # Select usable features in this group
        sub = g[feature_cols].astype('float32', copy=False)

        # Drop features that are too missing in this group
        missing_frac = sub.isna().mean(axis=0).values  # ndarray
        use_feat_mask = missing_frac <= float(nan_feature_frac)
        use_feats = [f for f, ok in zip(feature_cols, use_feat_mask) if ok]

        if not use_feats:
            # Cannot compute distances; keep group unchanged
            kept_masks.append(pd.Series(True, index=g_index))
            cutoff_val = np.nan
            qc_rows.append((key, int(g.shape[0]), int(g.shape[0]), 1.0, cutoff_val))
            continue

        X = sub[use_feats].values  # 2D float32 with NaNs as needed

        # Medians and MADs per feature within the group (ignore NaNs)
        med = np.nanmedian(X, axis=0)
        abs_dev = np.abs(X - med)
        mad = np.nanmedian(abs_dev, axis=0)

        # Skip features with MAD==0
        nonzero = mad > 0
        if not np.any(nonzero):
            kept_masks.append(pd.Series(True, index=g_index))
            cutoff_val = np.nan
            qc_rows.append((key, int(g.shape[0]), int(g.shape[0]), 1.0, cutoff_val))
            continue

        med = med[nonzero]
        mad = mad[nonzero]
        Xnz = X[:, nonzero]

        # Robust z
        Z = (Xnz - med) / mad

        # Count available features per cell
        valid_counts = np.sum(~np.isnan(Z), axis=1)
        min_req = int(np.ceil(float(min_features_per_cell) * Z.shape[1]))
        # Cells with too-few features become NaN distance → filtered out below
        too_few = valid_counts < max(min_req, 1)

        # Distance per object
        with np.errstate(invalid='ignore', divide='ignore'):
            if metric in ("q", "q95"):
                dist = np.nanquantile(np.abs(Z), trim_quantile, axis=1)
            elif metric == "l2":
                dist = np.sqrt(np.nanmean(Z * Z, axis=1))
            elif metric == "max":
                dist = np.nanmax(np.abs(Z), axis=1)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Invalidate rows with too few features
        dist[too_few] = np.nan

        # If everything is NaN, keep group unchanged
        finite_mask = np.isfinite(dist)
        if not np.any(finite_mask):
            kept_masks.append(pd.Series(True, index=g_index))
            cutoff_val = np.nan
            qc_rows.append((key, int(g.shape[0]), int(g.shape[0]), 1.0, cutoff_val))
            continue

        # Keep central fraction
        cutoff_val = np.nanquantile(dist[finite_mask], float(keep_central_fraction))
        keep_mask = (dist <= cutoff_val) | ~finite_mask  # keep NaN-distance rows? No: drop them
        # We will drop NaN-distance rows:
        keep_mask = (dist <= cutoff_val) & finite_mask

        n_rows = int(g.shape[0])
        n_kept = int(np.sum(keep_mask))
        frac_kept = (n_kept / n_rows) if n_rows else 1.0

        kept_masks.append(pd.Series(keep_mask, index=g_index))
        qc_rows.append((key, n_rows, n_kept, frac_kept, float(cutoff_val)))

    # Combine masks
    keep = pd.concat(kept_masks).reindex(df.index).fillna(False).astype(bool)
    trimmed = df.loc[keep].copy()

    # Build QC frame
    def _key_to_tuple(k):
        if isinstance(k, tuple):
            return k
        if k == ():
            return ("__global__",)
        return (k,)

    qc_cols = (groupby if groupby and len(groupby) > 0 else ["__global__"]) + \
              ["n_rows", "n_kept", "frac_kept", "cutoff_distance"]
    qc_records = []
    for k, n_rows, n_kept, frac_kept, cutoff in qc_rows:
        k_tuple = _key_to_tuple(k)
        qc_records.append(tuple(k_tuple) + (n_rows, n_kept, frac_kept, cutoff))
    qc_df = pd.DataFrame.from_records(qc_records, columns=qc_cols)

    logger.info(
        "Trimming complete: kept %d / %d objects (%.1f%%).",
        trimmed.shape[0], df.shape[0],
        100.0 * (trimmed.shape[0] / max(df.shape[0], 1))
    )
    return trimmed, qc_df



def correlation_filter_smart(
    X: pd.DataFrame,
    threshold: float = 0.99,
    strategy: str = "variance",
    protect: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
    method: str = "spearman"
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

        method : {"pearson","spearman","kendall"}, default "spearman"
        Correlation metric.

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

    # Compute absolute correlation using requested method (numeric only)
    corr = X_num.corr(method=method).abs().fillna(0.0)
    logger.info("Using %s correlation method for correlation filter.", method)

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




def impute_missing(
    df: pd.DataFrame,
    method: str = "knn",
    knn_neighbours: int = 5,
    logger=None,
    scope: str = "global",
    max_cells: int = 100_000_000,
    max_features: int = 300_000,
) -> pd.DataFrame:

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

        Impute missing values for numeric columns.

    - median: fully vectorised (global / per-plate / per-well) via groupby.transform
              + a global safety fill for any leftover NaNs.
    - knn:    uses sklearn KNNImputer; if scope != global, runs per group to
              reduce memory + improve locality. Falls back to median when groups
              are too small for KNN.

    'scope' requires columns:
        per_plate -> Plate_Metadata
        per_well  -> Plate_Metadata, Well_Metadata
        

    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed numeric columns.
    """
    logger = logger or logging.getLogger("impute_logger")
    n_rows, n_cols_total = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if n_rows > max_cells or len(numeric_cols) > max_features:
        logger.warning(
            "Imputation skipped: dataset too large (%s rows, %s numeric cols; "
            "thresholds: %s rows, %s cols).", f"{n_rows:,}", len(numeric_cols),
            f"{max_cells:,}", max_features
        )
        return df

    if not numeric_cols:
        logger.warning("No numeric columns found for imputation.")
        return df

    # Drop columns that are all-NaN (imputers cannot recover these)
    all_nan = [c for c in numeric_cols if df[c].isna().all()]
    if all_nan:
        logger.warning("Dropping %d all-NaN columns before imputation (e.g. %s...)",
                       len(all_nan), all_nan[:5])
        df = df.drop(columns=all_nan)
        numeric_cols = [c for c in numeric_cols if c not in all_nan]
        if not numeric_cols:
            return df

    # Replace inf/-inf/huge with NaN, and downcast for speed
    df.loc[:, numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # -------- median (fully vectorised) --------
    if method == "median":
        logger.info("Median imputation (%s scope) on %d numeric columns.", scope, len(numeric_cols))

        if scope == "global":
            med = df[numeric_cols].median(numeric_only=True)
            df.loc[:, numeric_cols] = df[numeric_cols].fillna(med)

        elif scope == "per_plate":
            if "Plate_Metadata" not in df.columns:
                raise ValueError("per_plate median imputation requires 'Plate_Metadata'.")
            med = df.groupby("Plate_Metadata", observed=False)[numeric_cols].transform("median")
            df.loc[:, numeric_cols] = df[numeric_cols].where(~df[numeric_cols].isna(), med)

        elif scope == "per_well":
            keys = ["Plate_Metadata", "Well_Metadata"]
            missing = [k for k in keys if k not in df.columns]
            if missing:
                raise ValueError(f"per_well median imputation requires {missing}.")
            med = df.groupby(keys, observed=False)[numeric_cols].transform("median")
            df.loc[:, numeric_cols] = df[numeric_cols].where(~df[numeric_cols].isna(), med)

        else:
            raise ValueError(f"Unknown impute_scope: {scope}")

        # global safety fill for any columns that were entirely NaN within a group
        global_med = df[numeric_cols].median(numeric_only=True)
        df.loc[:, numeric_cols] = df[numeric_cols].fillna(global_med).astype(np.float32)
        return df

    # -------- knn (vectorised inside sklearn; optionally grouped) --------
    if method == "knn":
        from sklearn.impute import KNNImputer
        logger.info("KNN imputation (k=%d, scope=%s) on %d numeric columns.",
                    knn_neighbours, scope, len(numeric_cols))

        def _knn_block(block: pd.DataFrame) -> pd.DataFrame:
            # if too small for KNN, fallback to median (vectorised)
            if block.shape[0] <= 1:
                return block
            k = min(knn_neighbours, max(1, block.shape[0] - 1))
            try:
                imputer = KNNImputer(n_neighbors=k)
                vals = imputer.fit_transform(block[numeric_cols].astype(np.float32))
                block.loc[:, numeric_cols] = vals.astype(np.float32)
                return block
            except Exception as e:
                # fallback to median within the block
                med = block[numeric_cols].median(numeric_only=True)
                block.loc[:, numeric_cols] = block[numeric_cols].fillna(med).astype(np.float32)
                logger.warning("KNN failed in a block (k=%d): %s; fell back to median.", k, e)
                return block

        if scope == "global":
            df = _knn_block(df)
        elif scope == "per_plate":
            if "Plate_Metadata" not in df.columns:
                raise ValueError("per_plate KNN requires 'Plate_Metadata'.")
            parts = []
            for _, g in df.groupby("Plate_Metadata", sort=False, observed=False):
                parts.append(_knn_block(g))
            df = pd.concat(parts, axis=0, ignore_index=False)
        elif scope == "per_well":
            keys = ["Plate_Metadata", "Well_Metadata"]
            missing = [k for k in keys if k not in df.columns]
            if missing:
                raise ValueError(f"per_well KNN requires {missing}.")
            parts = []
            for _, g in df.groupby(keys, sort=False, observed=False):
                parts.append(_knn_block(g))
            df = pd.concat(parts, axis=0, ignore_index=False)
        else:
            raise ValueError(f"Unknown impute_scope: {scope}")

        return df

    # -------- none --------
    logger.info("Imputation skipped (method=none).")
    return df    


def set_library_and_treatment(
    df: pd.DataFrame,
    library_tag: str | None = None,
    force_library_tag: bool = False,
    logger=None,
) -> pd.DataFrame:
    """
    Set dataset-level `Library` tags with CLI precedence and derive `Treatment`.

    Precedence:
    1) If `library_tag` is provided, apply it *before* any auto-filling:
       - If `force_library_tag` is True: set all rows' `Library = library_tag`.
       - Else: only fill rows where `Library` is missing/blank.
    2) Derive `Treatment` from metadata ('control' for DMSO, else 'compound').
    3) For any rows where `Library` is still missing, fill from `Treatment`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing Cell Painting features and metadata.
    library_tag : str | None, optional
        Dataset tag from CLI (e.g., 'STB_V1'). If None, step (1) is skipped.
    force_library_tag : bool, optional
        If True, overwrite any existing `Library` values with `library_tag`.
        If False (default), only fill missing/blank `Library`.
    logger : logging.Logger | None, optional
        Logger for progress messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with preserved/filled `Library` and a new `Treatment` column.

    Notes
    -----
    - This function does not modify `cpd_type` logic elsewhere; it just uses it.
    - It treats empty strings and 'na'/'NA' as missing for `Library`.
    """
    # Normalise presence of columns
    if "Library" not in df.columns:
        df["Library"] = pd.NA
    else:
        # Treat empty/blank as missing
        lib = df["Library"].astype("string")
        lib = lib.replace({"": pd.NA, " ": pd.NA, "na": pd.NA, "NA": pd.NA})
        df["Library"] = lib

    # 1) CLI precedence
    if library_tag is not None:
        if force_library_tag:
            df["Library"] = str(library_tag)
            if logger:
                logger.info("Forced Library tag to '%s' for all rows.", library_tag)
        else:
            n_before = df["Library"].isna().sum()
            df.loc[df["Library"].isna(), "Library"] = str(library_tag)
            n_after = df["Library"].isna().sum()
            if logger:
                logger.info(
                    "Filled %d missing Library values with CLI tag '%s' (remaining missing: %d).",
                    n_before - n_after, library_tag, n_after
                )

    # 2) Derive Treatment from cpd_id/cpd_type
    #    control = DMSO; compound = everything with a non-DMSO cpd_id
    mask_dmso = pd.Series(False, index=df.index)
    if "cpd_type" in df.columns:
        mask_dmso = mask_dmso | (df["cpd_type"] == "DMSO")
    if "cpd_id" in df.columns:
        mask_dmso = mask_dmso | (df["cpd_id"] == "DMSO")

    mask_has_cpd = df["cpd_id"].notna() if "cpd_id" in df.columns else pd.Series(False, index=df.index)
    mask_has_cpd = mask_has_cpd & ~mask_dmso

    df["Treatment"] = pd.NA
    df.loc[mask_dmso, "Treatment"] = "control"
    df.loc[mask_has_cpd, "Treatment"] = "compound"

    # 3) Final backfill: if Library still missing, use Treatment
    missing_after_cli = df["Library"].isna().sum()
    if missing_after_cli > 0:
        df.loc[df["Library"].isna() & (df["Treatment"].notna()), "Library"] = df.loc[
            df["Library"].isna() & (df["Treatment"].notna()), "Treatment"
        ]
        if logger:
            left = df["Library"].isna().sum()
            logger.info(
                "Backfilled %d remaining Library values from Treatment (still missing: %d).",
                missing_after_cli - left, left
            )

    # Optional: sanity counts
    if logger:
        n_ctrl = (df["Treatment"] == "control").sum()
        n_comp = (df["Treatment"] == "compound").sum()
        logger.info("Treatment counts — control: %d, compound: %d.", n_ctrl, n_comp)

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

    # Set Library column (fill-only; do not overwrite existing values)
    if "Library" in df.columns and "cpd_id" in df.columns:
        mask_dmso = (df["cpd_id"] == "DMSO") | (df.get("cpd_type", pd.Series(False, index=df.index)) == "DMSO")
        mask_has_cpd_id = df["cpd_id"].notna() & (df["cpd_id"] != "DMSO")
        # Only fill missing Library values
        missing_lib = df["Library"].isna()
        df.loc[missing_lib & mask_dmso, "Library"] = "control"
        df.loc[missing_lib & mask_has_cpd_id & ~mask_dmso, "Library"] = "compound"
        if logger:
            n_dmso = (missing_lib & mask_dmso).sum()
            n_compound = (missing_lib & mask_has_cpd_id & ~mask_dmso).sum()
            logger.info(f"Filled Library (missing only): control={n_dmso}, compound={n_compound}.")

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

    csv_files = [f for f in csv_files if not EXCLUDE_PAT.search(f.name)]
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
    image_cache: Optional[pd.DataFrame] = None

    for f in csv_files:
        logger.info(f"Reading file: {f.name}")
        dtype_map = infer_dtypes(f)
        df = pd.read_csv(f, dtype=dtype_map)

        # prefix feature columns from the filename (e.g., Nuclei, Acrosome) ---
        if not args.no_prefix_from_filename:
            _prefix = derive_object_prefix(f.name)
            df = prefix_feature_columns(
                df=df,
                prefix=_prefix,
                sep=args.prefix_separator,
                logger=logger,
            )


        # Add Plate/Well from the matching *_Image file if needed,
        # and coerce to a single, clean pair of Plate_Metadata/Well_Metadata
        df, image_cache = attach_plate_well_with_fallback(
            obj_df=df,
            input_dir=input_dir,
            obj_filename=f.name,
            image_cache=image_cache,
            logger=logger
        )


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

    # 1b) identify catergorical data and remove
    # It would be posible to include the catergorical data but would take a better man than I ... 
    present_meta = {
        "cpd_id", "cpd_type", "Library",
        "Plate_Metadata", "Well_Metadata",
        "ImageNumber", "ObjectNumber"
    }
    numeric_feature_candidates = [
        c for c in cp_df.columns
        if pd.api.types.is_numeric_dtype(cp_df[c]) and c not in present_meta
    ]

    if args.drop_categorical_like and numeric_feature_candidates:
        cat_report_path = (
            args.categorical_report
            if args.categorical_report
            else os.path.splitext(args.output_file)[0] + "_categorical_like.tsv"
        )

        flagged_cols, diag = detect_categorical_like_features(
            cp_df,
            candidates=numeric_feature_candidates,
            max_levels=args.categorical_max_levels,
            unique_ratio=args.categorical_unique_ratio,
            integer_levels_cap=args.categorical_integer_levels,
            topk=args.categorical_topk,
            topk_mass_thr=args.categorical_topk_mass,
            protect=args.categorical_protect or [],
            logger=logger,
        )

        # Write audit
        try:
            diag.to_csv(cat_report_path, sep=args.sep, index=False)
            logger.info("Wrote categorical-like diagnostics to %s", cat_report_path)
        except Exception as e:
            logger.warning("Failed to write categorical-like diagnostics: %s", e)

        if flagged_cols:
            logger.info("Dropping %d categorical-like features (e.g. %s%s).",
                        len(flagged_cols),
                        ", ".join(flagged_cols[:8]),
                        "..." if len(flagged_cols) > 8 else "")
            cp_df = cp_df.drop(columns=flagged_cols, errors="ignore")

            if 'per_object_df' in locals() and per_object_df is not None:
                keep_po = [c for c in per_object_df.columns if c not in flagged_cols]
                per_object_df = per_object_df[keep_po]
    else:
        logger.info("Categorical-like removal disabled or no numeric candidates found.")

        


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
    logger.info("Metadata well values: %s unique (NaNs ignored).",
                int(meta_df[meta_well_col].nunique(dropna=True))
            )



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
    logger.info("Metadata well values: %s unique (NaNs ignored).",
                int(meta_df[meta_well_col].nunique(dropna=True))
    )
    log_memory_usage(logger, prefix=" [After metadata loaded ] ")
    gc.collect()

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

    # --- 1) Audit duplicates by Plate x Well ---
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

    # Preserve dataset tag and derive Treatment right after merge
    merged_df = set_library_and_treatment(
        df=merged_df,
        library_tag=args.library,
        force_library_tag=getattr(args, "force_library_tag", False),
        logger=logger,
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

    # --- 5b. Optional per-object trimming (robust z-distance) ---
    if args.trim_objects and args.trim_stage == "pre_impute":
        if args.trim_scope == "per_well":
            trim_groups = ["Plate_Metadata", "Well_Metadata"]
        elif args.trim_scope == "per_plate":
            trim_groups = ["Plate_Metadata"]
        else:
            trim_groups = None  # global

        # Define feature columns (numeric, excluding metadata fields)

        # Right before building trim_features in main()
        present_meta = set(merged_df.columns)
        meta_like = present_meta.intersection({
            "cpd_id", "cpd_type", "Library",
            "Plate_Metadata", "Well_Metadata",
            "ImageNumber", "ObjectNumber"
        })
        trim_features = [
            c for c in merged_df.columns
            if pd.api.types.is_numeric_dtype(merged_df[c]) and c not in meta_like
        ]
        logger.info(f"Trimming objects using features: {len(trim_features)} numeric columns.")

        merged_df, trim_qc = trim_objects_by_robust_distance(
            df=merged_df,
            groupby=trim_groups,
            feature_cols=trim_features,
            keep_central_fraction=args.keep_central_fraction,
            metric=args.trim_metric,
            nan_feature_frac=args.trim_nan_feature_frac,
            min_features_per_cell=args.trim_min_features_per_cell,
            trim_quantile=args.trim_quantile,
            logger=logger,
        )

        if args.trim_qc_file:
            # Always write TSV (never comma), and do not compress the tiny QC
            trim_qc.to_csv(args.trim_qc_file, sep=args.sep, index=False)
            logger.info("Wrote trimming QC to %s", args.trim_qc_file)
    else:
        logger.info("Per-object trimming not requested at this stage.")
    log_memory_usage(logger, prefix=" [After trimming ] ")


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
        merged_df = impute_missing(merged_df,
                                    method=args.impute,
                                    knn_neighbours=args.knn_neighbours,
                                    logger=logger,
                                    scope=args.impute_scope,
                                )
    else:
        logger.info("Imputation skipped (impute=none).")



    # --- 6b. Optional per-object trimming (post-impute) ---
    if args.trim_objects and args.trim_stage == "post_impute":
        if args.trim_scope == "per_well":
            trim_groups = ["Plate_Metadata", "Well_Metadata"]
        elif args.trim_scope == "per_plate":
            trim_groups = ["Plate_Metadata"]
        else:
            trim_groups = None  # global

        present_meta = set(merged_df.columns)
        meta_like = present_meta.intersection({
            "cpd_id", "cpd_type", "Library",
            "Plate_Metadata", "Well_Metadata",
            "ImageNumber", "ObjectNumber"
        })
        trim_features = [
            c for c in merged_df.columns
            if pd.api.types.is_numeric_dtype(merged_df[c]) and c not in meta_like
        ]
        logger.info(f"[post-impute] Trimming objects using {len(trim_features)} numeric columns.")
        merged_df, trim_qc = trim_objects_by_robust_distance(
            df=merged_df,
            groupby=trim_groups,
            feature_cols=trim_features,
            keep_central_fraction=args.keep_central_fraction,
            metric=args.trim_metric,
            nan_feature_frac=args.trim_nan_feature_frac,
            min_features_per_cell=args.trim_min_features_per_cell,
            trim_quantile=args.trim_quantile,
            logger=logger,
        )
        if args.trim_qc_file:
            trim_qc.to_csv(args.trim_qc_file, sep=args.sep, index=False)
            logger.info("Wrote post-impute trimming QC to %s", args.trim_qc_file)
        


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
        meta_cols = ["cpd_id", "cpd_type", "Library",
                    "ImageNumber", "ObjectNumber", "Number_Object_Number"]
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

    # Exclude all ID / bookkeeping columns from feature selection
    exclude_cols = [
        "cpd_id", "cpd_type", "Library",
        "Plate_Metadata", "Well_Metadata",
        "ImageNumber", "ObjectNumber", "Number_Object_Number"
    ]

    # Also drop any accidental index columns like 'Unnamed: 0'
    unnamed = [c for c in fs_df.columns if re.match(r"^Unnamed:\s*\d+$", str(c))]
    if unnamed:
        logger.info(f"Dropping index-like columns before FS: {unnamed}")
        fs_df = fs_df.drop(columns=unnamed, errors="ignore")

    present_metadata = [c for c in exclude_cols if c in fs_df.columns]

    # Start with numeric, non-metadata columns
    feature_cols = [
        c for c in fs_df.columns
        if c not in present_metadata and pd.api.types.is_numeric_dtype(fs_df[c])
    ]

    # strip by name-patterns you consider non-informative
    #    (case-insensitive; anchors around common separators)

    if args.drop_by_name:
        name_drop_pat = re.compile(args.drop_by_name_pat, re.IGNORECASE)
        drop_by_name = [c for c in feature_cols if name_drop_pat.search(c)]
        if drop_by_name:
            logger.info(
                "Excluding %d feature columns by name pattern (e.g. %s%s).",
                len(drop_by_name),
                ", ".join(drop_by_name[:8]),
                "..." if len(drop_by_name) > 8 else ""
            )
            feature_cols = [c for c in feature_cols if c not in drop_by_name]
    else:
        logger.info("Name-based feature dropping disabled (enable with --drop_by_name).")



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
        # Apply correlation threshold filter
        # This will also generate a PDF with correlation diagnostics
        # Strategy: "mean", "median", "max", "min", "random"

        filtered_corr = correlation_filter_smart(filtered_var,
                                                threshold=args.correlation_threshold,
                                                strategy=args.corr_strategy,
                                                protect=args.protect_features,
                                                logger=logger,
                                                method=args.correlation_method,
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