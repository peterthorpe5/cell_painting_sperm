#!/usr/bin/env python3
# coding: utf-8
"""
Batch comparison: All compounds vs. DMSO — Acrosome Features
===========================================================

For each compound, compare its wells to DMSO wells using a non-parametric test
(Mann-Whitney U by default, or Kolmogorov-Smirnov) plus Earth Mover's Distance.
Outputs per-compound:
- Top N features (TSV + XLSX)
- All significant features by FDR (TSV + XLSX)
- Top acrosome-only features (TSV + XLSX)
- Significant acrosome-only features (TSV + XLSX)
- Acrosome group summary (TSV + XLSX), plus significant subgroup (by min pvalue)

Input
-----
--ungrouped_list : CSV/TSV containing a column 'path' listing well-level tables.

The well-level tables should contain numeric feature columns and at least:
- a compound id column (default: 'cpd_id')
- a compound type/label column (default: 'cpd_type') or specify --dmso_col

DMSO detection:
- Prefer an explicit column with --dmso_col (defaults to cpd_type).
- If that column is missing, fallback to scanning text-like columns for a
  case-insensitive substring match to --dmso_label across each row.

Usage
-----
python batch_acrosome_vs_dmso.py \
  --ungrouped_list dataset_paths.txt \
  --output_dir acrosome_vs_DMSO \
  --dmso_label DMSO \
  --dmso_col cpd_type \
  --test mw \
  --top_features 10 \
  --fdr_alpha 0.05
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from typing import Optional



# -------------------
# Logging & utilities
# -------------------

def setup_logger(log_file: str) -> logging.Logger:
    """
    Configure a logger for both file and console output.

    Parameters
    ----------
    log_file : str
        Path to the log file.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger("acrosome_vs_dmso_logger")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    # Reset handlers if re-run in the same interpreter
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def find_plate_well_in_meta(
    meta: pd.DataFrame,
    requested_keys: str,
    logger: logging.Logger,
) -> tuple[str, str]:
    """
    Resolve Plate and Well column names in a metadata frame.

    Tries the user-provided --merge_keys first (case-insensitive), then falls
    back to common variants. Returns the *actual* column names found.

    Examples tried for plate: Plate, metadata_plate, plate_id, platebarcode
    Examples tried for well:  Well, well_id, wellposition, well_name

    Raises ValueError if not found.
    """
    req = [k.strip() for k in str(requested_keys).split(",") if k.strip()]
    meta_lc = {c.lower(): c for c in meta.columns}

    # First: exact (case-insensitive) match of requested keys
    cand_plate = meta_lc.get(req[0].lower()) if len(req) >= 2 else None
    cand_well  = meta_lc.get(req[1].lower()) if len(req) >= 2 else None

    plate_aliases = ["plate", "metadata_plate", "plate_id", "platebarcode", "plate_name"]
    well_aliases  = ["well", "metadata_well", "well_id", "wellposition", "well_name", "well_address"]

    if not cand_plate:
        for k in plate_aliases:
            if k in meta_lc:
                cand_plate = meta_lc[k]
                break
    if not cand_well:
        for k in well_aliases:
            if k in meta_lc:
                cand_well = meta_lc[k]
                break

    if not cand_plate or not cand_well:
        logger.error(
            "Could not find Plate/Well columns in metadata. "
            "Requested keys=%r. Available columns: %s",
            requested_keys, ", ".join(map(str, meta.columns))
        )
        raise ValueError("Metadata must contain Plate and Well columns. Use --merge_keys PLATE,WELL to point at them.")

    logger.info("Metadata Plate/Well resolved to: %r / %r", cand_plate, cand_well)
    return cand_plate, cand_well



def detect_delimiter(path: str) -> str | None:
    """
    Detect delimiter in a small text file; prefer tab if both appear.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        Detected delimiter: '\\t' or ','.
    """
    candidates = ("\t", ",", ";", "|")
    for sep in candidates:
        try:
            pd.read_csv(path, sep=sep, nrows=50, compression="infer")
            return sep
        except Exception:
            continue
    # Final fallback: let pandas infer with the Python engine
    try:
        pd.read_csv(path, sep=None, nrows=50, engine="python", compression="infer")
        return None  # signal "let pandas infer"
    except Exception:
        return "\t"  # safe default



def read_table_auto(path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Read CSV/TSV with auto delimiter detection.

    Parameters
    ----------
    path : str
        Path to table.
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    sep, comp = _io_hints(path)
    logger.debug(
        "Reading table %s (sep=%r, compression=%r)",
        path, sep if sep is not None else "infer", comp,
    )
    try:
        if sep is None:
            df = pd.read_csv(path, sep=None, engine="python", compression=comp)
        else:
            df = pd.read_csv(path, sep=sep, compression=comp)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        raise

    # sanity check: if we *thought* we knew the sep but got 1 column, warn loudly
    if sep is not None and df.shape[1] == 1:
        logger.warning(
            "File read as a single column (cols=%s). "
            "Check that the extension matches the real delimiter for: %s",
            list(df.columns), path,
        )
    return df



def _io_hints(path: str) -> tuple[str | None, str | None]:
    """
    Decide delimiter and compression strictly from the filename.
    - *.csv           -> comma
    - *.tsv           -> tab
    - *.csv.gz        -> comma + gzip
    - *.tsv.gz        -> tab   + gzip
    Otherwise: sep=None (let pandas infer), compression="infer".
    """
    p = Path(str(path).lower())
    compression = "gzip" if p.suffix == ".gz" or p.suffixes[-1:] == [".gz"] else None

    # handle double suffixes like .csv.gz / .tsv.gz
    suffixes = "".join(p.suffixes)
    if suffixes.endswith(".csv.gz") or suffixes.endswith(".csv"):
        return ",", compression or None
    if suffixes.endswith(".tsv.gz") or suffixes.endswith(".tsv"):
        return "\t", compression or None

    # fallback: let pandas infer
    return None, "infer"


# Columns to always treat as metadata (never as features)
BANNED_FEATURES_EXACT = {
    "ImageNumber",
    "Number_Object_Number",
    "ObjectNumber",
    "TableNumber",
}

# Heuristics for metadata/housekeeping columns (case-insensitive)
BANNED_FEATURES_REGEX = re.compile(
    r"""(?ix)
        ( ^metadata($|_)         # Metadata*, *_Metadata
        | _metadata$
        | ^filename_             # FileName_*
        | ^pathname_             # PathName_*
        | ^url_                  # URL_*
        | ^parent_               # Parent_*
        | ^children_             # Children_*
        | (^|_)imagenumber$      # ImageNumber (allow a prefix_)
        | ^number_object_number$ # Number_Object_Number
        | ^objectnumber$         # ObjectNumber
        | ^tablenumber$          # TableNumber
        )
    """
)

def _is_metadata_like(col: str) -> bool:
    """
    Return True if a column name is metadata/housekeeping and must not be used as a feature.
    """
    cname = str(col)
    if cname in BANNED_FEATURES_EXACT:
        return True
    return bool(BANNED_FEATURES_REGEX.search(cname.lower()))


def ensure_columns(df: pd.DataFrame, required_cols: List[str], fill_value="missing") -> pd.DataFrame:
    """
    Ensure required columns exist; fill with a constant if missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    required_cols : list[str]
        Columns to ensure.
    fill_value : Any
        Value to fill for missing columns.

    Returns
    -------
    pd.DataFrame
        Updated table.
    """
    for col in required_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


def safe_write_tsv(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    """
    Robust TSV writer with clear logging.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : str
        Output path.
    logger : logging.Logger
        Logger.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(map(str, t)) for t in out.columns.to_list()]
    else:
        out.columns = out.columns.map(str)
    out.to_csv(path, sep="\t", index=False)
    logger.info("Wrote %s rows × %s cols -> %s", out.shape[0], out.shape[1], path)


def safe_write_xlsx(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    """
    Robust XLSX writer with clear logging.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : str
        Output path.
    logger : logging.Logger
        Logger.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(map(str, t)) for t in out.columns.to_list()]
    else:
        out.columns = out.columns.map(str)
    out.to_excel(path, index=False)
    logger.info("Wrote %s rows × %s cols -> %s", out.shape[0], out.shape[1], path)


def reorder_and_write(
    df: pd.DataFrame,
    filename: str,
    col_order: List[str],
    logger: logging.Logger,
    filetype: str = "tsv",
) -> None:
    """
    Ensure standard columns (in order), append any extras, and write.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    filename : str
        Output file path.
    col_order : list[str]
        Desired standard column order.
    logger : logging.Logger
        Logger.
    filetype : str
        'tsv' or 'excel'.
    """
    df_out = df.copy()
    for col in col_order:
        if col not in df_out.columns:
            df_out[col] = np.nan

    cols_in_order = [c for c in col_order if c in df_out.columns]
    remaining = [c for c in df_out.columns if c not in cols_in_order]
    df_out = df_out[cols_in_order + remaining]

    if filetype == "tsv":
        safe_write_tsv(df_out, filename, logger)
    elif filetype == "excel":
        safe_write_xlsx(df_out, filename, logger)
    else:
        raise ValueError("filetype must be 'tsv' or 'excel'")

    logger.debug("Columns in %s: %s", filename, df_out.columns.tolist())


# ----------------
# Data preparation
# ----------------

def load_ungrouped_files(
    list_file: str,
    logger: logging.Logger,
    required_cols: list[str] | None = None,
    *,
    drop_all_na: bool = False,
) -> pd.DataFrame:
    """
    Load and vertically concatenate all well-level tables listed in a file.

    This reader is used for the multi-file mode (file-of-files with a 'path'
    column). It reads each table with automatic delimiter detection, optionally
    drops columns that are entirely missing, enforces the presence of required
    identifier columns, and harmonises the inputs by taking the intersection
    of columns (to guard against schema drift).

    Parameters
    ----------
    list_file : str
        CSV/TSV with a column 'path' of file paths.
    logger : logging.Logger
        Logger for status messages.
    required_cols : list[str] | None
        Columns to ensure exist in each table. Missing ones are created and
        filled with the string 'missing'. Default: ['cpd_id', 'cpd_type'].
    drop_all_na : bool
        If True, drop columns that are entirely NA in each input table
        immediately after reading.

    Returns
    -------
    pandas.DataFrame
        Combined well-level table (harmonised by column intersection).

    Raises
    ------
    ValueError
        If the list file lacks a 'path' column, if no files are read, or if
        there are no common columns across inputs.
    """
    if required_cols is None:
        required_cols = ["cpd_id", "cpd_type"]

    logger.info("Reading file-of-files: %s", list_file)
    df_list = read_table_auto(list_file, logger)
    if "path" not in df_list.columns:
        logger.error("Input list must have a column named 'path'.")
        raise ValueError("Missing 'path' column in file-of-files.")

    dfs: list[pd.DataFrame] = []
    for p in df_list["path"]:
        logger.info("Reading well-level data: %s", p)
        tmp = read_table_auto(p, logger)

        if drop_all_na:
            tmp = drop_all_na_columns(tmp, logger=logger)

        # Ensure required ID/label columns exist
        tmp = ensure_columns(tmp, required_cols, fill_value="missing")

        dfs.append(tmp)

    if not dfs:
        raise ValueError("No input files were read.")

    # Harmonise by intersection of columns while preserving a stable order
    common_cols = list(dfs[0].columns)
    for d in dfs[1:]:
        common_cols = [c for c in common_cols if c in d.columns]

    if not common_cols:
        logger.error("No common columns across input files.")
        raise ValueError("No common columns in well-level files.")

    logger.info("%d columns are common to all files.", len(common_cols))

    combined = pd.concat([d[common_cols] for d in dfs], ignore_index=True)
    logger.info("Combined well-level data shape: %s", combined.shape)
    return combined




def select_feature_columns(
    df: pd.DataFrame,
    id_cols: List[str],
    logger: logging.Logger,
    drop_all_nan: bool = True,
    drop_constant: bool = True,
) -> List[str]:
    """
    Select numeric feature columns, excluding ID/label metadata and any
    housekeeping columns (e.g. ImageNumber, Number_Object_Number). Optionally
    drop all-NaN or constant-variance columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    id_cols : list[str]
        Non-feature columns to exclude (e.g. 'cpd_id', 'cpd_type').
    logger : logging.Logger
        Logger.
    drop_all_nan : bool
        Drop columns that are all NaN.
    drop_constant : bool
        Drop columns with zero variance.

    Returns
    -------
    list[str]
        Selected feature columns.
    """
    # Start with numeric candidates
    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Exclude explicit ID/label columns and anything metadata-like
    feature_cols = []
    dropped = []
    non_feature = set(id_cols)

    for c in numeric_candidates:
        if c in non_feature or _is_metadata_like(c):
            dropped.append(c)
        else:
            feature_cols.append(c)

    if dropped:
        logger.info(
            "Excluded %d metadata/housekeeping columns from feature set (first few: %s)",
            len(dropped), ", ".join(map(str, dropped[:10]))
        )

    if drop_all_nan and feature_cols:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if not df[c].isna().all()]
        logger.info("Dropped %d all-NaN feature columns.", before - len(feature_cols))

    if drop_constant and feature_cols:
        before = len(feature_cols)
        variances = df[feature_cols].var(numeric_only=True)
        feature_cols = [c for c in feature_cols if pd.notna(variances.loc[c]) and variances.loc[c] > 0.0]
        logger.info("Dropped %d constant-variance feature columns.", before - len(feature_cols))

    logger.info("Selected %d feature columns.", len(feature_cols))

    if not feature_cols:
        raise ValueError(
            "No usable feature columns remain after excluding metadata/housekeeping "
            "and dropping empty/constant columns. Please inspect inputs."
        )

    return feature_cols



def infer_acrosome_features(feature_cols: List[str], logger: logging.Logger) -> List[str]:
    """
    Identify features containing 'acrosome' (case-insensitive).

    Parameters
    ----------
    feature_cols : list[str]
        Feature column names.
    logger : logging.Logger
        Logger.

    Returns
    -------
    list[str]
        Feature names containing 'acrosome'.
    """
    acrosome_feats = [f for f in feature_cols if "acrosome" in f.lower()]
    logger.info("Found %d acrosome features.", len(acrosome_feats))
    if acrosome_feats:
        logger.debug("Example acrosome features: %s", acrosome_feats[:10])
    return acrosome_feats


def infer_compartments(feature_cols: List[str], logger: logging.Logger) -> Dict[str, List[str]]:
    """
    Infer compartments from feature names using the last underscore token.

    Parameters
    ----------
    feature_cols : list[str]
        Feature names.
    logger : logging.Logger
        Logger.

    Returns
    -------
    dict[str, list[str]]
        Mapping compartment -> feature list.
    """
    group_map: Dict[str, List[str]] = defaultdict(list)
    for feat in feature_cols:
        tokens = feat.split("_")
        compartment = tokens[-1].lower() if len(tokens) > 1 else "unknown"
        group_map[compartment].append(feat)
    logger.info("Inferred %d compartments.", len(group_map))
    return dict(group_map)


def dmso_mask_from_column(
    df: pd.DataFrame,
    label: str,
    col: str,
    logger: logging.Logger,
) -> Tuple[pd.Series, str]:
    """
    Build a boolean mask for DMSO wells based on an explicit column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label : str
        DMSO label (case-insensitive).
    col : str
        Column name to check.
    logger : logging.Logger
        Logger.

    Returns
    -------
    (mask, reason) : (pd.Series, str)
        Boolean mask and reason string for logging.
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index), f"column '{col}' missing"

    col_vals = df[col].astype(str).str.strip().str.upper()
    mask = col_vals.eq(label.strip().upper())

    return mask, f"matched {mask.sum()} rows in column '{col}'"


def select_single_plate_well(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """
    Coerce plate/well columns to the canonical pair 'Plate_Metadata' and 'Well_Metadata'.

    This harmonises common variants (Plate, Well, Metadata_Plate, Metadata_Well, etc.)
    and zero-pads well coordinates (e.g. A1 → A01). If none are found, the frame is
    returned unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardised 'Plate_Metadata' and 'Well_Metadata' if detected.
    """
    logger = logger or logging.getLogger("plate_well_std")
    plate_cands = ["Plate_Metadata", "Metadata_Plate", "Plate", "plate", "Image_Metadata_Plate"]
    well_cands = ["Well_Metadata", "Metadata_Well", "Well", "well", "Image_Metadata_Well"]

    plate_col = next((c for c in plate_cands if c in df.columns), None)
    well_col = next((c for c in well_cands if c in df.columns), None)

    out = df.copy()
    if plate_col and "Plate_Metadata" not in out.columns:
        out = out.rename(columns={plate_col: "Plate_Metadata"})
    if well_col and "Well_Metadata" not in out.columns:
        out = out.rename(columns={well_col: "Well_Metadata"})

    # Zero-pad wells like A1 → A01
    if "Well_Metadata" in out.columns:
        out["Well_Metadata"] = out["Well_Metadata"].astype(str).str.replace(r"\s+", "", regex=True)
        def _pad(w):
            m = re.match(r"^([A-Ha-h])(\d{1,2})$", str(w))
            if not m:
                return w
            return f"{m.group(1).upper()}{int(m.group(2)):02d}"
        out["Well_Metadata"] = out["Well_Metadata"].map(_pad)

    return out


def attach_plate_well_with_fallback_single(
    *,
    obj_df: pd.DataFrame,
    input_dir: str | os.PathLike,
    obj_filename: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Attach Plate/Well using a sibling or folder-wide *Image.csv[.gz]* if missing.

    Parameters
    ----------
    obj_df : pandas.DataFrame
        Object-level CP table (must contain ImageNumber for merging).
    input_dir : str or PathLike
        Directory containing object and image CSVs.
    obj_filename : str
        Basename of the object CSV file.
    logger : logging.Logger
        Logger for status.

    Returns
    -------
    pandas.DataFrame
        DataFrame with Plate_Metadata and Well_Metadata if resolvable; else original.
    """
    df = obj_df.copy()
    if {"Plate_Metadata", "Well_Metadata"}.issubset(df.columns):
        return df

    if "ImageNumber" not in df.columns:
        logger.warning("No ImageNumber column; cannot attach Plate/Well from image table.")
        return select_single_plate_well(df, logger=logger)

    input_dir = Path(input_dir)
    base = obj_filename[:-7] if obj_filename.endswith(".csv.gz") else obj_filename[:-4] if obj_filename.endswith(".csv") else obj_filename
    candidates = [input_dir / f"{base}_Image.csv.gz", input_dir / f"{base}_Image.csv"]
    candidates = [p for p in candidates if p.exists() and not p.name.startswith((".", "_"))]

    image_path = next(iter(candidates), None)
    if image_path is None:
        # Fallback: unique folder-wide image table
        images = []
        for pat in ("*Image.csv.gz", "*_Image.csv.gz", "*Image.csv", "*_Image.csv"):
            images += list(input_dir.glob(pat))
        images = [p for p in images if not p.name.startswith((".", "_"))]
        if len(images) != 1:
            logger.warning("Could not uniquely identify an image table; skipping Plate/Well attach.")
            return select_single_plate_well(df, logger=logger)
        image_path = images[0]

    try:
        head = pd.read_csv(image_path, nrows=5, low_memory=False)
        plate_col = next((c for c in ["Plate_Metadata", "Metadata_Plate", "Plate", "Image_Metadata_Plate"] if c in head.columns), None)
        well_col = next((c for c in ["Well_Metadata", "Metadata_Well", "Well", "Image_Metadata_Well"] if c in head.columns), None)
        usecols = ["ImageNumber"] + [c for c in (plate_col, well_col) if c]
        img = pd.read_csv(image_path, usecols=usecols, low_memory=False)
    except Exception as exc:
        logger.warning("Failed to read image table %s: %s", image_path.name, exc)
        return select_single_plate_well(df, logger=logger)

    merged = df.merge(img, on="ImageNumber", how="left", suffixes=("", "_img"))
    merged = select_single_plate_well(merged, logger=logger)
    return merged



def drop_all_na_columns(
    df: pd.DataFrame,
    *,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Drop columns that are entirely NA (all missing values).

    This is handy for raw CellProfiler exports where some feature columns are
    present for schema consistency but contain no data on a given plate. The
    function logs which columns were removed and returns a copy.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    logger : logging.Logger, optional
        Logger for status messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with all-NA columns removed.
    """
    logger = logger or logging.getLogger("drop_all_na_columns")
    before = df.shape[1]
    out = df.dropna(axis=1, how="all").copy()
    removed = [c for c in df.columns if c not in out.columns]
    if removed:
        logger.info(
            "Removed %d all-NA columns: %s",
            len(removed),
            ", ".join(removed[:10]) + (" …" if len(removed) > 10 else "")
        )
    else:
        logger.info("No all-NA columns found.")
    logger.info("Columns: %d → %d after cleaning.", before, out.shape[1])
    return out



def load_single_acrosome_csv(
    path: str,
    *,
    metadata_file: str | None,
    merge_keys: str = "Plate,Well",
    logger: logging.Logger,
    drop_all_na: bool = False,
) -> pd.DataFrame:
    """
    Load a single CellProfiler object CSV (Acrosome table), optionally attach Plate/Well
    from an Image table and merge a plate map to obtain labels.

    Parameters
    ----------
    path : str
        Path to the Acrosome object CSV.
    metadata_file : str or None
        Optional plate map CSV/TSV with at least Plate/Well → cpd_id/cpd_type (and optionally Library).
    merge_keys : str
        Comma-separated Plate,Well header names to prefer in the plate map (default: 'Plate,Well').
    logger : logging.Logger
        Logger.

    Returns
    -------
    pandas.DataFrame
        Well-level table ready for feature selection and DMSO labelling.
    """
    df = read_table_auto(path, logger)
    if drop_all_na:
        df = drop_all_na_columns(df, logger=logger)
    df.columns = df.columns.map(str)

    # Attach Plate/Well if missing
    df = attach_plate_well_with_fallback_single(
        obj_df=df,
        input_dir=os.path.dirname(path) or ".",
        obj_filename=os.path.basename(path),
        logger=logger,
    )
    df = select_single_plate_well(df, logger=logger)

    # If present, aggregate to well medians (object-level → well-level)
    meta_like = {"Plate_Metadata", "Well_Metadata", "ImageNumber", "ObjectNumber"}
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in meta_like]
    if {"Plate_Metadata", "Well_Metadata"}.issubset(df.columns) and feature_cols:
        grouped = df.groupby(["Plate_Metadata", "Well_Metadata"], observed=False, sort=False)[feature_cols].median()
        out = grouped.reset_index()
    else:
        out = df.copy()

    # Merge plate map
    if metadata_file:
        meta = read_table_auto(metadata_file, logger)
        # Resolve actual plate/well column names in the metadata (case/variant tolerant)
        plate_key, well_key = find_plate_well_in_meta(meta, merge_keys, logger)

        # Standardise keys
        meta = meta.rename(columns={plate_key: "Plate_Metadata", well_key: "Well_Metadata"})

        # Zero-pad wells
        meta["Well_Metadata"] = (
            meta["Well_Metadata"].astype(str).str.replace(r"\s+", "", regex=True)
            .str.replace(r"^([A-Ha-h])(\d{1,2})$", lambda m: f"{m.group(1).upper()}{int(m.group(2)):02d}", regex=True)
        )

        out = out.merge(meta, on=["Plate_Metadata", "Well_Metadata"], how="left", validate="m:1")
        logger.info("Merged plate map %s (%d rows) on Plate/Well.", metadata_file, meta.shape[0])   
        logger.info(
        "Merged plate map %s (%d rows) on Plate/Well. Post-merge rows: %d",
        metadata_file, meta.shape[0], out.shape[0]
        )
        for col in ("cpd_id", "cpd_type"):
            miss = out[col].isna().sum() if col in out.columns else out.shape[0]
            logger.info("Post-merge: missing %s in %d rows.", col, miss)

        if {"Plate_Metadata", "Well_Metadata"}.issubset(out.columns):
            logger.info(
                "Example mappings: %s",
                out[["Plate_Metadata", "Well_Metadata", "cpd_id", "cpd_type"]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
        else:
            logger.warning("Plate/Well still missing after attempted attach + merge.")


    # Ensure expected ID columns exist for downstream
    for col in ("cpd_id", "cpd_type"):
        if col not in out.columns:
            out[col] = "missing"

    return out


def dmso_mask_fallback_scan(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Fallback: mark a row as DMSO if ANY text-like column contains the label.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label : str
        DMSO label (case-insensitive).

    Returns
    -------
    pd.Series
        Boolean mask.
    """
    upper_label = label.upper()
    text_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(text_cols) == 0:
        return pd.Series(False, index=df.index)
    # Fast partial: build a per-column mask and OR-reduce
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        mask |= df[col].astype(str).str.upper().str.contains(upper_label, na=False)
    return mask


# ---------------
# Stats utilities
# ---------------

def compare_distributions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_cols: List[str],
    logger: logging.Logger,
    test: str = "mw",
) -> pd.DataFrame:
    """
    Compare distributions of each feature between two well sets using a
    non-parametric test and EMD (Wasserstein distance).

    Parameters
    ----------
    df1 : pd.DataFrame
        Query group (e.g., a compound's wells).
    df2 : pd.DataFrame
        Comparator group (e.g., DMSO wells).
    feature_cols : list[str]
        Feature columns to test.
    logger : logging.Logger
        Logger.
    test : str
        'mw' for Mann–Whitney U (default) or 'ks' for Kolmogorov–Smirnov.

    Returns
    -------
    pd.DataFrame
        Columns: feature, stat, raw_pvalue, abs_median_diff, emd, med_query, med_comp
    """
    rows = []
    use_mw = (test.lower() == "mw")

    for feat in feature_cols:
        x1 = df1[feat].dropna()
        x2 = df2[feat].dropna()

        if len(x1) < 2 or len(x2) < 2:
            stat, p, emd = np.nan, np.nan, np.nan
        else:
            try:
                if use_mw:
                    stat, p = mannwhitneyu(x1, x2, alternative="two-sided")
                else:
                    stat, p = ks_2samp(x1, x2)
            except Exception as exc:
                logger.debug("Test failed for %s: %s", feat, exc)
                stat, p = np.nan, np.nan

            try:
                emd = wasserstein_distance(x1, x2)
            except Exception:
                emd = np.nan

        med1 = np.median(x1) if len(x1) else np.nan
        med2 = np.median(x2) if len(x2) else np.nan
        rows.append({
            "feature": feat,
            "stat": stat,
            "raw_pvalue": p,
            "abs_median_diff": np.abs(med1 - med2),
            "emd": emd,
            "med_query": med1,
            "med_comp": med2,
        })

    return pd.DataFrame(rows)


def add_fdr(
    df_stats: pd.DataFrame,
    alpha: float,
    logger: logging.Logger,
    p_col: str = "raw_pvalue",
) -> pd.DataFrame:
    """
    Add BH-FDR to a stats table safely (handles NaNs).

    Parameters
    ----------
    df_stats : pd.DataFrame
        Stats table with a raw p-value column.
    alpha : float
        FDR threshold.
    logger : logging.Logger
        Logger.
    p_col : str
        Name of raw p-value column.

    Returns
    -------
    pd.DataFrame
        With new columns: pvalue_bh, reject_bh (bool).
    """
    out = df_stats.copy()
    pvals = out[p_col].to_numpy()
    valid = ~np.isnan(pvals)

    p_bh = np.full_like(pvals, fill_value=np.nan, dtype=float)
    reject = np.zeros_like(valid, dtype=bool)

    if valid.sum() > 0:
        rej, p_corr, _, _ = multipletests(pvals[valid], method="fdr_bh", alpha=alpha)
        p_bh[valid] = p_corr
        reject[valid] = rej
    else:
        logger.warning("No valid p-values for FDR correction.")

    out["pvalue_bh"] = p_bh
    out["reject_bh"] = reject
    return out


def dmso_robust_normalise(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    dmso_mask: pd.Series,
    by_plate: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Robustly normalise features relative to DMSO using median/MAD.
    If 'by_plate' is provided, normalise within each plate independently.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing metadata and feature columns.
    feature_cols : list[str]
        Names of feature columns to normalise.
    dmso_mask : pandas.Series[bool]
        Boolean mask selecting DMSO control rows.
    by_plate : pandas.Series or None, optional
        Optional plate identifier to normalise within (e.g., df['Plate_Metadata']).

    Returns
    -------
    pandas.DataFrame
        A copy of df with the specified feature columns normalised.
    """
    X = df.copy()
    if by_plate is None:
        dmso_df = X.loc[dmso_mask, feature_cols]
        med = dmso_df.median(axis=0)
        mad = (dmso_df.subtract(med, axis=1)).abs().median(axis=0)
        mad = mad.replace(0.0, np.nan)
        X.loc[:, feature_cols] = X.loc[:, feature_cols].subtract(med, axis=1).divide(mad, axis=1)
    else:
        # groupwise normalisation
        for plate, idx in X.groupby(by_plate).groups.items():
            idx = list(idx)
            dmso_idx = [i for i in idx if dmso_mask.iloc[i]]
            if not dmso_idx:
                continue  # skip if no DMSO on this plate
            dmso_block = X.loc[dmso_idx, feature_cols]
            med = dmso_block.median(axis=0)
            mad = (dmso_block.subtract(med, axis=1)).abs().median(axis=0)
            mad = mad.replace(0.0, np.nan)
            X.loc[idx, feature_cols] = X.loc[idx, feature_cols].subtract(med, axis=1).divide(mad, axis=1)
    return X


def infer_prefix_groups(
    feature_cols: list[str],
    *,
    prefixes: list[str],
    logger: logging.Logger,
) -> dict[str, list[str]]:
    """
    Group features by the double-underscore style channel prefix (e.g. 'Acrosome__').

    Returns a dict keyed by normalised, lower-cased prefix without underscores,
    e.g. 'acrosome' -> [list of feature names].
    """
    px = [p.lower() for p in prefixes]
    groups: dict[str, list[str]] = defaultdict(list)
    for f in feature_cols:
        fl = f.lower()
        for p in px:
            if fl.startswith(p.lower()):
                key = p.strip("_").lower()
                groups[key].append(f)
                break
    logger.info("Built %d prefix groups from %d features.", len(groups), len(feature_cols))
    return dict(groups)



def filter_feature_columns_by_prefix(
    *,
    df: pd.DataFrame,
    prefixes: list[str],
    protected: set[str],
) -> pd.DataFrame:
    """
    Keep only columns that are either protected metadata or start with one of
    the given prefixes (case-insensitive). Designed for preprocessed names like
    'Acrosome__AreaShape_Area'.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with metadata + features.
    prefixes : list[str]
        Feature prefixes to keep (e.g. ["Acrosome__"]).
    protected : set[str]
        Metadata columns to keep regardless of prefix (e.g. Plate/Well/cpd_id).

    Returns
    -------
    pandas.DataFrame
        DataFrame restricted to protected + prefixed feature columns.
    """
    px_lower = [p.lower() for p in prefixes if isinstance(p, str)]
    keep_cols: list[str] = []
    for c in df.columns:
        if c in protected:
            keep_cols.append(c)
            continue
        cl = str(c).lower()
        if any(cl.startswith(p) for p in px_lower):
            keep_cols.append(c)
    return df.loc[:, keep_cols]



def group_feature_stats(
    feat_stats: pd.DataFrame,
    group_map: Dict[str, List[str]],
    fdr_alpha: float = 0.05,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Summarise per-feature stats by group, including size and FDR counts.

    Parameters
    ----------
    feat_stats : pd.DataFrame
        Per-feature stats table (should include 'pvalue_bh').
    group_map : dict[str, list[str]]
        group -> feature list mapping.
    fdr_alpha : float
        FDR threshold.
    logger : logging.Logger | None
        Logger.

    Returns
    -------
    pd.DataFrame
        Columns: group, mean_abs_median_diff, min_raw_pvalue, mean_emd,
                 min_pvalue_bh, n_features_in_group, n_features_sig_fdr
    """
    grouped = []
    for group, feats in group_map.items():
        group_df = feat_stats[feat_stats["feature"].isin(feats)]
        if group_df.empty:
            continue
        grouped.append({
            "group": group,
            "mean_abs_median_diff": group_df["abs_median_diff"].mean(),
            "min_raw_pvalue": group_df["raw_pvalue"].min(),
            "mean_emd": group_df["emd"].mean() if "emd" in group_df.columns else np.nan,
            "min_pvalue_bh": group_df["pvalue_bh"].min() if "pvalue_bh" in group_df.columns else np.nan,
            "n_features_in_group": len(group_df),
            "n_features_sig_fdr": int((group_df.get("pvalue_bh", pd.Series(dtype=float)) <= fdr_alpha).sum()),
        })

    out = pd.DataFrame(grouped)
    if logger:
        logger.debug("Summarised %d groups.", out.shape[0])
    return out


# ----
# Main
# ----

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Batch compare all compounds to DMSO for acrosome features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--ungrouped_list", help="CSV/TSV with column 'path' of well-level files.")
    mx.add_argument("--acrosome_csv", help="Single CellProfiler Acrosome object CSV (per-object).")

    parser.add_argument("--metadata_file", default=None, help="Optional plate map to add cpd_id/cpd_type/Library.")
    parser.add_argument("--merge_keys", default="Plate,Well", help="Comma-separated keys in metadata_file (default: Plate,Well).")


    parser.add_argument("--cpd_id_col", default="cpd_id", help="Compound ID column.")
    parser.add_argument("--cpd_type_col", default="cpd_type", help="Compound type column (metadata).")
    parser.add_argument("--dmso_label", default="DMSO", help="Label used for DMSO rows.")
    parser.add_argument("--dmso_col", default="cpd_type", help="Column to check for DMSO (fallback: scan all text columns).")
    parser.add_argument("--output_dir", default="acrosome_vs_DMSO", help="Output folder.")
    parser.add_argument("--top_features", type=int, default=10, help="Number of top features to report.")
    parser.add_argument("--fdr_alpha", type=float, default=0.05, help="FDR threshold.")
    parser.add_argument("--test", choices=["mw", "ks"], default="mw", help="Non-parametric test to use.")
    parser.add_argument(
        "--feature_prefix",
        action="append",
        default=None,
        help=("Restrict analysis to columns starting with this prefix (case-insensitive). "
            "Repeat for multiple prefixes, e.g. --feature_prefix Acrosome__")
    )


    parser.add_argument(
        "--normalise",
        choices=["none", "dmso_robust", "dmso_robust_per_plate"],
        default="none",
        help="Optional robust normalisation: centre by DMSO median and scale by DMSO MAD "
            "(globally or per-plate). MW/KS are rank-based, but normalising helps interpretability.",
    )

    parser.add_argument(
        "--drop_all_na",
        action="store_true",
        help="If set, drop columns that are entirely NA immediately after reading input tables."
    )


    parser.add_argument("--log_file", default="acrosome_vs_dmso.log", help="Log file name.")
    args = parser.parse_args()

    # Output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    sig_dir = os.path.join(args.output_dir, "significant")
    os.makedirs(sig_dir, exist_ok=True)

    logger = setup_logger(os.path.join(args.output_dir, args.log_file))
    logger.info("Starting batch analysis: all compounds vs. DMSO.")

    # Load & select features

    if args.acrosome_csv:
        logger.info("Reading single Acrosome CSV: %s", args.acrosome_csv)

        df = load_single_acrosome_csv(
                args.acrosome_csv,
                metadata_file=args.metadata_file,
                merge_keys=args.merge_keys,
                logger=logger,
                drop_all_na=args.drop_all_na,
            )
    else:
        df = load_ungrouped_files(
                    list_file=args.ungrouped_list,
                    logger=logger,
                    required_cols=["cpd_id", "cpd_type"],
                    drop_all_na=args.drop_all_na,
                )


    id_cols = [args.cpd_id_col, args.cpd_type_col]

    protected = {args.cpd_id_col, args.cpd_type_col, "Plate_Metadata", "Well_Metadata", "Library", "Sample"}


    # Use normalised prefixes (lower-case)
    prefixes = (args.feature_prefix or ["Acrosome__"])
    df = filter_feature_columns_by_prefix(
        df=df,
        prefixes=prefixes,
        protected=protected
    )
    logger.info("After --feature_prefix: %d columns kept.", df.shape[1])

    feature_cols = select_feature_columns(df, id_cols=[args.cpd_id_col, args.cpd_type_col], logger=logger)
    if not feature_cols:
        logger.error("No feature columns remain after filtering.")
        sys.exit(2)

    # Build prefix-based group map so we definitely have an 'acrosome' group
    group_map = infer_prefix_groups(feature_cols, prefixes=prefixes, logger=logger)
    acrosome_feats = group_map.get("acrosome", [])
    if not acrosome_feats:
        logger.warning("No acrosome-prefixed features detected. Check your --feature_prefix.")


    # Hard guard: do not continue if nothing to test
    if len(feature_cols) == 0:
        logger.error("After --feature_prefix and feature selection, no feature columns remain. "
                    "Check your prefix (e.g., use --feature_prefix Acrosome__) or remove the flag.")
        sys.exit(2)


    # DMSO mask
    dmso_mask, reason = dmso_mask_from_column(df, args.dmso_label, args.dmso_col, logger)
    if not dmso_mask.any():
        logger.warning(
            "No DMSO rows found using column=%r (%s). Falling back to scanning text columns.",
            args.dmso_col, reason,
        )
        dmso_mask = dmso_mask_fallback_scan(df, args.dmso_label)

    dmso_df = df[dmso_mask]
    logger.info("Found %d DMSO wells.", dmso_df.shape[0])
    logger.info("Proceeding with %d feature columns after filtering.", len(feature_cols))
    if not acrosome_feats:
        logger.warning("No acrosome features detected in the selected feature set. "
                    "If you expected columns like 'Acrosome__...', check --feature_prefix.")




    if args.normalise != "none":
        logger.info("Applying %s normalisation.", args.normalise)
        by_plate = df["Plate_Metadata"] if args.normalise == "dmso_robust_per_plate" else None
        df = dmso_robust_normalise(
            df=df,
            feature_cols=feature_cols,
            dmso_mask=dmso_mask,
            by_plate=by_plate,
        )
        logger.info("Applied %s normalisation to %d features.", args.normalise, len(feature_cols))
    

    # Non-DMSO compound ids
    compounds = df.loc[~dmso_mask, args.cpd_id_col].dropna().astype(str).unique().tolist()
    logger.info("Found %d non-DMSO compounds for analysis.", len(compounds))

    # Standard column order for outputs
    standard_cols = [
        "feature", "stat", "raw_pvalue", "abs_median_diff", "emd",
        "med_query", "med_comp", "pvalue_bh", "reject_bh",
        "n_query_wells", "n_dmso_wells",
    ]
    group_cols = [
        "group", "mean_abs_median_diff", "min_raw_pvalue", "mean_emd",
        "min_pvalue_bh", "n_features_in_group", "n_features_sig_fdr",
    ]

    # Per-compound analysis
    for cpd_id in compounds:
        logger.info("==== Analysing compound: %s ====", cpd_id)
        query_df = df[(df[args.cpd_id_col].astype(str) == str(cpd_id)) & (~dmso_mask)]
        if query_df.empty:
            logger.warning("No wells found for %s, skipping.", cpd_id)
            continue

        # All features
        logger.info("Comparing %s to DMSO across all features.", cpd_id)
        all_stats = compare_distributions(query_df, dmso_df, feature_cols, logger, test=args.test)
        all_stats["n_query_wells"] = query_df.shape[0]
        all_stats["n_dmso_wells"] = dmso_df.shape[0]
        all_stats = add_fdr(all_stats, alpha=args.fdr_alpha, logger=logger)
        all_stats = all_stats.sort_values(["reject_bh", "abs_median_diff"], ascending=[False, False])

        # Top features
        # if we want all features out, then uncomment
        # top_all = all_stats.head(args.top_features).copy()
        # all_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_top_features.tsv")
        # all_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_top_features.xlsx")
        # reorder_and_write(top_all, all_tsv, standard_cols, logger, "tsv")
        # reorder_and_write(top_all, all_xlsx, standard_cols, logger, "excel")

        # Significant features
        sig_all = all_stats[all_stats["reject_bh"]].copy()
        if not sig_all.empty:
            sig_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_significant_features.tsv")
            sig_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_significant_features.xlsx")
            reorder_and_write(sig_all, sig_tsv, standard_cols, logger, "tsv")
            reorder_and_write(sig_all, sig_xlsx, standard_cols, logger, "excel")

        # Acrosome-only features
        if acrosome_feats:
            logger.info("Comparing %s to DMSO across acrosome features.", cpd_id)
            acro_stats = compare_distributions(query_df, dmso_df, acrosome_feats, logger, test=args.test)
            acro_stats["n_query_wells"] = query_df.shape[0]
            acro_stats["n_dmso_wells"] = dmso_df.shape[0]
            acro_stats = add_fdr(acro_stats, alpha=args.fdr_alpha, logger=logger)
            acro_stats = acro_stats.sort_values(["reject_bh", "abs_median_diff"], ascending=[False, False])

            # top_acro = acro_stats.head(args.top_features).copy()
            # acro_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_top_features.tsv")
            # acro_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_top_features.xlsx")
            # reorder_and_write(top_acro, acro_tsv, standard_cols, logger, "tsv")
            # reorder_and_write(top_acro, acro_xlsx, standard_cols, logger, "excel")

            sig_acro = acro_stats[acro_stats["reject_bh"]].copy()
            if not sig_acro.empty:
                sig_acro_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_features.tsv")
                sig_acro_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_features.xlsx")
                reorder_and_write(sig_acro, sig_acro_tsv, standard_cols, logger, "tsv")
                reorder_and_write(sig_acro, sig_acro_xlsx, standard_cols, logger, "excel")

        # Acrosome group summary (based on *all* features’ stats)
        if "acrosome" in group_map:
            logger.info("Summarising acrosome group for %s.", cpd_id)
            acro_group = {"acrosome": group_map["acrosome"]}
            # Use the all-features stats table so FDR columns exist
            acro_group_stats = group_feature_stats(all_stats, acro_group, fdr_alpha=args.fdr_alpha, logger=logger)

            acro_grp_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_group.tsv")
            acro_grp_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_group.xlsx")
            reorder_and_write(acro_group_stats, acro_grp_tsv, group_cols, logger, "tsv")
            reorder_and_write(acro_group_stats, acro_grp_xlsx, group_cols, logger, "excel")

            # “Significant” group by min_raw_pvalue threshold
            sig_grp = acro_group_stats[acro_group_stats["min_raw_pvalue"] <= args.fdr_alpha].copy()
            if not sig_grp.empty:
                sig_grp_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_group.tsv")
                sig_grp_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_group.xlsx")
                reorder_and_write(sig_grp, sig_grp_tsv, group_cols, logger, "tsv")
                reorder_and_write(sig_grp, sig_grp_xlsx, group_cols, logger, "excel")

    logger.info("Batch comparison complete.")


if __name__ == "__main__":
    main()
