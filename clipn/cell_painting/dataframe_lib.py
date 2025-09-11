#!/usr/bin/env python3
# coding: utf-8



from __future__ import annotations

import argparse
import glob
import logging
import os  # must be before 'import torch'
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# One known-good set; pick CPU/GPU build to match your cluster
#pip install --upgrade --no-deps \
#  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
#pip install --upgrade onnx==1.14.1 onnxruntime==1.16.3 onnxscript==0.1.0
#
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Callable
import re
import numpy as np
import pandas as pd
from typing import Sequence

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import psutil
import concurrent.futures
import torch
import torch.serialization
import gzip
from clipn.model import CLIPn
import math
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn import set_config
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def extract_latent_and_meta(
    *,
    decoded_df: pd.DataFrame,
    level: str = "compound",
    aggregate: str = "median",
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare a latent matrix X and aligned metadata for diagnostics.

    Parameters
    ----------
    decoded_df : pandas.DataFrame
        Decoded table containing integer-named latent columns ('0','1',...),
        plus 'Dataset', 'Sample', 'cpd_id', and optionally 'cpd_type'/'Library'.
    level : str
        'compound' (aggregates by 'cpd_id') or 'image'.
    aggregate : str
        'median' or 'mean' for compound aggregation.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    tuple
        (X, meta, latent_cols) where X is numeric latent DataFrame, meta is
        aligned metadata, latent_cols are the latent column names used.
    """
    latent_cols = [c for c in decoded_df.columns if str(c).isdigit()]
    if not latent_cols:
        raise ValueError("No latent columns detected (expected '0','1',...).")

    if level == "image":
        meta_cols = [c for c in ["cpd_id", "cpd_type", "Dataset", "Library", "Plate_Metadata", "Well_Metadata"]
                     if c in decoded_df.columns]
        X = decoded_df.loc[:, latent_cols].copy()
        meta = decoded_df.loc[:, meta_cols].copy()
        logger.info("Diagnostics at image level: %d rows, %d dims.", X.shape[0], len(latent_cols))
        return X, meta, latent_cols

    if level == "compound":
        grouped = decoded_df.groupby(by="cpd_id", dropna=False, sort=False)
        aggfunc = "median" if aggregate == "median" else "mean"
        X = grouped[latent_cols].agg(func=aggfunc)
        def _mode_safe(s: pd.Series) -> str | None:
            s = s.dropna()
            return None if s.empty else str(s.mode(dropna=True).iloc[0])
        meta = pd.DataFrame({
            "cpd_id": X.index.astype(str),
            "cpd_type": grouped["cpd_type"].apply(func=_mode_safe) if "cpd_type" in decoded_df.columns else None,
            "Dataset": grouped["Dataset"].apply(func=_mode_safe) if "Dataset" in decoded_df.columns else None,
            "Library": grouped["Library"].apply(func=_mode_safe) if "Library" in decoded_df.columns else None,
        }).dropna(axis=1, how="all")
        meta.index = X.index
        logger.info("Diagnostics at compound level: %d compounds, %d dims.", X.shape[0], len(latent_cols))
        return X.reset_index(drop=True), meta.reset_index(drop=True), latent_cols

    raise ValueError("level must be 'compound' or 'image'.")




def detect_csv_delimiter(csv_path: str) -> str:
    """
    Detect the delimiter of a small text file (prefer tab if ambiguous).
    Transparently supports gzip (.gz) inputs.

    Parameters
    ----------
    csv_path : str
        Path to the text file.

    Returns
    -------
    str
        Detected delimiter, one of: '\\t' or ','.
    """
    opener = gzip.open if str(csv_path).endswith(".gz") else open
    with opener(csv_path, mode="rt", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)

    has_tab = "\t" in sample
    has_comma = "," in sample
    if has_tab and has_comma:
        return "\t"  # prefer TSV
    if has_tab:
        return "\t"
    if has_comma:
        return ","
    return "\t"


def mode_nonnull(series: pd.Series) -> Optional[str]:
    """
    Return the most frequent non-null value in a Series.

    Parameters
    ----------
    series : pandas.Series
        Series of values.

    Returns
    -------
    Optional[str]
        The modal value, or None if no non-null values exist.

    Notes
    -----
    - For ties, the first modal value is returned.
    - This function does not coerce dtypes; pass decoded (string) columns.
    """
    s = series.dropna()
    if s.empty:
        return None
    modes = s.mode(dropna=True)
    return None if modes.empty else str(modes.iloc[0])


def aggregate_latent_from_decoded(
    decoded_df: pd.DataFrame,
    aggregate: str = "median",
    logger: Optional["logging.Logger"] = None,
) -> pd.DataFrame:
    """
    Aggregate CLIPn latent space per compound using the *decoded* table.

    Parameters
    ----------
    decoded_df : pandas.DataFrame
        Wide decoded table containing latent dimension columns named '0','1',...
        plus 'cpd_id' and optional categorical columns (e.g. 'cpd_type', 'Library').
    aggregate : str
        Aggregation for latent dimensions. One of {'median', 'mean'}.
    logger : logging.Logger, optional
        Logger for progress messages.

    Returns
    -------
    pandas.DataFrame
        One row per 'cpd_id' with aggregated latent dimensions and decoded
        categorical columns (mode per compound).

    Notes
    -----
    - Latent columns are detected via regex r'^\\d+$' on column names.
    - Categorical columns are taken as mode; ties break by first observed.
    - Ensures 'cpd_id' is a stripped string in the output.
    """
    # Detect latent columns named as integers "0", "1", ...
    latent_cols = [c for c in decoded_df.columns if re.fullmatch(pattern=r"^\d+$", string=str(c))]
    if not latent_cols:
        raise ValueError("No latent columns detected (expected integer-named columns like '0','1',...).")

    # Columns we will try to keep as decoded categories, if present
    categorical_cols = [c for c in ["cpd_type", "Library", "Plate_Metadata", "Well_Metadata"] if c in decoded_df.columns]

    # Group by compound
    grouped = decoded_df.groupby(by="cpd_id", dropna=False, observed=True)

    # Aggregate latent dims
    if aggregate == "median":
        lat_agg = grouped[latent_cols].median(numeric_only=True)
    elif aggregate == "mean":
        lat_agg = grouped[latent_cols].mean(numeric_only=True)
    else:
        raise ValueError(f"Unsupported aggregate='{aggregate}'. Use 'median' or 'mean'.")

    # Aggregate categorical (decoded) as mode
    cat_frames = {}
    for col in categorical_cols:
        cat_frames[col] = grouped[col].apply(func=mode_nonnull)

    cat_df = pd.DataFrame(data=cat_frames) if cat_frames else pd.DataFrame(index=lat_agg.index)

    # Merge and tidy
    out = lat_agg.reset_index()
    if not cat_df.empty:
        out = out.merge(right=cat_df.reset_index(), on="cpd_id", how="left")

    out["cpd_id"] = out["cpd_id"].astype(str).str.strip()

    if logger is not None:
        logger.info("Aggregated %d compounds; kept %d latent dims; categorical cols: %s",
                    out.shape[0], len(latent_cols), categorical_cols)

    # Order columns: cpd_id, latent dims, then categorical
    ordered = ["cpd_id"] + latent_cols + [c for c in categorical_cols if c in out.columns]
    return out.loc[:, ordered]


# Technical, non-biological columns that must never be used as features
TECHNICAL_FEATURE_BLOCKLIST = {"ImageNumber","Number_Object_Number","ObjectNumber","TableNumber"}
def exclude_technical_features(cols, logger):
    """Remove any technical columns from a list of feature columns."""
    dropped = sorted(c for c in cols if c in TECHNICAL_FEATURE_BLOCKLIST)
    kept = [c for c in cols if c not in TECHNICAL_FEATURE_BLOCKLIST]
    if dropped:
        logger.info("Excluding technical feature columns: %s", ", ".join(dropped))
    return kept


def ensure_library_column(
    df: pd.DataFrame,
    filepath: str,
    logger: logging.Logger,
    value: str | None = None,
) -> pd.DataFrame:
    """
    Ensure a 'Library' column exists; use provided value or file stem.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to update.
    filepath : str
        Source path for the data (used for fallback name).
    logger : logging.Logger
        Logger for status messages.
    value : str | None
        Explicit value for the Library column; if None, uses file stem.

    Returns
    -------
    pd.DataFrame
        DataFrame with ensured 'Library' column.
    """
    if "Library" not in df.columns:
        base_library = value if value is not None else Path(filepath).stem
        df["Library"] = base_library
        logger.info("'Library' column not found. Set to: %s", base_library)
    return df


def load_single_dataset(
    name: str,
    path: str,
    logger: logging.Logger,
    metadata_cols: List[str],
) -> pd.DataFrame:
    """
    Load one dataset, standardise metadata names, and wrap with a MultiIndex.

    Parameters
    ----------
    name : str
        Dataset name used for the MultiIndex level 'Dataset'.
    path : str
        Path to the input TSV/CSV file.
    logger : logging.Logger
        Logger instance.
    metadata_cols : list[str]
        Required metadata column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with harmonised metadata and MultiIndex ('Dataset', 'Sample').

    Raises
    ------
    ValueError
        If mandatory metadata columns are missing after standardisation.
    """
    delimiter = detect_csv_delimiter(path)
    delim_name = "tab" if delimiter == "\t" else "comma"
    logger.info("[%s] Reading %s (delimiter=%s)", name, path, delim_name)
    df = read_csv_fast(path, delimiter)

    logger.debug("[%s] Columns after initial load: %s", name, df.columns.tolist())

    if df.index.name in metadata_cols:
        promoted_col = df.index.name
        df[promoted_col] = df.index
        df.index.name = None
        logger.warning("[%s] Promoted index '%s' to column to preserve metadata.", name, promoted_col)

    df = ensure_library_column(df=df, filepath=path, logger=logger, value=name)
    df = standardise_metadata_columns(df, logger=logger, dataset_name=name)

    num_cols = df.select_dtypes(include=[np.number]).shape[1]
    non_num_cols = df.shape[1] - num_cols
    logger.info("[%s] Column types: numeric=%d, non-numeric=%d",
                name, num_cols, non_num_cols)


    missing_cols = [col for col in metadata_cols if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            logger.error("[%s] Mandatory column '%s' missing after standardisation.", name, col)
        raise ValueError(f"[{name}] Mandatory column(s) {missing_cols} missing after standardisation.")

    df = df.reset_index(drop=True)
    df.index = pd.MultiIndex.from_frame(pd.DataFrame({"Dataset": name, "Sample": range(len(df))}))
    logger.debug("[%s] Final columns: %s", name, df.columns.tolist())
    logger.debug("[%s] Final shape: %s", name, df.shape)
    logger.debug("[%s] Final index names: %s", name, df.index.names)
    meta = {"cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"}
    feat_guess = [c for c in df.columns if c not in meta]
    df.loc[:, feat_guess] = df[feat_guess].apply(pd.to_numeric, errors="ignore")
    return df

def safe_to_csv(df: pd.DataFrame, path: Path | str, sep: str = "\t", logger: logging.Logger | None = None) -> None:
    """
    Write a DataFrame to CSV/TSV robustly by stringifying column names and
    flattening any MultiIndex columns before saving.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : Path | str
        Output file path.
    sep : str
        Delimiter (default: tab).
    logger : logging.Logger | None
        Logger instance.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(map(str, t)) for t in out.columns.to_list()]
    else:
        out.columns = out.columns.map(str)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep=sep, index=False)
    if logger:
        logger.info("Wrote %s rows x %s cols -> %s", out.shape[0], out.shape[1], path)



def harmonise_numeric_columns(
    dataframes: Dict[str, pd.DataFrame],
    logger: logging.Logger,
    audit_dir: Path | None = None,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Subset to the intersection of numeric columns and preserve metadata columns,
    with detailed diagnostics.
    """
    metadata_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]

    # 0) Best-effort: coerce potential numeric dtypes (avoid losing int/float stored as strings)
    for name, df in dataframes.items():
        before = df.select_dtypes(include=[np.number]).shape[1]
        cand = [c for c in df.columns if c not in metadata_cols]
        # Coerce "object" columns to numeric where possible (leave real strings untouched)
        dataframes[name].loc[:, cand] = df[cand].apply(pd.to_numeric, errors="coerce")
        after = dataframes[name].select_dtypes(include=[np.number]).shape[1]
        if after != before:
            logger.info("[%s] dtype coercion: numeric cols %d -> %d", name, before, after)

    # 1) Build numeric sets per dataset
    num_sets: Dict[str, set] = {
        name: set(df.select_dtypes(include=[np.number]).columns) for name, df in dataframes.items()
    }
    union = set().union(*num_sets.values()) if num_sets else set()
    inter = set.intersection(*num_sets.values()) if num_sets else set()

    # 2) Log headline stats before blocklist
    jacc = (len(inter) / max(1, len(union))) if union else 0.0
    logger.info("Numeric feature union=%d, intersection=%d (Jaccard=%.3f)", len(union), len(inter), jacc)

    # 3) Apply technical blocklist, but log what it removed
    inter_before_block = sorted(inter)
    common_cols = sorted(exclude_technical_features(inter_before_block, logger))
    removed_by_block = [c for c in inter_before_block if c not in common_cols]
    if removed_by_block:
        logger.info("Technical blocklist removed %d intersecting features (e.g. %s%s)",
                    len(removed_by_block),
                    ", ".join(removed_by_block[:5]),
                    "..." if len(removed_by_block) > 5 else "")

    logger.info("Harmonised numeric columns across datasets (after blocklist): %d", len(common_cols))

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        pd.Series(sorted(union), name="feature").to_csv(
            audit_dir / "feature_union.tsv", sep="\t", index=False
        )
        pd.Series(sorted(inter), name="feature").to_csv(
            audit_dir / "feature_intersection_pre_blocklist.tsv", sep="\t", index=False
        )
        pd.Series(common_cols, name="feature").to_csv(
            audit_dir / "feature_intersection_post_blocklist.tsv", sep="\t", index=False
        )

    # 4) Per-dataset diagnostics
    for name, cols in num_sets.items():
        # What this dataset is missing vs the global intersection (pre-blocklist) and union
        missing_from_inter = sorted(list(inter - cols))[:20]
        missing_from_union = sorted(list(union - cols))[:20]
        logger.debug("[%s] Missing from intersection (first 20): %s",
                     name, ", ".join(missing_from_inter) if missing_from_inter else "<none>")
        logger.debug("[%s] Missing from union (first 20): %s",
                     name, ", ".join(missing_from_union) if missing_from_union else "<none>")

    # 5) Assemble harmonised frames
    for name, df in dataframes.items():
        numeric_df = df[common_cols] if common_cols else df.select_dtypes(include=[np.number])
        metadata_df = df[metadata_cols]
        df_harmonised = pd.concat([numeric_df, metadata_df], axis=1)
        assert df_harmonised.index.equals(df.index), f"Index mismatch after harmonisation in '{name}'."
        dataframes[name] = df_harmonised
        logger.debug("[%s] Harmonisation successful, final columns: %s",
                     name, df_harmonised.columns.tolist())

    return dataframes, common_cols




def load_and_harmonise_datasets(
    datasets_csv: str,
    logger: logging.Logger,
    mode: str | None = None,
    audit_dir: Path | None = None,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load all datasets listed in the 'datasets_csv' file and harmonise.

    Parameters
    ----------
    datasets_csv : str
        Path to a TSV/CSV mapping of dataset name to path. Must have columns:
        'dataset' and 'path'.
    logger : logging.Logger
        Logger instance.
    mode : str | None
        Included for API stability; not used.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], list[str]]
        Mapping dataset name -> harmonised DataFrame, and common numeric columns.
    """
    delimiter = detect_csv_delimiter(datasets_csv)
    datasets_df = pd.read_csv(filepath_or_buffer=datasets_csv, delimiter=delimiter)
    if not {"dataset", "path"}.issubset(set(datasets_df.columns)):
        raise ValueError("datasets_csv must contain 'dataset' and 'path' columns.")

    dataset_paths = datasets_df.set_index("dataset")["path"].to_dict()
    metadata_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]

    dataframes: Dict[str, pd.DataFrame] = {}

    logger.info("Loading datasets individually (%d listed in %s)",
            len(dataset_paths), datasets_csv)
    for name, path in dataset_paths.items():
        try:
            logger.info(" -> [%s] %s", name, path)
            dataframes[name] = load_single_dataset(name=name, path=path, logger=logger, metadata_cols=metadata_cols)
        except ValueError as exc:
            logger.error("Loading dataset '%s' failed: %s", name, exc)
            raise

    return harmonise_numeric_columns(dataframes=dataframes, logger=logger, audit_dir=audit_dir)



def read_csv_fast(path: str, delimiter: str) -> pd.DataFrame:
    # Try pyarrow engine (fast); fall back to pandas' python engine.
    try:
        return pd.read_csv(path, delimiter=delimiter, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, delimiter=delimiter, engine="python", compression="infer")
