#!/usr/bin/env python3
# coding: utf-8



from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import set_config
import csv
set_config(transform_output="pandas")

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

logger = logging.getLogger(__name__)

# Global timer (for memory log timestamps)
_SCRIPT_START_TIME = time.time()



def log_memory_usage(
    logger: logging.Logger,
    prefix: str = "",
    extra_msg: str | None = None,
) -> None:
    """
    Log the current and peak memory usage (resident set size).

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    prefix : str
        Optional prefix for the log message.
    extra_msg : str | None
        Optional additional message.
    """
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)

    peak_gb = None
    try:
        # ru_maxrss is kilobytes on Linux
        import resource  # noqa: PLC0415

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Linux":
            peak_gb = peak_rss / (1024 ** 2)
        else:
            peak_gb = peak_rss / (1024 ** 3)
    except Exception:
        pass

    elapsed = time.time() - _SCRIPT_START_TIME
    msg = f"{prefix} Memory usage: {mem_gb:.2f} GB (resident set size)"
    if peak_gb is not None:
        msg += f", Peak: {peak_gb:.2f} GB"
    msg += f", Elapsed: {elapsed/60:.1f} min"
    if extra_msg:
        msg += " | " + extra_msg
    logger.info(msg)


def read_table_auto(path: str) -> pd.DataFrame:
    """Read CSV/TSV with automatic delimiter detection (prefers tab)."""
    sep = detect_csv_delimiter(path)
    return pd.read_csv(filepath_or_buffer=path, sep=sep)



def merge_annotations(
    latent_df_or_path: str | pd.DataFrame,
    annotation_file: str,
    output_prefix: str,
    logger: logging.Logger,
) -> None:
    """
    Merge compound annotations into the CLIPn latent output on plate/well.

    Parameters
    ----------
    latent_df_or_path : str | pd.DataFrame
        Path to latent TSV or a DataFrame of latent outputs.
    annotation_file : str
        Path to an annotation TSV file with plate/well mappings.
    output_prefix : str
        Base path prefix for output files (no extension).
    logger : logging.Logger
        Logger instance.
    """
    try:
        if isinstance(latent_df_or_path, str):
            latent_df = pd.read_csv(filepath_or_buffer=latent_df_or_path, sep="\t")
        else:
            latent_df = latent_df_or_path.copy()

        # instead of fixed sep="\t"
        annot_df = read_table_auto(annotation_file)


        # Try to derive Plate/Well if only generic columns provided
        if "Plate_Metadata" not in annot_df.columns and "Plate" in annot_df.columns:
            annot_df["Plate_Metadata"] = annot_df["Plate"]
        if "Well_Metadata" not in annot_df.columns and "Well" in annot_df.columns:
            annot_df["Well_Metadata"] = annot_df["Well"]

        logger.info("Merging annotations on keys: Plate_Metadata, Well_Metadata")
        logger.info("Latent columns: %s", latent_df.columns.tolist())
        logger.info("Annotation columns: %s", annot_df.columns.tolist())
        logger.info("Latent shape: %s, Annotation shape: %s", latent_df.shape, annot_df.shape)

        if "Plate_Metadata" not in latent_df.columns or "Well_Metadata" not in latent_df.columns:
            logger.warning("Plate_Metadata or Well_Metadata missing in latent data — merge skipped.")
            return

        merged = pd.merge(
            left=latent_df,
            right=annot_df,
            on=["Plate_Metadata", "Well_Metadata"],
            how="left",
            validate="many_to_one",
        )

        logger.info("Merged shape: %s", merged.shape)
        if "cpd_id" in merged.columns:
            n_merged = merged["cpd_id"].notna().sum()
            logger.info("Successfully merged rows with non-null cpd_id: %s", n_merged)

        merged_tsv = f"{output_prefix}_latent_with_annotations.tsv"
        merged.to_csv(path_or_buf=merged_tsv, sep="\t", index=False)
        logger.info("Merged annotation saved to: %s", merged_tsv)

    except Exception as exc:
        logger.warning("Annotation merging failed: %s", exc)



def clean_and_impute_features_knn(
    df: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
    *,
    groupby_cols: list[str] = None,           # e.g. ["Dataset", "Plate_Metadata"]
    max_nan_col_frac: float = 0.30,           # drop features with >30% NaN
    max_nan_row_frac: float = 0.80,           # drop rows with >80% NaN across kept features
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """
    KNN imputation (optionally per-group), with robust in-place scaling
    (median/IQR) before KNN to avoid scale dominance, and unscaling after.

    Returns
    -------
    (df_imputed, dropped_features)
    """
    if not feature_cols:
        return df, []

    df = df.copy()

    # Replace inf with NaN
    df.loc[:, feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop very sparse features
    col_nan_frac = df[feature_cols].isna().mean(axis=0)
    drop_feats = col_nan_frac[col_nan_frac > max_nan_col_frac].index.tolist()
    keep_feats = [c for c in feature_cols if c not in drop_feats]
    if drop_feats:
        logger.warning("KNN: dropping %d/%d features with > %.0f%% NaN (e.g. %s)",
                       len(drop_feats), len(feature_cols), max_nan_col_frac*100, drop_feats[:10])

    # Drop extremely incomplete rows
    if keep_feats:
        row_nan_frac = df[keep_feats].isna().mean(axis=1)
        drop_rows = row_nan_frac > max_nan_row_frac
        n_drop_rows = int(drop_rows.sum())
        if n_drop_rows:
            logger.warning("KNN: dropping %d rows with > %.0f%% NaN across kept features.",
                           n_drop_rows, max_nan_row_frac*100)
        df = df.loc[~drop_rows]
    else:
        logger.error("KNN: all features would be dropped. Loosen thresholds.")
        return df.iloc[0:0], feature_cols

    if not keep_feats:
        return df, drop_feats

    # Grouped KNN impute
    groupby_cols = groupby_cols or ["Dataset"]
    imputer = None  # created per group (k may change if group very small)

    def _robust_scale_matrix(mat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        med = mat.median(axis=0, skipna=True)
        iqr = mat.quantile(0.75) - mat.quantile(0.25)
        iqr = iqr.replace(0, 1.0)  # avoid divide by zero
        scaled = (mat - med) / iqr
        return scaled, med, iqr

    def _unscale_matrix(scaled: np.ndarray, med: pd.Series, iqr: pd.Series) -> pd.DataFrame:
        return (scaled * iqr.values) + med.values

    def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
        X = g[keep_feats]

        # If group too small for KNN, fallback to median
        if len(X) < 2:
            med = X.median(numeric_only=True)
            g.loc[:, keep_feats] = X.fillna(med)
            return g

        # Robust scale → KNN → unscale
        X_scaled, med, iqr = _robust_scale_matrix(X)
        k_eff = max(1, min(n_neighbors, len(X)))
        imp = KNNImputer(n_neighbors=k_eff, weights="uniform")
        X_imp_scaled = imp.fit_transform(X_scaled.values)
        X_imp = _unscale_matrix(X_imp_scaled, med, iqr)

        g.loc[:, keep_feats] = X_imp
        return g

    logger.info("Imputing with KNN (k=%d) per group=%s", n_neighbors, groupby_cols)
    df = df.groupby(groupby_cols, dropna=False, sort=False).apply(_impute_group)

    # pandas >=2.1 can leave group keys in the index; restore original
    if isinstance(df.index, pd.MultiIndex) and df.index.names != ["Dataset", "Sample"]:
        try:
            df.index = df.index.droplevel(list(range(len(groupby_cols))))
        except Exception:
            pass

    # Final check (rare): any NaNs left → fill with global medians to be safe
    if df[keep_feats].isna().any().any():
        logger.warning("KNN: residual NaNs after impute; filling with global medians.")
        df.loc[:, keep_feats] = df[keep_feats].fillna(df[keep_feats].median(numeric_only=True))

    return df, drop_feats


def clean_and_impute_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
    *,
    groupby_cols: list[str] = None,           # e.g. ["Dataset", "Plate_Metadata"]
    max_nan_col_frac: float = 0.3,            # drop features with >30% NaN
    max_nan_row_frac: float = 0.8,            # drop rows with >80% NaN across features
) -> tuple[pd.DataFrame, list[str]]:
    """
    Replace ±inf with NaN, drop very sparse features/rows, and impute remaining NaNs.

    Returns
    -------
    (df_clean, dropped_features)
    """
    if not feature_cols:
        return df, []

    df = df.copy()
    # 1) replace inf
    df.loc[:, feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 2) drop sparse features
    col_nan_frac = df[feature_cols].isna().mean(axis=0)
    drop_feats = col_nan_frac[col_nan_frac > max_nan_col_frac].index.tolist()
    if drop_feats:
        logger.warning(
            "Dropping %d/%d features with > %.0f%% NaN: first few %s",
            len(drop_feats), len(feature_cols), max_nan_col_frac * 100, drop_feats[:10],
        )
        keep_feats = [c for c in feature_cols if c not in drop_feats]
    else:
        keep_feats = feature_cols

    # 3) drop extremely incomplete rows (across kept features)
    if keep_feats:
        row_nan_frac = df[keep_feats].isna().mean(axis=1)
        drop_rows = row_nan_frac > max_nan_row_frac
        n_drop_rows = int(drop_rows.sum())
        if n_drop_rows:
            logger.warning(
                "Dropping %d rows with > %.0f%% NaN across kept features.",
                n_drop_rows, max_nan_row_frac * 100,
            )
        df = df.loc[~drop_rows]
    else:
        logger.error("All features would be dropped. Loosen max_nan_col_frac or inspect inputs.")
        return df.iloc[0:0], feature_cols  # empty df

    # 4) impute remaining NaNs (median per group)
    groupby_cols = groupby_cols or ["Dataset"]
    missing_before = int(df[keep_feats].isna().sum().sum())
    if missing_before:
        logger.info(
            "Imputing %d remaining NaNs using median per group=%s.",
            missing_before, groupby_cols,
        )
        def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
            med = g[keep_feats].median(numeric_only=True)
            g.loc[:, keep_feats] = g[keep_feats].fillna(med)
            return g
        df = df.groupby(groupby_cols, dropna=False, sort=False).apply(_impute_group)
        # pandas >= 2.1 leaves group keys in index name sometimes; ensure same index
        if isinstance(df.index, pd.MultiIndex) and df.index.names != ["Dataset", "Sample"]:
            try:
                df.index = df.index.droplevel(list(range(len(groupby_cols))))
            except Exception:
                pass

    still_missing = int(df[keep_feats].isna().sum().sum())
    if still_missing:
        logger.warning("After median impute, %d NaNs remain; filling with global medians.", still_missing)
        global_med = df[keep_feats].median(numeric_only=True)
        df.loc[:, keep_feats] = df[keep_feats].fillna(global_med)

    return df, drop_feats


def clean_nonfinite_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
    label: str = "",
) -> pd.DataFrame:
    """
    Replace ±inf with NaN in selected feature columns and log counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_cols : list[str]
        Numeric feature column names to clean.
    logger : logging.Logger
        Logger for status messages.
    label : str
        Short label to include in log messages.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with non-finite values replaced by NaN.
    """
    out = df.copy()
    n_pos = int(np.isinf(out[feature_cols]).sum().sum())
    n_neg = int(np.isneginf(out[feature_cols]).sum().sum())
    if n_pos or n_neg:
        logger.warning("[%s] Replacing %d inf and %d -inf with NaN.", label, n_pos, n_neg)
        out.loc[:, feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    return out


def aggregate_latent_per_compound(
    df: pd.DataFrame,
    group_col: str = "cpd_id",
    latent_cols: List[str] | None = None,
    method: str = "median",
) -> pd.DataFrame:
    """
    Aggregate image-level latent vectors to a single row per compound.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing image-level latent and metadata.
    group_col : str
        Name of the compound identifier column.
    latent_cols : list[str] | None
        Names of latent columns. If None, uses integer-named columns.
    method : str
        Aggregation method: 'median', 'mean', 'min', or 'max'.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame (one row per compound).
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame.")

    if latent_cols is None:
        latent_cols = [
            col for col in df.columns
            if isinstance(col, int) or (isinstance(col, str) and col.isdigit())
        ]
        if not latent_cols:
            raise ValueError("No integer-named latent columns found.")

    latent_cols = sorted(latent_cols, key=int)
    aggfunc = method if method in {"mean", "median", "min", "max"} else "median"
    aggregated = df.groupby(group_col, as_index=False)[latent_cols].agg(aggfunc)
    return aggregated
