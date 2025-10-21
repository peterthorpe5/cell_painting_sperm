#!/usr/bin/env python
# coding: utf-8

"""
library of data processing modules. 
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import re
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_selection import VarianceThreshold
import subprocess
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import umap.umap_ as umap
from pathlib import Path
from clipn import CLIPn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import gc
from contextlib import nullcontext
# so we keep the index .. fingers crossed!
from sklearn import set_config
set_config(transform_output="pandas")

from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import optuna
logger = logging.getLogger(__name__)




# Reduce allocator fragmentation before torch initialises (CLIPn will import torch)
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Technical, non-biological columns that must never be used as features
TECHNICAL_FEATURE_BLOCKLIST = {"ImageNumber","Number_Object_Number","ObjectNumber","TableNumber"}

# Columns that are metadata (never model inputs), case-insensitive match by name
METADATA_COL_BLOCKLIST = {
    "cpd_id", "cpd_type", "dataset", "sample", "replicate", "library",
    "plate_metadata", "well_metadata", "plate", "well",
    "chemid", "compound_id", "concentration", "dose", "timepoint", "batch", "site"
}



##################################################################
# functions

def load_datasets_from_folderlist(list_file_path, exclude_file_substring=None):
    """
    Load datasets from a list file specifying dataset names and folder paths.
    Each line in the list file should be:
        dataset_name    /path/to/folder

    The function will read all `.csv` files in each folder (except ones containing `exclude_file_substring`).
    Each CSV is assumed to have a MultiIndex: ['cpd_id', 'Library', 'cpd_type'].

    Parameters
    ----------
    list_file_path : str
        Path to the list file with dataset names and folders.

    exclude_file_substring : str, optional
        If provided, any CSV file containing this substring in the filename will be skipped.

    Returns
    -------
    dict
        Dictionary mapping dataset names to concatenated DataFrames.
    """
    datasets = {}

    with open(list_file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line in list file: '{line.strip()}'")

            dataset_name, folder_path = parts
            folder_path = folder_path.strip()

            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            csv_files = [f for f in os.listdir(folder_path)
                         if f.endswith(".csv") and
                         (exclude_file_substring not in f if exclude_file_substring else True)]

            dataframes = []
            for csv_file in csv_files:
                csv_path = os.path.join(folder_path, csv_file)
                df = pd.read_csv(csv_path, index_col=[0, 1, 2])
                df.index.names = ["cpd_id", "Library", "cpd_type"]
                dataframes.append(df)

            if not dataframes:
                raise ValueError(f"No CSV files found for dataset '{dataset_name}' in '{folder_path}'")

            combined_df = pd.concat(dataframes)
            datasets[dataset_name] = combined_df

    return datasets



def objective(trial, X, y):
    """
    Objective function for Optuna hyperparameter tuning of CLIPn.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.
    X : dict
        Dictionary of dataset inputs (e.g., {0: stb_data, 1: experiment_data}).
    y : dict
        Dictionary of dataset labels matching the structure of `X`.

    Returns
    -------
    float
        Validation loss or score to be minimised.
    """
    latent_dim = trial.suggest_int("latent_dim", 10, 60)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 200, 500)

    logger.info(f"Trying CLIPn with latent_dim={latent_dim}, lr={lr:.6f}, epochs={epochs}")
    for dataset_id in X:
        logger.info(f"Dataset {dataset_id}: X shape = {X[dataset_id].shape}, y shape = {y[dataset_id].shape}")


    clipn_model = CLIPn(X, y, latent_dim=latent_dim)
    loss = clipn_model.fit(X, y, lr=lr, epochs=epochs)
    # Return final loss if it's a list or array, otherwise assume scalar
    if isinstance(loss, (list, tuple, np.ndarray)):
        return loss[-1]
    else:
        return loss


def optimise_clipn(X, y, n_trials=40):
    """
    Runs Optuna Bayesian optimisation to tune CLIPn hyperparameters.

    Parameters
    ----------
    X : dict
        Dictionary of dataset inputs.

    y : dict
        Dictionary of dataset labels.

    n_trials : int, optional
        Number of optimisation trials (default: 20).

    Returns
    -------
    dict
        Best hyperparameter set found.
    """
    logger.info("Starting Bayesian optimisation with %d trials.", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    logger.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params



def variance_threshold_selector(
    *,
    data: pd.DataFrame,
    threshold: float = 0.05,
    pdf_path: Optional[str] = None,
    log_pdf_path: Optional[str] = None,
    title: Optional[str] = None,
    log_x_linear_pdf: bool = False,
    bins: Optional[int] = None,
    bin_scale: float = 3.0,
    max_bins: Optional[int] = 240,
    log_sorted_x: bool = False
) -> pd.DataFrame:
    """
    Select features by variance threshold and optionally emit diagnostics PDFs.

    Behaviour
    ---------
    - Returns columns whose population variance (ddof=0) is strictly greater
      than ``threshold`` (matches scikit-learn's VarianceThreshold).
    - If ``pdf_path`` is provided, writes a two-page *linear-scale* PDF:
        1) Histogram of variances (optionally log10 x if ``log_x_linear_pdf``).
        2) Sorted variances (linear axes) with the threshold line.
    - If ``log_pdf_path`` is provided, writes a three-page *log-focused* PDF:
        1) Histogram of log10(variance) with **linear y**.
        2) Histogram of log10(variance) with **log y**.
        3) Sorted variances with **log y** (and optional log x for rank if
           ``log_sorted_x=True``).
      Pages 1-2 let you compare with/without log y (as requested).

    Parameters
    ----------
    data : pandas.DataFrame
        Input feature matrix. Non-numeric columns are ignored for variance
        calculation and filtering.
    threshold : float, optional
        Keep features with variance strictly greater than this value. Default 0.05.
    pdf_path : str or None, optional
        Output path for the linear-scale diagnostics PDF.
    log_pdf_path : str or None, optional
        Output path for the log-focused diagnostics PDF (see Behaviour).
    title : str or None, optional
        Optional title used on plots; a default is used if None.
    log_x_linear_pdf : bool, optional
        If True, the histogram in the *linear* PDF uses log10(variance) on the
        x-axis (y remains linear). Default False.
    bins : int or None, optional
        If set, use this exact bin count for histograms. If None, choose
        automatically and scale by ``bin_scale`` with an upper cap ``max_bins``.
    bin_scale : float, optional
        Multiplier for the automatic bin choice. Default 3.0.
    max_bins : int or None, optional
        Maximum number of bins when auto-choosing. Default 240.
    log_sorted_x : bool, optional
        Apply log10 scale to the rank axis on the *log-focused* sorted plot.
        Default False.

    Returns
    -------
    pandas.DataFrame
        Columns with variance greater than ``threshold`` (order preserved).

    Notes
    -----
    - Variances for plotting use ddof=0 (population variance) to match the
      selector. Selection itself uses VarianceThreshold.
    """
    logger = logging.getLogger(__name__)

    # Keep only numeric columns; cast to float32 for efficiency
    X = data.select_dtypes(include=[np.number]).astype(np.float32, copy=False)
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found for variance selection.")

    # Selection (matches scikit-learn behaviour)
    selector = VarianceThreshold(threshold=threshold)
    mask = selector.fit(X).get_support()
    kept_cols = X.columns[mask]
    result = X.loc[:, kept_cols]

    # Pre-filter variances for diagnostics
    variances = X.var(axis=0, ddof=0).astype(np.float64)
    vals = variances.values
    n_total = variances.size
    n_kept = int(mask.sum())
    n_dropped = int(n_total - n_kept)
    pct_kept = 100.0 * n_kept / max(n_total, 1)
    pct_drop = 100.0 * n_dropped / max(n_total, 1)
    n_nonpos = int((vals <= 0).sum())

    def _auto_bins(n_features: int) -> int:
        base = min(80, max(12, int(np.sqrt(n_features) * 2)))
        scaled = max(1, int(round(base * bin_scale)))
        return min(scaled, max_bins) if max_bins is not None else scaled

    # --------------------------
    # Linear-scale diagnostics
    # --------------------------
    if pdf_path is not None:
        os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
        fig_title = title or "Feature variance distribution"
        bin_count = int(bins) if bins is not None else _auto_bins(n_total)

        with PdfPages(pdf_path) as pdf:
            # Page 1: histogram (linear y; optionally log-x)
            fig, ax = plt.subplots(figsize=(8, 5))
            if log_x_linear_pdf:
                safe = vals[vals > 0]
                if safe.size == 0:
                    ax.text(0.5, 0.5, "All variances ≤ 0; cannot plot log10 histogram.",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_xlabel("log10(variance)")
                    ax.set_ylabel("Number of features")
                else:
                    ax.hist(np.log10(safe), bins=bin_count, edgecolor="black", alpha=0.75)
                    ax.set_xlabel("log10(variance)")
                    if threshold > 0:
                        ax.axvline(np.log10(threshold), linestyle="--", linewidth=1.5)
                        ax.text(np.log10(threshold), ax.get_ylim()[1] * 0.95,
                                f"threshold = {threshold:g}", rotation=90, va="top", ha="right")
                if n_nonpos:
                    ax.annotate(f"Excluded non-positive variances: {n_nonpos}",
                                xy=(0.02, 0.95), xycoords="axes fraction", ha="left", va="top")
            else:
                ax.hist(vals, bins=bin_count, edgecolor="black", alpha=0.75)
                ax.set_xlabel("Variance")
                ax.axvline(threshold, linestyle="--", linewidth=1.5)
                ax.text(threshold, ax.get_ylim()[1] * 0.95,
                        f"threshold = {threshold:g}", rotation=90, va="top", ha="right")

            ax.set_ylabel("Number of features")
            ax.set_title(fig_title)
            ax.annotate(
                f"Features: {n_total:,}\nKept (> {threshold:g}): {n_kept:,} ({pct_kept:.1f}%)\n"
                f"Dropped (≤ {threshold:g}): {n_dropped:,} ({pct_drop:.1f}%)\nBins: {bin_count}",
                xy=(0.98, 0.98), xycoords="axes fraction", xytext=(-5, -5),
                textcoords="offset points", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9)
            )
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: sorted variances (linear y)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sorted_vals = np.sort(vals)
            ax2.plot(sorted_vals, linewidth=1.25)
            ax2.axhline(threshold, linestyle="--", linewidth=1.5)
            ax2.set_xlabel("Feature rank (sorted by variance)")
            ax2.set_ylabel("Variance")
            ax2.set_title(f"{fig_title} (sorted)")
            ax2.text(x=len(sorted_vals) * 0.99, y=threshold,
                     s=f"threshold = {threshold:g}", ha="right", va="bottom")
            pdf.savefig(fig2)
            plt.close(fig2)

        logger.info(
            "Variance diagnostics (linear) → %s (kept %d / %d, %.1f%%; bins=%d).",
            pdf_path, n_kept, n_total, pct_kept, bin_count
        )

    # --------------------------
    # Log-focused diagnostics
    # --------------------------
    if log_pdf_path is not None:
        os.makedirs(os.path.dirname(log_pdf_path) or ".", exist_ok=True)
        fig_title = (title or "Feature variance distribution") + " [log diagnostics]"
        bin_count = int(bins) if bins is not None else _auto_bins(n_total)
        safe = vals[vals > 0]

        with PdfPages(log_pdf_path) as pdf:
            # Page 1: histogram of log10(variance), linear y
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            if safe.size == 0:
                ax1.text(0.5, 0.5, "All variances ≤ 0; cannot plot log10 histogram.",
                         ha="center", va="center", transform=ax1.transAxes)
            else:
                ax1.hist(np.log10(safe), bins=bin_count, edgecolor="black", alpha=0.75)
                if threshold > 0:
                    thr_log = np.log10(threshold)
                    ax1.axvline(thr_log, linestyle="--", linewidth=1.5)
                    ax1.text(thr_log, ax1.get_ylim()[1] * 0.95,
                             f"threshold = {threshold:g}", rotation=90, va="top", ha="right")
            ax1.set_xlabel("log10(variance)")
            ax1.set_ylabel("Number of features")
            ax1.set_title(fig_title)
            if n_nonpos:
                ax1.annotate(f"Excluded non-positive variances: {n_nonpos}",
                             xy=(0.02, 0.95), xycoords="axes fraction", ha="left", va="top")
            pdf.savefig(fig1)
            plt.close(fig1)

            # Page 2: histogram of log10(variance), **log y**
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            if safe.size == 0:
                ax2.text(0.5, 0.5, "All variances ≤ 0; cannot plot log10 histogram.",
                         ha="center", va="center", transform=ax2.transAxes)
            else:
                ax2.hist(np.log10(safe), bins=bin_count, edgecolor="black", alpha=0.75)
                ax2.set_yscale("log")
                if threshold > 0:
                    thr_log = np.log10(threshold)
                    ax2.axvline(thr_log, linestyle="--", linewidth=1.5)
                    ax2.text(thr_log, ax2.get_ylim()[1],
                             f"threshold = {threshold:g}", rotation=90, va="top", ha="right")
            ax2.set_xlabel("log10(variance)")
            ax2.set_ylabel("Number of features (log scale)")
            ax2.set_title(fig_title)
            if n_nonpos:
                ax2.annotate(f"Excluded non-positive variances: {n_nonpos}",
                             xy=(0.02, 0.95), xycoords="axes fraction", ha="left", va="top")
            pdf.savefig(fig2)
            plt.close(fig2)

            # Page 3: sorted variances with **log y** (and optional log x)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sorted_vals = np.sort(vals)
            ax3.plot(sorted_vals, linewidth=1.25)
            ax3.set_yscale("log")
            if log_sorted_x:
                ax3.set_xscale("log")
                ax3.set_xlabel("Feature rank (log scale)")
            else:
                ax3.set_xlabel("Feature rank (sorted by variance)")
            ax3.set_ylabel("Variance (log scale)")
            if threshold > 0:
                ax3.axhline(threshold, linestyle="--", linewidth=1.5)
                ax3.text(x=ax3.get_xlim()[1], y=threshold,
                         s=f"threshold = {threshold:g}", ha="right", va="bottom")
            ax3.set_title(f"{fig_title} (sorted)")
            pdf.savefig(fig3)
            plt.close(fig3)

        logger.info(
            "Variance diagnostics (log) → %s (kept %d / %d, %.1f%%; bins=%d; log_sorted_x=%s).",
            log_pdf_path, n_kept, n_total, pct_kept, bin_count, str(log_sorted_x)
        )

    return result



def correlation_filter(data: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """
    Remove highly correlated features from a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input feature matrix (numeric).
    threshold : float, optional
        Correlation threshold above which a feature will be removed (default: 0.99).

    Returns
    -------
    pandas.DataFrame
        DataFrame with highly correlated features removed.
    """
    # Defensive: Convert to float32
    data = data.astype(np.float32, copy=False)

    # Compute absolute correlation matrix (float32 if possible)
    corr_matrix = data.corr().abs().astype(np.float32)
    # Only look at upper triangle, skip self-correlations
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    # Get columns to drop (first in pair is always kept)
    drop_cols = set()
    columns = data.columns
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if corr_matrix.iat[i, j] > threshold:
                drop_cols.add(columns[j])
    # Drop all identified columns
    return data.drop(columns=list(drop_cols))


def load_annotation(annotation_path):
    """Load annotation file safely."""
    try:
        annotation_df = pd.read_csv(annotation_path)
        annotation_df.columns = annotation_df.columns.str.strip().str.replace(" ", "_")
        annotation_df.set_index(['Plate_Metadata', 'Well_Metadata'], inplace=True)
        logger.info(f"Annotation file loaded with shape {annotation_df.shape}")
        return annotation_df
    except Exception as e:
        logger.warning(f"Annotation file could not be loaded: {e}")
        return None


def standardise_annotation_columns(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns in the annotation DataFrame to match expected schema.

    Parameters
    ----------
    annotation_df : pd.DataFrame
        The raw annotation DataFrame.

    Returns
    -------
    pd.DataFrame
        Standardised annotation DataFrame with expected columns.
    """
    rename_map = {
        "COMPOUND_NAME": "cpd_id",
        "Library": "Library",
        "Source_Plate_Barcode": "Plate_Metadata",
        "Source_Well": "Well_Metadata"
    }

    annotation_df = annotation_df.rename(columns={
        k: v for k, v in rename_map.items() if k in annotation_df.columns
    })

    # Ensure required columns exist
    required = ["cpd_id", "Library", "Plate_Metadata", "Well_Metadata"]
    missing = [col for col in required if col not in annotation_df.columns]
    if missing:
        logger.warning(f"Annotation file is missing expected columns: {missing}")

    return annotation_df


# This function is a pain too!! 
def standardise_metadata_columns(df, logger=None, dataset_name=None):
    """
    Standardise column names for metadata and recover 'cpd_id' from index if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to standardise.
    logger : logging.Logger, optional
        Logger instance for status messages.
    dataset_name : str, optional
        Dataset name used to infer 'cpd_type' if missing.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised metadata columns.

    Raises
    ------
    ValueError
        If mandatory metadata columns are missing and cannot be recovered.
    """
    rename_map = {
        "library": "Library",
        "Library": "Library",
        "compound_name": "cpd_id",
        "COMPOUND_NAME": "cpd_id",
        "Source_Plate_Barcode": "Plate_Metadata",
        "Source_Well": "Well_Metadata"
    }

    # Rename columns using mapping
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
            if logger:
                logger.info(f"Renamed column '{old}' to '{new}'")

    # Attempt to recover 'cpd_id' from index if not in columns
    if "cpd_id" not in df.columns:
        if df.index.name == "cpd_id":
            df["cpd_id"] = df.index
            if logger:
                logger.warning(f"[{dataset_name}] 'cpd_id' recovered from Index.")
        else:
            if logger:
                logger.error(f"[{dataset_name}] 'cpd_id' not found in columns or index.")
            raise ValueError(f"[{dataset_name}] 'cpd_id' not found in columns or index.")

    # Handle missing 'cpd_type' explicitly (but allow fallback)
    if "cpd_type" not in df.columns:
        fallback_type = dataset_name if dataset_name else "unknown"
        df["cpd_type"] = fallback_type
        if logger:
            logger.warning(f"[{dataset_name}] 'cpd_type' missing, inferred as '{fallback_type}'.")

    return df


def group_and_filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups data by cpd_id and Library, averages numeric features,
    and preserves a representative Plate_Metadata and Well_Metadata
    if available.

    Parameters
    ----------
    df : pd.DataFrame
        The imputed dataset to process. Must have MultiIndex with
        levels ['cpd_id', 'Library', 'cpd_type'].

    Returns
    -------
    pd.DataFrame
        The grouped and cleaned DataFrame.
    """
    if df is None or df.empty:
        return df

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex DataFrame with ['cpd_id', 'Library', 'cpd_type'].")

    # Drop noisy metadata columns, retain key ones
    noisy_metadata_pattern = (
        r"COMPOUND_NUMBER|Notes|Seahorse_alert|Treatment|Number|"
        r"Child|Paren|Location_[XYZ]|ZernikePhase|Euler|Plate$|Well$|Field|Center_[XYZ]|"
        r"no_|fn_|Source_Well|Source_Plate_Well|Source_Well"
    )
    filter_cols = df.columns[df.columns.str.contains(noisy_metadata_pattern, case=False, regex=True)]
    df = df.drop(columns=filter_cols, errors="ignore")

    # Drop index level columns that also exist in .columns to avoid reset_index clashes
    overlapping = [col for col in df.index.names if col in df.columns]
    if overlapping:
        df = df.drop(columns=overlapping)

    # Preserve Plate_Metadata and Well_Metadata
    available_meta = [col for col in ["Plate_Metadata", "Well_Metadata"] if col in df.columns]
    if available_meta:
        meta_cols = ["cpd_id", "Library"] + available_meta
        meta_df = df.reset_index()[meta_cols].drop_duplicates()
        meta_df = meta_df.groupby(["cpd_id", "Library"], as_index=False).first()
        meta_df.set_index(["cpd_id", "Library"], inplace=True)
    else:
        meta_df = None

    # Group numeric features by compound
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    grouped = df_numeric.groupby(["cpd_id", "Library"], as_index=True).mean()

    # Join metadata back, if available
    if meta_df is not None:
        grouped = grouped.join(meta_df, how="left")

    # Reset index so all metadata are columns again
    grouped.reset_index(inplace=True)

    return grouped



def decode_clipn_predictions(predicted_labels, predicted_cpd_ids,
                             cpd_type_encoder, cpd_id_encoder):
    """
    Decode numeric predictions back to original 'cpd_type' and 'cpd_id' labels.

    Parameters
    ----------
    predicted_labels : np.ndarray or list
        Predicted numeric `cpd_type` values (from CLIPn or similar models).
    predicted_cpd_ids : np.ndarray or list
        Predicted numeric `cpd_id` values (optional, if used for clustering).

    cpd_type_encoder : LabelEncoder
        Fitted LabelEncoder used to encode 'cpd_type'.

    cpd_id_encoder : LabelEncoder
        Fitted LabelEncoder used to encode 'cpd_id'.

    Returns
    -------
    tuple
        - original_labels : np.ndarray
            Decoded `cpd_type` labels.
        - original_cpd_ids : np.ndarray
            Decoded `cpd_id` labels.
    """
    original_labels = cpd_type_encoder.inverse_transform(predicted_labels)
    original_cpd_ids = cpd_id_encoder.inverse_transform(predicted_cpd_ids)
    return original_labels, original_cpd_ids


def align_features_and_labels(X, y):
    """
    Ensures that features and labels for each dataset ID are aligned in length.

    Parameters
    ----------
    X : dict
        Feature arrays for each dataset index.
    y : dict
        Label arrays for each dataset index.

    Returns
    -------
    tuple
        Aligned versions of X and y.
    """
    X_aligned, y_aligned = {}, {}

    for k in X:
        x_len = X[k].shape[0]
        y_len = len(y[k])
        if x_len != y_len:
            logger.warning(f"Dataset {k}: Length mismatch (X: {x_len}, y: {y_len}). Truncating to min length.")
            min_len = min(x_len, y_len)
            X_aligned[k] = X[k][:min_len]
            y_aligned[k] = y[k][:min_len]
        else:
            X_aligned[k] = X[k]
            y_aligned[k] = y[k]

    return X_aligned, y_aligned


def ensure_multiindex(df, required_levels=("cpd_id", "Library", "cpd_type"), logger=None, dataset_name="dataset"):
    """
    Ensures the given DataFrame has a MultiIndex on required levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be checked or modified.

    required_levels : tuple of str, optional
        Columns to use as MultiIndex (default: ('cpd_id', 'Library', 'cpd_type')).

    logger : logging.Logger or None, optional
        Logger instance for info/warning messages. If None, prints will be used.

    dataset_name : str, optional
        Name of the dataset (used for logging).

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex applied if needed.
    """
    if df is None or df.empty:
        if logger:
            logger.warning(f"{dataset_name}: DataFrame is empty or None, skipping MultiIndex restoration.")
        return df

    if isinstance(df.index, pd.MultiIndex) and all(level in df.index.names for level in required_levels):
        if logger:
            logger.info(f"{dataset_name}: MultiIndex already present.")
        return df  # Already has correct MultiIndex

    # Check that all required levels exist as columns
    missing_cols = set(required_levels) - set(df.columns)
    if missing_cols:
        msg = f"{dataset_name}: Cannot restore MultiIndex, missing columns: {missing_cols}"
        if logger:
            logger.error(msg)
        else:
            print("ERROR:", msg)
        return df  # Return unchanged to prevent crash

    # Set MultiIndex
    df = df.set_index(list(required_levels))
    if logger:
        logger.info(f"{dataset_name}: MultiIndex restored using columns {required_levels}")
    return df



def compute_pairwise_distances(latent_df):
    """
    Compute the pairwise Euclidean distance between compounds in latent space.

    Parameters:
    -----------
    latent_df : pd.DataFrame
        DataFrame containing latent representations indexed by `cpd_id`.

    Returns:
    --------
    pd.DataFrame
        Distance matrix with `cpd_id` as row and column labels.
    """
    dist_matrix = cdist(latent_df.values, latent_df.values, metric="euclidean")
    dist_df = pd.DataFrame(dist_matrix, index=latent_df.index, columns=latent_df.index)
    return dist_df


def generate_similarity_summary(dist_df):
    """
    Generate a summary of the closest and farthest compounds.

    Parameters:
    -----------
    dist_df : pd.DataFrame
        Pairwise distance matrix with `cpd_id` as row and column labels.

    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with closest and farthest compounds.
    """
    closest_compounds = dist_df.replace(0, np.nan).idxmin(axis=1)  # Ignore self-comparison
    farthest_compounds = dist_df.idxmax(axis=1)

    summary_df = pd.DataFrame({
        "Compound": dist_df.index,  # Preserve cpd_id
        "Closest Compound": closest_compounds,
        "Distance to Closest": dist_df.min(axis=1),
        "Farthest Compound": farthest_compounds,
        "Distance to Farthest": dist_df.max(axis=1)
    })

    return summary_df


def restore_encoded_labels(encoded_series, encoder):
    """
    Restores original categorical labels from an encoded Series using the fitted LabelEncoder.

    Parameters
    ----------
    encoded_series : pd.Series or np.ndarray
        Encoded numeric labels to be restored.

    encoder : sklearn.preprocessing.LabelEncoder
        The fitted LabelEncoder used during encoding.

    Returns
    -------
    np.ndarray
        Array of original categorical labels.
    """
    if encoder is None:
        raise ValueError("Encoder must be provided and fitted.")
    return encoder.inverse_transform(encoded_series)



def reconstruct_combined_latent_df(Z, dataset_index_map, index_lookup):
    """
    Reconstructs a combined DataFrame of latent representations using original MultiIndex.

    Parameters
    ----------
    Z : dict
        Dictionary containing CLIPn latent representations, with integer keys (dataset labels).
    
    dataset_index_map : dict
        Dictionary mapping integer dataset indices (from CLIPn) to dataset names (e.g., {0: "experiment", 1: "stb"}).

    index_lookup : dict
        Dictionary mapping dataset names to their original MultiIndex (e.g., {"experiment": index_df, "stb": index_df}).

    Returns
    -------
    pd.DataFrame
        Combined latent representation DataFrame with correct MultiIndex restored.
    """
    latent_frames = []

    for index_id, dataset_name in dataset_index_map.items():
        if dataset_name not in index_lookup:
            raise ValueError(f"Missing index for dataset '{dataset_name}'")

        latent_array = Z[index_id]
        dataset_index = index_lookup[dataset_name]

        if latent_array.shape[0] != len(dataset_index):
            raise ValueError(
                f"Mismatch: latent array for '{dataset_name}' has shape {latent_array.shape[0]}, "
                f"but index has length {len(dataset_index)}"
            )

        df = pd.DataFrame(latent_array, index=dataset_index)
        df["dataset"] = dataset_name  # Optional: helpful for visualisation
        latent_frames.append(df)

    combined_latent_df = pd.concat(latent_frames)
    return combined_latent_df



def impute_missing_values(experiment_data, stb_data, impute_method="median", knn_neighbors=5):
    """
    Perform missing value imputation while preserving MultiIndex (cpd_id, Library, cpd_type).

    Parameters
    ----------
    experiment_data : pd.DataFrame or None
        Experiment dataset with MultiIndex (`cpd_id`, `Library`, `cpd_type`).

    stb_data : pd.DataFrame or None
        STB dataset with MultiIndex (`cpd_id`, `Library`, `cpd_type`).

    impute_method : str, optional
        Imputation method: "median" (default) or "knn".

    knn_neighbors : int, optional
        Number of neighbours for KNN imputation (default: 5).

    Returns
    -------
    tuple
        - experiment_data_imputed : pd.DataFrame or None
        - stb_data_imputed : pd.DataFrame or None
        - stb_labels : np.array
        - stb_cpd_id_map : dict
    """
    logger.info(f"Performing imputation using {impute_method} strategy.")

    # Choose and configure imputer
    if impute_method == "median":
        imputer = SimpleImputer(strategy="median")
    elif impute_method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    else:
        raise ValueError("Invalid imputation method. Choose 'median' or 'knn'.")

    # Enable pandas output to preserve index and column names
    imputer.set_output(transform="pandas")

    # Helper to apply imputation to a DataFrame
    def impute_dataframe(df):
        if df is None or df.empty:
            return df
        numeric_df = df.select_dtypes(include=[np.number])
        return imputer.fit_transform(numeric_df)

    # Apply imputation
    experiment_data_imputed = impute_dataframe(experiment_data)
    stb_data_imputed = impute_dataframe(stb_data)

    logger.info(f"Imputation complete. Experiment shape: {experiment_data_imputed.shape if experiment_data_imputed is not None else 'None'}, "
                f"STB shape: {stb_data_imputed.shape if stb_data_imputed is not None else 'None'}")

    # Encode STB labels if available
    if stb_data is not None and "cpd_type" in stb_data.index.names:
        try:
            label_encoder = LabelEncoder()
            stb_labels = label_encoder.fit_transform(stb_data.index.get_level_values("cpd_type"))
            stb_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            stb_cpd_id_map = dict(zip(stb_data.index.get_level_values("cpd_id"), stb_labels))
        except Exception as e:
            logger.warning(f"Failed to encode STB labels: {e}")
            stb_labels = np.zeros(stb_data_imputed.shape[0]) if stb_data_imputed is not None else np.array([])
            stb_cpd_id_map = {}
    else:
        stb_labels = np.zeros(stb_data_imputed.shape[0]) if stb_data_imputed is not None else np.array([])
        stb_cpd_id_map = {}
        logger.warning("Warning: No STB labels available!")
    # Restore original non-numeric columns (e.g., cpd_id, Library, cpd_type)
    if experiment_data is not None:
        for col in ["cpd_id", "Library", "cpd_type"]:
            if col in experiment_data.index.names:
                experiment_data_imputed[col] = experiment_data.index.get_level_values(col)
            elif col in experiment_data.columns:
                experiment_data_imputed[col] = experiment_data[col]

    if stb_data is not None:
        for col in ["cpd_id", "Library", "cpd_type"]:
            if col in stb_data.index.names:
                stb_data_imputed[col] = stb_data.index.get_level_values(col)
            elif col in stb_data.columns:
                stb_data_imputed[col] = stb_data[col]
    # Log presence of restored columns
    logger.debug(f"experiment_data_imputed columns: {experiment_data_imputed.columns.tolist()}")
    logger.debug(f"stb_data_imputed columns: {stb_data_imputed.columns.tolist()}")
    return experiment_data_imputed, stb_data_imputed, stb_labels, stb_cpd_id_map


def process_common_columns(df1, df2, step="before"):
    """
    Identify and apply common numerical columns between two datasets.

    Parameters:
    -----------
    df1 : pd.DataFrame or None
        First dataset (e.g., experiment data).
    df2 : pd.DataFrame or None
        Second dataset (e.g., STB data).
    step : str, optional
        Whether this is executed "before" or "after" imputation (default: "before").

    Returns:
    --------
    df1_filtered : pd.DataFrame or None
        First dataset filtered to keep only common columns.
    df2_filtered : pd.DataFrame or None
        Second dataset filtered to keep only common columns.
    common_columns : Index
        The set of common columns between the two datasets.
    """
    if df1 is not None and df2 is not None:
        common_columns = df1.columns.intersection(df2.columns)
    elif df1 is not None:
        common_columns = df1.columns
    elif df2 is not None:
        common_columns = df2.columns
    else:
        raise ValueError(f"Error: No valid numerical data available at step '{step}'!")

    logger.info(f"Common numerical columns {step} imputation: {list(common_columns)}")

    # Filter datasets to keep only common columns
    df1_filtered = df1[common_columns] if df1 is not None else None
    df2_filtered = df2[common_columns] if df2 is not None else None

    return df1_filtered, df2_filtered, common_columns


def encode_cpd_data(dataframes, encode_labels=False):
    """
    Applies MultiIndex and optionally encodes 'cpd_id' and 'cpd_type' for ML.

    Parameters
    ----------
    dataframes : dict
        Dictionary of dataset name to DataFrame.
    encode_labels : bool, optional
        If True, returns encoded labels and mappings. Default is False.

    Returns
    -------
    dict
        Dictionary with structure:
        {
            "dataset_name": {
                "data": DataFrame,
                "cpd_type_encoded": np.ndarray,
                "cpd_type_mapping": dict,
                "cpd_id_mapping": dict
            }
        }
        If encode_labels is False, only returns {"dataset_name": DataFrame}.
    """
    from sklearn.preprocessing import LabelEncoder

    results = {}

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue

        # Ensure proper index
        if {"cpd_id", "Library", "cpd_type"}.issubset(df.columns):
            df = df.set_index(["cpd_id", "Library", "cpd_type"])
        elif not isinstance(df.index, pd.MultiIndex):
            raise ValueError(f"{name} is missing MultiIndex or required columns.")

        output = {"data": df}

        if encode_labels:
            # Encode cpd_type
            le_type = LabelEncoder()
            cpd_type_encoded = le_type.fit_transform(df.index.get_level_values("cpd_type"))
            cpd_type_mapping = dict(zip(le_type.classes_, le_type.transform(le_type.classes_)))

            # Encode cpd_id
            le_id = LabelEncoder()
            cpd_id_encoded = le_id.fit_transform(df.index.get_level_values("cpd_id"))
            cpd_id_mapping = dict(zip(le_id.classes_, le_id.transform(le_id.classes_)))
            # Add to DataFrame (with restored MultiIndex)
            df = df.copy()
            df["cpd_type_encoded"] = cpd_type_encoded
            df["cpd_id_encoded"] = cpd_id_encoded
            output.update({
                "data": df,
                "cpd_type_encoded": cpd_type_encoded,
                "cpd_type_mapping": cpd_type_mapping,
                "cpd_type_encoder": le_type,
                "cpd_id_mapping": cpd_id_mapping,
                "cpd_id_encoder": le_id
            })
        results[name] = output
    return results



from contextlib import nullcontext
import numpy as np
import gc

def _to_numpy_safe(x):
    """
    Convert torch tensor or array-like to a NumPy array without assuming torch is present.
    """
    try:
        import torch  # noqa: F401
        if hasattr(x, "detach") and hasattr(x, "device"):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _predict_chunked_indexed(
    *,
    model,
    indexed_data: dict[int, np.ndarray],
    chunk_rows: int,
    logger: logging.Logger
) -> dict[int, np.ndarray]:
    """
    Predict latents for an indexed data dict in bounded-sized row chunks.

    Behaviour
    ---------
    - If ``chunk_rows <= 0``, runs a single-shot predict and normalises outputs to NumPy.
    - Otherwise, iterates each dataset id and calls ``model.predict`` on successive row
      slices, concatenating outputs in order.
    """
    # Single-shot path
    if chunk_rows is None or int(chunk_rows) <= 0:
        lat = model.predict(indexed_data)
        return {k: _to_numpy_safe(v) for k, v in lat.items()}

    out: dict[int, list[np.ndarray]] = {k: [] for k in indexed_data}

    # Use inference_mode if torch is present; otherwise no-op
    try:
        import torch  # noqa: F401
        inference_ctx = torch.inference_mode()
    except Exception:
        inference_ctx = nullcontext()

    for k, X in indexed_data.items():
        n_rows = int(X.shape[0])
        logger.info("Chunked predict: dataset_id=%d, rows=%d, chunk_rows=%d", k, n_rows, int(chunk_rows))
        if n_rows == 0:
            continue

        start = 0
        with inference_ctx:
            while start < n_rows:
                end = min(start + int(chunk_rows), n_rows)
                X_slice = X[start:end, :]

                lat_k_dict = model.predict({k: X_slice})
                lat_k_np = _to_numpy_safe(lat_k_dict[k])
                out[k].append(lat_k_np)

                # Tidy up between slices
                del X_slice, lat_k_dict, lat_k_np
                gc.collect()
                try:
                    import torch  # noqa: F401
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                start = end

    # Concatenate parts per dataset id
    z: dict[int, np.ndarray] = {}
    for k, parts in out.items():
        z[k] = np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32)
    return z



def _iter_row_chunks(total_rows: int, *, chunk_rows: int) -> list[tuple[int, int]]:
    """
    Yield (start, end) index pairs that partition [0, total_rows) into row chunks.

    Parameters
    ----------
    total_rows : int
        Number of rows to cover.
    chunk_rows : int
        Maximum number of rows per chunk.

    Returns
    -------
    list[tuple[int, int]]
        Consecutive [start, end) half-open intervals.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_rows:
        end = min(start + chunk_rows, total_rows)
        ranges.append((start, end))
        start = end
    return ranges


def _predict_chunked_indexed(
    *,
    model,
    indexed_data: dict[int, np.ndarray],
    chunk_rows: int,
    logger: logging.Logger
) -> dict[int, np.ndarray]:
    """
    Predict latents for an indexed data dict in bounded-sized row chunks.

    Parameters
    ----------
    model : Any
        Trained CLIPn model exposing .predict(dict[int -> np.ndarray]).
    indexed_data : dict[int, np.ndarray]
        Mapping dataset_id -> feature matrix (N_i x D).
    chunk_rows : int
        Max rows per forward pass. If <= 0, runs single-shot.
    logger : logging.Logger
        Logger for status.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping dataset_id -> concatenated latent array (N_i x L).
    """
    import gc
    import torch

    if chunk_rows is None or int(chunk_rows) <= 0:
        # Original single-shot behaviour
        return model.predict(indexed_data)

    out: dict[int, list[np.ndarray]] = {k: [] for k in indexed_data}

    for k, X in indexed_data.items():
        n_rows = int(X.shape[0])
        logger.info("Chunked predict: dataset_id=%d, rows=%d, chunk_rows=%d", k, n_rows, int(chunk_rows))
        if n_rows == 0:
            out[k] = []
            continue

        for start, end in _iter_row_chunks(n_rows, chunk_rows=int(chunk_rows)):
            X_slice = X[start:end, :]
            # Use inference_mode to minimise overhead
            with torch.inference_mode():
                lat_dict = model.predict({k: X_slice})
            out[k].append(lat_dict[k].detach().cpu().numpy())

            # Proactively release memory between chunks
            del X_slice, lat_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate slices
    z: dict[int, np.ndarray] = {}
    for k, parts in out.items():
        if not parts:
            z[k] = np.empty((0, getattr(model, "latent_dim", 0)), dtype=np.float32)
        else:
            z[k] = np.concatenate(parts, axis=0)
    return z


def prepare_data_for_clipn_from_df(
    df: pd.DataFrame,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           Dict[str, Optional[Dict[int, str]]], Dict[str, List[str]],
           Dict[int, str]]:
    """
    Build CLIPn-ready dictionaries from a MultiIndex DataFrame.

    Assumptions
    -----------
    - Index is a MultiIndex with levels ['Dataset', 'Sample'].
    - 'cpd_type' has been globally label-encoded to integer already.
    - Input features X must exclude all metadata (case-insensitive match
      against METADATA_COL_BLOCKLIST and TECHNICAL_FEATURE_BLOCKLIST),
      and must exclude 'cpd_type' even if it is integer-encoded.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table indexed by ('Dataset', 'Sample'), containing numeric
        features and metadata columns (e.g., 'cpd_id', 'cpd_type').

    Returns
    -------
    tuple
        data_dict : dict[int, np.ndarray]
            Integer dataset key -> feature matrix (X), numeric only and
            metadata-free.
        label_dict : dict[int, np.ndarray]
            Integer dataset key -> encoded labels (y) taken directly from
            the (already-encoded) 'cpd_type' column. If absent, an empty
            array of length N is returned for that dataset.
        label_mappings : dict[str, Optional[dict[int, str]]]
            Per-dataset label mapping placeholder (None here; keep the
            global LabelEncoder mapping elsewhere).
        cpd_ids : dict[str, list[str]]
            Dataset name -> list of 'cpd_id' values aligned to rows (empty
            strings if column absent).
        dataset_key_mapping : dict[int, str]
            Integer-to-dataset-name mapping for CLIPn.

    Raises
    ------
    ValueError
        If the index is not a MultiIndex with names ['Dataset', 'Sample'].
        If 'cpd_type' exists but is not integer-encoded.
    """
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Dataset", "Sample"]:
        raise ValueError("Expected a MultiIndex with levels ['Dataset', 'Sample'].")

    # Use the global blocklists defined elsewhere in this module.
    # They must be available in the module scope:
    #   TECHNICAL_FEATURE_BLOCKLIST = {...}
    #   METADATA_COL_BLOCKLIST = {...}
    meta_lower = {m.lower() for m in METADATA_COL_BLOCKLIST | TECHNICAL_FEATURE_BLOCKLIST}

    def is_meta(col: str) -> bool:
        return col.lower() in meta_lower

    data_dict_by_name: Dict[str, np.ndarray] = {}
    label_dict_by_name: Dict[str, np.ndarray] = {}
    label_mappings: Dict[str, Optional[Dict[int, str]]] = {}
    cpd_ids: Dict[str, List[str]] = {}

    dataset_names = df.index.get_level_values("Dataset").unique().tolist()

    for ds_name in dataset_names:
        ds_df = df.loc[ds_name]

        # Labels (y): use globally encoded integers if present
        if "cpd_type" in ds_df.columns:
            if not pd.api.types.is_integer_dtype(ds_df["cpd_type"]):
                raise ValueError(
                    "Column 'cpd_type' must be globally label-encoded to integer "
                    "before calling prepare_data_for_clipn_from_df()."
                )
            # Fill any NA with -1 to keep dtype stable (model can ignore -1s if needed)
            y = ds_df["cpd_type"].fillna(-1).astype(int).to_numpy()
        else:
            y = np.empty((len(ds_df),), dtype=int)

        # Features (X): numeric only, drop all metadata and 'cpd_type'
        numeric_cols = ds_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if (not is_meta(c)) and c != "cpd_type"]
        X = ds_df.loc[:, feature_cols].to_numpy()

        # IDs aligned to rows
        if "cpd_id" in ds_df.columns:
            ids = ds_df["cpd_id"].astype(str).tolist()
        else:
            ids = ["" for _ in range(len(ds_df))]

        data_dict_by_name[ds_name] = X
        label_dict_by_name[ds_name] = y
        label_mappings[ds_name] = None  # keep global mapping externally
        cpd_ids[ds_name] = ids

    # Reindex dataset keys to integers for CLIPn
    dataset_key_mapping: Dict[int, str] = {i: name for i, name in enumerate(data_dict_by_name.keys())}
    data_dict: Dict[int, np.ndarray] = {i: data_dict_by_name[name] for i, name in dataset_key_mapping.items()}
    label_dict: Dict[int, np.ndarray] = {i: label_dict_by_name[name] for i, name in dataset_key_mapping.items()}

    return data_dict, label_dict, label_mappings, cpd_ids, dataset_key_mapping



def prepare_data_for_clipn(experiment_data_imputed, experiment_labels, experiment_label_mapping,
                           stb_data_imputed, stb_labels, stb_label_mapping):
    """
    Prepare data for CLIPn clustering by encoding datasets, removing non-numeric columns,
    and structuring inputs for training.

    Parameters
    ----------
    experiment_data_imputed : pd.DataFrame or None
        The imputed experimental dataset.
    experiment_labels : np.array
        Encoded labels for experiment compounds.
    experiment_label_mapping : dict
        Mapping of encoded labels to original experiment cpd_type.
    stb_data_imputed : pd.DataFrame or None
        The imputed STB dataset.
    stb_labels : np.array
        Encoded labels for STB compounds.
    stb_label_mapping : dict
        Mapping of encoded labels to original STB cpd_type.

    Returns
    -------
    tuple
        X (dict): Dictionary containing dataset arrays for CLIPn.
        y (dict): Dictionary of corresponding labels.
        label_mappings (dict): Mapping of dataset indices to original labels.
    """
    X, y, label_mappings = {}, {}, {}

    # Ensure at least one dataset exists
    dataset_names = []
    if experiment_data_imputed is not None and not experiment_data_imputed.empty:
        dataset_names.append("experiment_assay_combined")
    if stb_data_imputed is not None and not stb_data_imputed.empty:
        dataset_names.append("STB_combined")

    if not dataset_names:
        logger.error("No valid datasets available for CLIPn analysis.")
        raise ValueError("Error: No valid datasets available for CLIPn analysis.")

    # Encode dataset names
    dataset_encoder = LabelEncoder()
    dataset_indices = dataset_encoder.fit_transform(dataset_names)
    dataset_mapping = dict(zip(dataset_indices, dataset_names))

    logger.info(f"Dataset Mapping: {dataset_mapping}")

    # Define non-numeric columns to drop before passing to CLIPn
    non_numeric_cols = ["cpd_id", "Library", "cpd_type"]

    # Process experiment data
    if experiment_data_imputed is not None and not experiment_data_imputed.empty:
        experiment_data_imputed = experiment_data_imputed.drop(columns=[col for col in non_numeric_cols if col in experiment_data_imputed], errors="ignore")
        
        exp_index = dataset_encoder.transform(["experiment_assay_combined"])[0]
        X[exp_index] = experiment_data_imputed.values
        y[exp_index] = experiment_labels
        label_mappings[exp_index] = experiment_label_mapping

        logger.info(f"  Added Experiment Data to X with shape: {experiment_data_imputed.shape}")
    else:
        logger.warning(" No valid experiment data for CLIPn.")

    # Process STB data
    if stb_data_imputed is not None and not stb_data_imputed.empty:
        stb_data_imputed = stb_data_imputed.drop(columns=[col for col in non_numeric_cols if col in stb_data_imputed], errors="ignore")

        stb_index = dataset_encoder.transform(["STB_combined"])[0]
        X[stb_index] = stb_data_imputed.values
        y[stb_index] = stb_labels
        label_mappings[stb_index] = stb_label_mapping

        logger.info(f"  Added STB Data to X with shape: {stb_data_imputed.shape}")
    else:
        logger.warning(" No valid STB data for CLIPn.")

    # Debugging: Log dataset keys before passing to CLIPn
    logger.info(f" X dataset keys before CLIPn: {list(X.keys())}")
    logger.info(f" y dataset keys before CLIPn: {list(y.keys())}")

    # Ensure at least one dataset is available
    if not X:
        logger.error(" No valid datasets available for CLIPn analysis. Aborting!")
        raise ValueError("Error: No valid datasets available for CLIPn analysis.")

    logger.info(" Datasets successfully structured for CLIPn.")
    logger.info(f" Final dataset shapes being passed to CLIPn: { {k: v.shape for k, v in X.items()} }")

    return X, y, label_mappings, dataset_mapping


def run_clipn_simple(data_dict, label_dict, latent_dim=20, lr=1e-5, epochs=300):
    """
    Runs CLIPn training given input features and labels.

    Parameters
    ----------
    data_dict : dict
        Mapping from dataset name -> np.ndarray of features.
    label_dict : dict
        Mapping from dataset name -> np.ndarray of label ids.
    latent_dim : int
        Dimensionality of the latent space.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.

    Returns
    -------
    tuple
        latent_named_dict : dict
            Dictionary mapping dataset name -> latent representations.
        model : CLIPn
            Trained CLIPn model.
        loss : float
            Final training loss.
    """
    # Use a single ordered key list for consistent indexing
    keys = list(data_dict.keys())

    indexed_data_dict = {i: data_dict[k] for i, k in enumerate(keys)}
    indexed_label_dict = {i: label_dict[k] for i, k in enumerate(keys)}
    reverse_mapping = {i: k for i, k in enumerate(keys)}

    model = CLIPn(indexed_data_dict, indexed_label_dict, latent_dim=latent_dim)
    loss = model.fit(indexed_data_dict, indexed_label_dict, lr=lr, epochs=epochs)

    # --- Memory hygiene before inference (safe eval) ---
    import gc
    import torch

    def _safe_eval(m) -> None:
        """
        Put the model into eval mode if supported, trying both the wrapper
        and its inner torch module (commonly on .model). Silently no-ops if
        neither exposes .eval().
        """
        try:
            if hasattr(m, "eval") and callable(getattr(m, "eval")):
                m.eval()
            inner = getattr(m, "model", None)
            if inner is not None and hasattr(inner, "eval") and callable(getattr(inner, "eval")):
                inner.eval()
        except Exception as e:
            # Keep quiet in normal runs; switch to logger.debug if you prefer
            pass

    _safe_eval(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Decide chunk size in Python (no shell needed), default 100k
    chunk_env = os.environ.get("CLIPN_PREDICT_CHUNK_ROWS", "").strip()
    try:
        chunk_rows = int(chunk_env) if chunk_env else 100_000
    except ValueError:
        chunk_rows = 100_000

    # Chunked, GPU-friendly prediction
    latent_dict = _predict_chunked_indexed(
        model=model,
        indexed_data=indexed_data_dict,
        chunk_rows=chunk_rows,
        logger=logger,
    )

    latent_named_dict = {reverse_mapping[i]: latent_dict[i] for i in latent_dict}
    return latent_named_dict, model, loss




def project_query_to_latent(model, query_df):
    """
    Projects query samples into the trained CLIPn latent space.

    Parameters
    ----------
    model : CLIPn
        Trained CLIPn model.
    query_df : pd.DataFrame
        MultiIndex DataFrame with numeric features and metadata.

    Returns
    -------
    dict
        Dictionary mapping dataset name to latent embedding array (np.ndarray).
    """
    from collections import defaultdict

    projected = {}
    dataset_keys = query_df.index.get_level_values("Dataset").unique()

    for dataset in dataset_keys:
        dataset_df = query_df.loc[dataset]
        meta_cols = ['cpd_id', 'cpd_type', 'Library']
        feature_cols = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in meta_cols]
        X_query = dataset_df[feature_cols].to_numpy()

        projected[dataset] = model.predict({dataset: X_query})[dataset]

    return projected


def run_clipn(X, y, output_folder, args):
    """
    Runs CLIPn clustering with optional hyperparameter optimization.

    Parameters
    ----------
    X : dict
        Dictionary containing dataset arrays for CLIPn.
    y : dict
        Dictionary of corresponding labels.
    output_folder : str
        Directory to save CLIPn output files.
    args : argparse.Namespace
        Command-line arguments, including latent_dim, learning rate, and epoch count.

    Returns
    -------
    dict
        Dictionary containing latent representations from CLIPn.
    """
    hyperparam_file = os.path.join(output_folder, "best_hyperparameters.json")

    # Check if optimized hyperparameters should be used
    if args.use_optimized_params:
        try:
            logger.info(f"Loading optimized hyperparameters from {args.use_optimized_params}")
            with open(args.use_optimized_params, "r") as f:
                best_params = json.load(f)
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load optimized parameters: {e}")
            raise ValueError("Invalid or missing hyperparameter JSON file.")

        # Update args with loaded parameters
        args.latent_dim = best_params["latent_dim"]
        args.lr = best_params["lr"]
        args.epoch = best_params["epochs"]

        logger.info(f"Using pre-trained parameters: latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")

        # Initialize model and directly run prediction
        clipn_model = CLIPn(X, y, latent_dim=args.latent_dim)
        logger.info("Skipping training. Generating latent representations using pre-trained parameters.")
        Z = clipn_model.predict(X)

    else:
        # Run Hyperparameter Optimization
        logger.info("Running Hyperparameter Optimization")
        best_params = optimise_clipn(X,y, n_trials=40)  # Bayesian Optimization

        # Save optimized parameters
        with open(hyperparam_file, "w") as f:
            json.dump(best_params, f, indent=4)

        logger.info(f"Optimized hyperparameters saved to {hyperparam_file}")

        # Update args with best parameters
        args.latent_dim = best_params["latent_dim"]
        args.lr = best_params["lr"]
        args.epoch = best_params["epochs"]

        logger.info(f"Using optimized parameters: latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")

        # Train the model with the optimized parameters
        logger.info(f"Running CLIPn with optimized latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")
        clipn_model = CLIPn(X, y, latent_dim=args.latent_dim)
        logger.info("Fitting CLIPn model...")
        loss = clipn_model.fit(X, y, lr=args.lr, epochs=args.epoch)
        logger.info(f"CLIPn training completed. Final loss: {loss[-1]:.6f}")

        # Generate latent representations
        logger.info("Generating latent representations.")
        Z = clipn_model.predict(X)

    return Z


# Ensure index backup is not empty and restore MultiIndex properly
def restore_multiindex(imputed_df, index_backup, dataset_name="Unknown"):
    """
    Restore MultiIndex after imputation using backed-up index dataframe.

    Parameters
    ----------
    imputed_df : pd.DataFrame
        The DataFrame after imputation, which needs its index restored.
    index_backup : pd.DataFrame
        The backed-up index (before reset), containing original metadata columns.
    dataset_name : str, optional
        Name of the dataset, used for logging.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex restored if possible.
    """
    import logging
    logger = logging.getLogger("imputation_logger")

    # Define expected index columns
    index_cols = ["cpd_id", "Library", "cpd_type", "Plate_Metadata", "Well_Metadata"]

    # If mismatch in row counts, do not attempt to restore
    if len(imputed_df) != len(index_backup):
        logger.error(
            f"{dataset_name}: Cannot restore MultiIndex — row count mismatch "
            f"(df: {len(imputed_df)}, index_backup: {len(index_backup)}). Skipping index restore."
        )
        return imputed_df

    # Restore any missing index columns from backup
    for col in index_cols:
        if col not in imputed_df.columns and col in index_backup.columns:
            imputed_df[col] = index_backup[col]
            logger.warning(f"{dataset_name}: Reattached missing index column '{col}' from backup.")

    # Ensure all index columns exist before setting MultiIndex
    missing_cols = [col for col in index_cols if col not in imputed_df.columns]
    if missing_cols:
        logger.error(
            f"{dataset_name}: Cannot restore MultiIndex — missing columns: {missing_cols}."
        )
        return imputed_df

    try:
        imputed_df.set_index(index_cols, inplace=True)
        logger.info(f"{dataset_name}: MultiIndex restored using columns {index_cols}")
    except Exception as e:
        logger.error(f"{dataset_name}: Failed to restore MultiIndex: {e}")

    return imputed_df




def assert_cpd_type_encoded(*, df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Assert that 'cpd_type' exists and is globally label-encoded to integer.

    Parameters
    ----------
    df : pandas.DataFrame
        Table expected to contain an integer-encoded 'cpd_type' column.
    logger : logging.Logger
        Logger for status messages.

    Raises
    ------
    ValueError
        If 'cpd_type' is missing or not integer dtype.
    """
    if "cpd_type" not in df.columns:
        raise ValueError("Expected column 'cpd_type' before CLIPn training.")
    if not pd.api.types.is_integer_dtype(df["cpd_type"]):
        raise ValueError("Column 'cpd_type' must be globally integer-encoded before training.")
    n_na = int(df["cpd_type"].isna().sum())
    if n_na:
        logger.warning("cpd_type contains %d NA values; these rows may be ignored by the trainer.", n_na)
    logger.info("cpd_type dtype is integer; OK.")


def validate_frozen_features_manifest(
    *,
    feature_list_path: Path,
    logger: logging.Logger,
) -> list[str]:
    """
    Validate that the frozen feature manifest contains no metadata or 'cpd_type'.

    Parameters
    ----------
    feature_list_path : pathlib.Path
        Path to the '<experiment>_features_used.tsv' file.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    list[str]
        Ordered list of features from the manifest.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    ValueError
        If any metadata or 'cpd_type' is present in the feature list.
    """
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Frozen feature manifest not found: {feature_list_path}")

    feats = pd.read_csv(feature_list_path, sep="\t")["feature"].astype(str).tolist()
    lower = {f.lower() for f in feats}
    meta = {m.lower() for m in (METADATA_COL_BLOCKLIST | TECHNICAL_FEATURE_BLOCKLIST)}  # uses your globals
    bad = lower.intersection(meta.union({"cpd_type"}))
    if bad:
        raise ValueError(f"Feature manifest contains metadata columns: {sorted(bad)}")
    logger.info("Frozen feature manifest validated: %d features; no metadata present.", len(feats))
    return feats


def assert_xy_alignment_strict(
    *,
    X: Dict[int, np.ndarray],
    y: Dict[int, np.ndarray],
    logger: logging.Logger,
) -> None:
    """
    Strictly assert that X and y lengths match per dataset.

    Parameters
    ----------
    X : dict[int, numpy.ndarray]
        Feature matrices per dataset id.
    y : dict[int, numpy.ndarray]
        Label vectors per dataset id.
    logger : logging.Logger
        Logger for status messages.

    Raises
    ------
    ValueError
        If any dataset has a length mismatch.
    """
    for k in sorted(set(X) | set(y)):
        if k not in X:
            raise ValueError(f"Dataset id {k} present in y but missing in X.")
        if k not in y:
            raise ValueError(f"Dataset id {k} present in X but missing in y.")
        nx = int(X[k].shape[0])
        ny = int(len(y[k]))
        if nx != ny:
            raise ValueError(f"Length mismatch for dataset id {k}: X={nx}, y={ny}.")
    logger.info("X/y alignment OK across %d dataset(s).", len(X))
