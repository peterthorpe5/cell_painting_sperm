#!/usr/bin/env python3
# coding: utf-8

"""
Run CLIPn Integration on Cell Painting Data
-------------------------------------------

This script:
- Loads and merges multiple reference and query datasets.
- Harmonises column features across datasets.
- Encodes labels for compatibility with CLIPn.
- Runs CLIPn integration analysis (either train on references or integrate all).
- Decodes labels post-analysis, restoring original annotations.
- Outputs results, including latent representations and similarity matrices.

All outputs are tab-separated (TSV). No comma-separated outputs are written.

Command-line arguments:
-----------------------
    --datasets_csv      : Path to TSV/CSV listing dataset names and paths
                          with columns: 'dataset' and 'path'.
    --out               : Directory to save outputs.
    --experiment        : Experiment name for file naming.
    --mode              : Operation mode ('reference_only' or 'integrate_all').
    --clipn_param       : Optional CLIPn parameter for logging only.
    --latent_dim        : Dimensionality of latent space (default: 20).
    --lr                : Learning rate for CLIPn (default: 1e-5).
    --epoch             : Number of training epochs (default: 500).
    --save_model        : If set, save the trained CLIPn model after training.
    --load_model        : Path (or glob) to a previously saved CLIPn model to load.
    --scaling_mode      : 'all', 'per_plate', or 'none' (default: 'all').
    --scaling_method    : 'robust' or 'standard' (default: 'robust').
    --skip_standardise  : If set, skip feature scaling.
    --reference_names   : Space-separated list of dataset names to use as
                          references (only used in 'reference_only' mode).
    --aggregate_method  : Aggregate image-level latent to compound-level
                          using 'median' (default), 'mean', 'min', or 'max'.
    --annotations       : Optional annotation file (TSV) to merge using
                          Plate_Metadata and Well_Metadata.
"""

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
import psutil
import concurrent.futures
import torch
import torch.serialization
import gzip
from clipn.model import CLIPn
from sklearn import set_config
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



from cell_painting.process_data import (
    prepare_data_for_clipn_from_df,
    run_clipn_simple,
    standardise_metadata_columns,)

# Global timer (for memory log timestamps)
_SCRIPT_START_TIME = time.time()

# Make sklearn return DataFrames
set_config(transform_output="pandas")



# =========================
# Logging and small helpers
# =========================

def setup_logging(out_dir: str | Path, experiment: str) -> logging.Logger:
    """
    Configure logging with stream (stderr) and file handlers.

    Parameters
    ----------
    out_dir : str | Path
        Output directory for logs.
    experiment : str
        Experiment name; used for the log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{experiment}_clipn.log"

    logger = logging.getLogger("clipn_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_filename, mode="w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("Python Version: %s", sys.version_info)
    logger.info("Command-line Arguments: %s", " ".join(sys.argv))
    logger.info("Experiment Name: %s", experiment)

    return logger


def configure_torch_performance(logger: logging.Logger) -> None:
    """
    Small runtime hints for speed.
    - On GPU: enable better matmul + cuDNN autotune.
    - On CPU: leave defaults (MKL/OpenBLAS already multithreaded).
    """
    try:
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")  # TF32/fast FP32 on Ampere+
            torch.backends.cudnn.benchmark = True       # tune best conv algo for fixed shapes
            logger.info("Torch perf: enabled high-precision matmul + cuDNN benchmark on GPU.")
        else:
            logger.info("Torch perf: CPU mode (no special tweaks).")
    except Exception as exc:
        logger.warning("Torch perf configuration skipped: %s", exc)


def _register_clipn_for_pickle() -> None:
    """
    Best-effort registration of CLIPn for Torch's safe unpickling.

    On older PyTorch versions the 'add_safe_globals' helper does not exist.
    In that case this function becomes a no-op and loading still works
    provided the CLIPn class is importable.
    """
    try:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([CLIPn])
    except Exception:
        # Optional hardening only; safe to ignore on older Torch.
        pass



def torch_load_compat(model_path: str, *, map_location: str | None = None, weights_only: bool | None = None):
    """
    Backwards-compatible torch.load.

    Tries to use the 'weights_only' argument when supported; falls back
    silently if the running PyTorch does not accept it.
    """
    try:
        if weights_only is None:
            return torch.load(f=model_path, map_location=map_location)
        return torch.load(f=model_path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older Torch without 'weights_only'
        return torch.load(f=model_path, map_location=map_location)


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


def _mode_nonnull(series: pd.Series) -> Optional[str]:
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
        cat_frames[col] = grouped[col].apply(func=_mode_nonnull)

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
TECHNICAL_FEATURE_BLOCKLIST = {"ImageNumber", "Number_Object_Number"}

def _exclude_technical_features(cols, logger):
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


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    plate_col: str | None = None,
    mode: str = "all",
    method: str = "robust",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Scale features globally or per-plate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and metadata.
    feature_cols : list[str]
        Names of feature columns to scale.
    plate_col : str | None
        Plate column name (required if mode='per_plate').
    mode : str
        One of: 'all', 'per_plate', 'none'.
    method : str
        One of: 'robust' or 'standard'.
    logger : logging.Logger | None
        Logger for status messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled features.
    """
    logger = logger or logging.getLogger("scaling")

    if not feature_cols:
        logger.warning("No feature columns to scale; skipping scaling.")
        return df

    if mode == "none":
        logger.info("No scaling applied.")
        return df

    scaler_cls = RobustScaler if method == "robust" else StandardScaler
    df_scaled = df.copy()

    if mode == "all":
        scaler = scaler_cls()
        df_scaled.loc[:, feature_cols] = scaler.fit_transform(df[feature_cols])
        logger.info("Scaled all features together using %s scaler.", method)

    elif mode == "per_plate":
        if plate_col is None or plate_col not in df.columns:
            raise ValueError("plate_col must be provided for per_plate scaling.")
        n_groups = df[plate_col].nunique(dropna=False)
        logger.info("Scaling per-plate across %d plate groups using %s scaler.", n_groups, method)
        for plate, idx in df.groupby(plate_col).groups.items():
            scaler = scaler_cls()
            idx = list(idx)
            df_scaled.loc[idx, feature_cols] = scaler.fit_transform(df.loc[idx, feature_cols])

    else:
        logger.warning("Unknown scaling mode '%s'. No scaling applied.", mode)

    return df_scaled


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

def _mode_strict(series: pd.Series) -> Optional[str]:
    """
    Return the most frequent non-null string in a Series, or None.

    Parameters
    ----------
    series : pandas.Series

    Returns
    -------
    Optional[str]
    """
    s = series.dropna()
    if s.empty:
        return None
    mode_vals = s.mode(dropna=True)
    return None if mode_vals.empty else str(mode_vals.iloc[0])


def aggregate_for_knn(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    level: str = "compound",
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate the table to the chosen granularity for k-NN.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned and scaled table with MultiIndex (Dataset, Sample) and metadata.
    feature_cols : list[str]
        Numeric feature columns used for distance computations.
    level : str
        One of {'compound', 'well', 'image'}.
    logger : logging.Logger

    Returns
    -------
    (X, meta) : tuple[pd.DataFrame, pd.DataFrame]
        X  : numeric matrix (rows = entities).
        meta : metadata for each row with an 'EntityID' column.

    Notes
    -----
    - 'compound' groups by cpd_id and takes the median of features.
    - 'well' groups by (Plate_Metadata, Well_Metadata).
    - 'image' keeps rows as-is, with EntityID = 'Dataset::Sample'.
    """
    meta_cols = [c for c in ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"] if c in df.columns]

    if level == "compound":
        if "cpd_id" not in df.columns:
            raise ValueError("aggregate_for_knn(level='compound') requires 'cpd_id' column.")
        g = df.groupby("cpd_id", sort=False, dropna=False)
        X = g[feature_cols].median(numeric_only=True)
        meta = pd.DataFrame({
            "EntityID": X.index.astype(str),
            "cpd_id": X.index.astype(str),
            "cpd_type": g["cpd_type"].apply(_mode_strict) if "cpd_type" in df.columns else None,
            "Library": g["Library"].apply(_mode_strict) if "Library" in df.columns else None,
        })
        meta = meta.dropna(axis=1, how="all")
        logger.info("Aggregated to compounds: %d unique cpd_id.", X.shape[0])

    elif level == "well":
        needed = {"Plate_Metadata", "Well_Metadata"}
        if not needed.issubset(df.columns):
            raise ValueError("aggregate_for_knn(level='well') requires Plate_Metadata and Well_Metadata.")
        g = df.groupby(["Plate_Metadata", "Well_Metadata"], sort=False, dropna=False)
        X = g[feature_cols].median(numeric_only=True)
        meta = X.reset_index()[["Plate_Metadata", "Well_Metadata"]].copy()
        meta["EntityID"] = meta["Plate_Metadata"].astype(str) + "::" + meta["Well_Metadata"].astype(str)
        # Attach optional modes
        if "cpd_id" in df.columns:
            meta["cpd_id"] = g["cpd_id"].apply(_mode_strict).reset_index(drop=True)
        if "cpd_type" in df.columns:
            meta["cpd_type"] = g["cpd_type"].apply(_mode_strict).reset_index(drop=True)
        if "Library" in df.columns:
            meta["Library"] = g["Library"].apply(_mode_strict).reset_index(drop=True)
        meta = meta.set_index(X.index)
        logger.info("Aggregated to wells: %d unique Plate×Well.", X.shape[0])

    elif level == "image":
        # Keep as-is; define an ID string from MultiIndex
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Dataset", "Sample"]:
            raise ValueError("Expected MultiIndex ['Dataset','Sample'] for image-level KNN.")
        X = df[feature_cols].copy()
        meta = df[meta_cols].copy()
        meta = meta if not meta.empty else pd.DataFrame(index=X.index)
        meta = meta.reset_index()
        meta["EntityID"] = meta["Dataset"].astype(str) + "::" + meta["Sample"].astype(str)
        meta = meta.set_index(X.index)
        logger.info("Using image-level entities: %d rows.", X.shape[0])

    else:
        raise ValueError("level must be one of: 'compound', 'well', 'image'.")

    # Ensure clean indices for downstream
    X = X.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    return X, meta


def run_knn_analysis(
    *,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    k: int = 10,
    metric: str = "cosine",
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute top-k nearest neighbours for each row in X.

    Parameters
    ----------
    X : pandas.DataFrame
        Numeric feature matrix; rows are entities.
    meta : pandas.DataFrame
        Metadata with 'EntityID' column, aligned to X by row.
    k : int
        Number of neighbours to return (excluding self).
    metric : str
        One of {'cosine', 'euclidean', 'correlation'}.
    logger : logging.Logger

    Returns
    -------
    pandas.DataFrame
        Long table with columns:
        ['QueryID','NeighbourID','rank','distance', ...metadata columns for query and neighbour...]
    """
    if metric in {"cosine", "euclidean"}:
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric=metric)
        nn.fit(X.values)
        dists, idxs = nn.kneighbors(X.values, return_distance=True)
    elif metric == "correlation":
        # Full pairwise distances just once; for large X this can be heavy
        logger.info("Computing pairwise correlation distances; this can be memory intensive.")
        D = pairwise_distances(X.values, metric="correlation")
        # Argsort rows; skip the diagonal later
        idxs = np.argsort(D, axis=1)[:, : min(k + 1, X.shape[0])]
        # Gather distances
        row_indices = np.arange(X.shape[0])[:, None]
        dists = D[row_indices, idxs]
        del D
    else:
        raise ValueError("metric must be 'cosine', 'euclidean' or 'correlation'.")

    rows = []
    entity_ids = meta["EntityID"].astype(str).tolist()
    # Prepare neighbour metadata access
    meta_cols = [c for c in meta.columns if c != "EntityID"]

    for i in range(len(X)):
        # Remove self if present at position 0
        neigh_i = idxs[i].tolist()
        dist_i = dists[i].tolist()
        pairs = [(j, d) for j, d in zip(neigh_i, dist_i) if j != i]
        pairs = pairs[:k]

        for rank, (j, d) in enumerate(pairs, start=1):
            row = {
                "QueryID": entity_ids[i],
                "NeighbourID": entity_ids[j],
                "rank": rank,
                "distance": float(d),
            }
            # Attach selected metadata for query and neighbour
            for c in meta_cols:
                row[f"Query_{c}"] = meta.iloc[i][c]
                row[f"Neighbour_{c}"] = meta.iloc[j][c]
            rows.append(row)

    out = pd.DataFrame(rows)
    logger.info("Computed k-NN: %d query rows × %d neighbours -> %d pairs.",
                len(X), k, out.shape[0])
    return out


def simple_knn_qc(
    *,
    knn_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute simple quality metrics from a k-NN table.

    Parameters
    ----------
    knn_df : pandas.DataFrame
        Output from run_knn_analysis().

    Returns
    -------
    pandas.DataFrame
        One-row summary with basic neighbour "hit rates" where available.
    """
    mets = {}
    # Same cpd_id rate (if both sides available)
    if {"Query_cpd_id", "Neighbour_cpd_id"}.issubset(knn_df.columns):
        same = (knn_df["Query_cpd_id"].astype(str) == knn_df["Neighbour_cpd_id"].astype(str))
        mets["same_cpd_id_rate"] = float(same.mean())

    if {"Query_cpd_type", "Neighbour_cpd_type"}.issubset(knn_df.columns):
        same = (knn_df["Query_cpd_type"].astype(str) == knn_df["Neighbour_cpd_type"].astype(str))
        mets["same_cpd_type_rate"] = float(same.mean())

    # Dataset leakage proxy (prefer mixing rather than matching)
    if {"Query_Library", "Neighbour_Library"}.issubset(knn_df.columns):
        same_lib = (knn_df["Query_Library"].astype(str) == knn_df["Neighbour_Library"].astype(str))
        mets["same_library_neighbour_rate"] = float(same_lib.mean())

    summary = pd.DataFrame([mets]) if mets else pd.DataFrame([{}])
    logger.info("k-NN QC summary: %s", summary.to_dict(orient="records")[0])
    return summary


def save_knn_outputs(
    *,
    knn_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    out_dir: Path,
    experiment: str,
    save_full_matrix: bool = False,
    metric: str,
    logger: logging.Logger,
) -> None:
    """
    Persist k-NN outputs as TSVs. Optionally save a full pairwise matrix
    (guarded to small sizes).

    Parameters
    ----------
    knn_df : pandas.DataFrame
    qc_df : pandas.DataFrame
    X : pandas.DataFrame
    meta : pandas.DataFrame
    out_dir : pathlib.Path
    experiment : str
    save_full_matrix : bool
    metric : str
    logger : logging.Logger
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nn_path = out_dir / f"{experiment}_nearest_neighbours.tsv"
    knn_df.to_csv(nn_path, sep="\t", index=False)
    logger.info("Wrote k-NN pairs -> %s", nn_path)

    qc_path = out_dir / f"{experiment}_knn_qc_summary.tsv"
    qc_df.to_csv(qc_path, sep="\t", index=False)
    logger.info("Wrote k-NN QC summary -> %s", qc_path)

    if save_full_matrix:
        n = X.shape[0]
        if n > 5000:
            logger.warning("Full pairwise matrix skipped (n=%d too large).", n)
            return
        if metric == "correlation":
            D = pairwise_distances(X.values, metric="correlation")
        else:
            # Use cosine/Euclidean
            D = pairwise_distances(X.values, metric=metric)
        dm = pd.DataFrame(D, index=meta["EntityID"], columns=meta["EntityID"])
        dm_path = out_dir / f"{experiment}_pairwise_distance_matrix.tsv"
        dm.to_csv(dm_path, sep="\t")
        logger.info("Wrote full pairwise matrix (%d×%d) -> %s", n, n, dm_path)



# ==========================
# I/O and harmonisation path
# ==========================

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
    df = _read_csv_fast(path, delimiter)

    logger.debug("[%s] Columns after initial load: %s", name, df.columns.tolist())

    if df.index.name in metadata_cols:
        promoted_col = df.index.name
        df[promoted_col] = df.index
        df.index.name = None
        logger.warning("[%s] Promoted index '%s' to column to preserve metadata.", name, promoted_col)

    df = ensure_library_column(df=df, filepath=path, logger=logger, value=name)
    df = standardise_metadata_columns(df, logger=logger, dataset_name=name)

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
        logger.info("Wrote %s rows × %s cols -> %s", out.shape[0], out.shape[1], path)



def harmonise_numeric_columns(
    dataframes: Dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Subset to the intersection of numeric columns and preserve metadata columns.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Mapping from dataset name to DataFrame.
    logger : logging.Logger
        Logger.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], list[str]]
        Harmonised dataframes and the list of common numeric columns preserved.
    """
    numeric_cols_sets = [set(df.select_dtypes(include=[np.number]).columns) for df in dataframes.values()]
    common_cols = sorted(set.intersection(*numeric_cols_sets)) if numeric_cols_sets else []
    common_cols = _exclude_technical_features(common_cols, logger)
    logger.info("Harmonised numeric columns across datasets: %d", len(common_cols))

    metadata_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]

    for name, df in dataframes.items():
        numeric_df = df[common_cols] if common_cols else df.select_dtypes(include=[np.number])
        metadata_df = df[metadata_cols]
        df_harmonised = pd.concat(objs=[numeric_df, metadata_df], axis=1)
        assert df_harmonised.index.equals(df.index), f"Index mismatch after harmonisation in '{name}'."
        dataframes[name] = df_harmonised
        logger.debug("[%s] Harmonisation successful, final columns: %s", name, df_harmonised.columns.tolist())

    return dataframes, common_cols


def load_and_harmonise_datasets(
    datasets_csv: str,
    logger: logging.Logger,
    mode: str | None = None,
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

    logger.info("Loading datasets individually")
    for name, path in dataset_paths.items():
        try:
            dataframes[name] = load_single_dataset(name=name, path=path, logger=logger, metadata_cols=metadata_cols)
        except ValueError as exc:
            logger.error("Loading dataset '%s' failed: %s", name, exc)
            raise

    return harmonise_numeric_columns(dataframes=dataframes, logger=logger)


# ============================
# Encoding / Decoding utilities
# ============================

def decode_labels(df: pd.DataFrame, encoders: Dict[str, LabelEncoder], logger: logging.Logger) -> pd.DataFrame:
    """
    Decode categorical columns using fitted LabelEncoders.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns may be label-encoded.
    encoders : dict[str, LabelEncoder]
        Mapping of column names to fitted LabelEncoder objects.
    logger : logging.Logger
        Logger for progress and warnings.

    Returns
    -------
    pandas.DataFrame
        DataFrame with decoded columns where possible.

    Notes
    -----
    - Robust to columns that are integer-like but stored as 'object' or 'float'
      (e.g. '0'/'1' strings or 0.0/1.0 floats).
    - Leaves already-decoded string columns unchanged.
    """
    for col, le in encoders.items():
        if col not in df.columns:
            logger.warning("decode_labels: Column '%s' not found in DataFrame. Skipping.", col)
            continue

        s = df[col]

        # Try to coerce to integer codes robustly
        s_codes = pd.to_numeric(s, errors="coerce").astype("Int64")
        n_codes = int(s_codes.notna().sum())

        if n_codes == 0:
            # Nothing integer-like to decode; assume already-decoded strings
            logger.info("decode_labels: Column '%s' appears already decoded or non-integer; leaving as-is.", col)
            continue

        # Build a mapping from code -> original label
        mapping = {i: cls for i, cls in enumerate(le.classes_)}

        decoded = s_codes.map(mapping)
        # Keep original values where decode failed (e.g. unexpected codes)
        df[col] = decoded.where(decoded.notna(), other=s.astype(str))

        logger.info("decode_labels: Decoded column '%s' (%d/%d values).", col, int(decoded.notna().sum()), len(s))
    return df


def encode_labels(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode object/category columns (excluding 'cpd_id') using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to encode.
    logger : logging.Logger
        Logger for debug information.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        The encoded DataFrame and a mapping of column name to LabelEncoder.
    """
    encoders: Dict[str, LabelEncoder] = {}
    skip_columns = {"cpd_id"}

    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("Encoding %d object/category columns (excluding 'cpd_id' if present).", len(obj_cols))

    for col in obj_cols:
        if col in skip_columns:
            logger.debug("Skipping encoding for column '%s'", col)
            continue
        le = LabelEncoder()
        df.loc[:, col] = le.fit_transform(df[col])
        encoders[col] = le
        logger.debug("Encoded column '%s' with %d classes.", col, len(le.classes_))
    return df, encoders


# ====================
# CLIPn core functions
# ====================

def extend_model_encoders(
    model: CLIPn,
    new_keys: Iterable[int],
    reference_key: int,
    logger: logging.Logger,
) -> None:
    """
    Extend CLIPn model's encoder mapping for new datasets using a reference encoder.

    Parameters
    ----------
    model : CLIPn
        Trained CLIPn model object.
    new_keys : Iterable[int]
        Keys of new datasets to be projected.
    reference_key : int
        Key of the reference dataset to copy the encoder from.
    logger : logging.Logger
        Logger instance.
    """
    for new_key in new_keys:
        model.model.encoders[new_key] = model.model.encoders[reference_key]
        logger.debug("Assigned encoder for dataset key %s using reference encoder %s", new_key, reference_key)


def _apply_threads(n: int, logger):
    """
    Set BLAS/OpenMP and PyTorch threads to exactly 'n'.

    Parameters
    ----------
    n : int
        Thread count requested on the command line.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    int
        The thread count actually set.
    """
    n = max(1, int(n))

    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
    ):
        os.environ[var] = str(n)

    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger.info("CPU threads set to %d (from --cpu_threads)", n)
    return n




def run_clipn_integration(
    df: pd.DataFrame,
    logger: logging.Logger,
    clipn_param: str,
    output_path: str | Path,
    experiment: str,
    mode: str,
    latent_dim: int,
    lr: float,
    epochs: int,
    skip_standardise: bool = False,  # kept for signature parity
) -> Tuple[pd.DataFrame, Dict[str, List[str]], CLIPn, Dict[int, str]]:
    """
    Train a CLIPn model on the provided DataFrame and return latent representations.

    Parameters
    ----------
    df : pd.DataFrame
        Combined input DataFrame with MultiIndex (Dataset, Sample).
    logger : logging.Logger
        Logger instance.
    clipn_param : str
        Optional parameter for logging (no functional effect here).
    output_path : str | Path
        Directory to save latent arrays.
    experiment : str
        Experiment name.
    mode : str
        Operation mode (for filename context).
    latent_dim : int
        Dimensionality of the latent space.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    skip_standardise : bool
        Unused here (kept for API compatibility).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, list[str]], CLIPn, dict[int, str]]
        Combined latent DataFrame (MultiIndex),
        dictionary of cpd_ids per dataset,
        trained CLIPn model,
        dataset key mapping.
    """
    logger.info("Running CLIPn integration with param: %s", clipn_param)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Numeric feature column count: %d", len(numeric_cols))
    logger.info("Combined DataFrame shape: %s", df.shape)
    logger.debug("Head of combined DataFrame:\n%s", df.head())

    if not numeric_cols:
        logger.error(
            "No numeric feature columns found after harmonisation. "
            "Possible causes: no overlap of features, all numeric columns are NaN, or wrong dtypes."
        )
        raise ValueError("No numeric columns available for CLIPn.")

    data_dict, label_dict, label_mappings, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df)
    latent_dict, model, loss = run_clipn_simple(
        data_dict,
        label_dict,
        latent_dim=latent_dim,
        lr=lr,
        epochs=epochs,
    )

    if isinstance(loss, (list, np.ndarray)):
        logger.info("CLIPn final loss: %.6f", loss[-1])
    else:
        logger.info("CLIPn loss: %s", loss)

    latent_frames = []
    for i, latent in latent_dict.items():
        name = dataset_key_mapping[i]
        df_latent = pd.DataFrame(latent)
        df_latent.index = pd.MultiIndex.from_product(
            [[name], range(len(df_latent))], names=["Dataset", "Sample"]
        )
        latent_frames.append(df_latent)

    latent_combined = pd.concat(latent_frames)

    # Save latent as NPZ (with string keys)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    latent_file = output_path / f"{experiment}_{mode}_CLIPn_latent_representations.npz"
    latent_dict_str_keys = {str(k): v for k, v in latent_dict.items()}
    np.savez(file=latent_file, **latent_dict_str_keys)
    logger.info("Latent representations saved to: %s", latent_file)

    latent_file_id = output_path / f"{experiment}_{mode}_CLIPn_latent_representations_cpd_id.npz"
    cpd_ids_array = {f"cpd_ids_{k}": np.array(v) for k, v in cpd_ids.items()}
    np.savez(file=latent_file_id, **latent_dict_str_keys, **cpd_ids_array)

    post_clipn_dir = output_path / "post_clipn"
    post_clipn_dir.mkdir(parents=True, exist_ok=True)
    post_latent_file = post_clipn_dir / f"{experiment}_{mode}_CLIPn_latent_representations.npz"
    np.savez(file=post_latent_file, **latent_dict_str_keys)

    return latent_combined, cpd_ids, model, dataset_key_mapping


# =========================
# Downstream / merge helpers
# =========================

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


def _read_csv_fast(path: str, delimiter: str) -> pd.DataFrame:
    # Try pyarrow engine (fast); fall back to pandas' python engine.
    try:
        return pd.read_csv(path, delimiter=delimiter, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, delimiter=delimiter, engine="python", compression="infer")


def read_table_auto(path: str) -> pd.DataFrame:
    """Read CSV/TSV with automatic delimiter detection (prefers tab)."""
    sep = detect_csv_delimiter(path)
    return pd.read_csv(filepath_or_buffer=path, sep=sep)


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


# =====
# Main
# =====

def main(args: argparse.Namespace) -> None:
    """
    Execute CLIPn integration pipeline from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    logger = setup_logging(out_dir=args.out, experiment=args.experiment)
    configure_torch_performance(logger)
    logger.info("Starting CLIPn integration pipeline")
    logger.info("PyTorch Version: %s", getattr(torch, "__version__", "unknown"))


    # Threading: simple and explicit
    _CLIPN_THREADS = _apply_threads(args.cpu_threads, logger)



    _register_clipn_for_pickle()
    post_clipn_dir = Path(args.out) / "post_clipn"
    post_clipn_dir.mkdir(parents=True, exist_ok=True)

    # Load + harmonise
    dataframes, common_cols = load_and_harmonise_datasets(
        datasets_csv=args.datasets_csv,
        logger=logger,
        mode=args.mode,
    )
    logger.info("Loaded and harmonised %d datasets from %s", len(dataframes), args.datasets_csv)

    # Per-dataset sanity checks
    for name, df in dataframes.items():
        missing_meta = [c for c in ["cpd_id", "cpd_type", "Library"] if c not in df.columns]
        if missing_meta:
            raise ValueError(
                f"Sanity check failed after harmonisation for '{name}': missing {missing_meta}"
            )

    bad_index = [
        (name, type(df.index), df.index.names)
        for name, df in dataframes.items()
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Dataset", "Sample"]
    ]
    if bad_index:
        details = "; ".join([f"{n} -> {t} names={names}" for n, t, names in bad_index])
        raise ValueError(f"Expected MultiIndex ['Dataset','Sample'] for all datasets; got: {details}")

    logger.info("Per-dataset sanity checks passed.")

    # Concatenate (deterministic order), normalise index names, sort for determinism
    combined_df = pd.concat(
        objs=[dataframes[name] for name in dataframes.keys()],
        axis=0,
        sort=False,
        copy=False,
    )
    if list(combined_df.index.names) != ["Dataset", "Sample"]:
        combined_df.index = combined_df.index.set_names(names=["Dataset", "Sample"])
    combined_df = combined_df.sort_index()

    dupe_count = combined_df.index.duplicated().sum()
    if dupe_count:
        logger.warning("Found %d duplicate (Dataset, Sample) index rows after concat.", dupe_count)

    logger.debug("Columns at this stage, combined: %s", combined_df.columns.tolist())
    log_memory_usage(logger=logger, prefix="[After loading datasets] ")

    # Metadata columns (never scale these)
    meta_columns = ["cpd_id", "cpd_type", "Plate_Metadata", "Well_Metadata", "Library"]
    for col in meta_columns:
        if col not in combined_df.columns:
            raise ValueError(f"Metadata column '{col}' not found in combined DataFrame after harmonisation.")
    logger.info("Metadata columns present in combined DataFrame: %s", meta_columns)
    logger.info("Combined DataFrame shape after harmonisation: %s", combined_df.shape)

    # Identify feature columns
    feature_cols = [
        col for col in combined_df.columns
        if col not in meta_columns and pd.api.types.is_numeric_dtype(combined_df[col])
    ]
    # drop technicals from the features we’re going to scale/project
    feature_cols = _exclude_technical_features(feature_cols, logger)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after harmonisation. Check feature overlap and dtypes.")

    # Clean obvious non-finite values before any scaling
    combined_df = clean_nonfinite_features(
        df=combined_df,
        feature_cols=feature_cols,
        logger=logger,
        label="pre-scaling",
    )


    # Robust clean + impute (pre-scaling)
    combined_df, dropped_feats = clean_and_impute_features(
        df=combined_df,
        feature_cols=feature_cols,
        logger=logger,
        groupby_cols=["Dataset", "Plate_Metadata"] if "Plate_Metadata" in combined_df.columns else ["Dataset"],
        max_nan_col_frac=0.30,
        max_nan_row_frac=0.80,
    )
    if dropped_feats:
        feature_cols = [c for c in feature_cols if c not in dropped_feats]

    # Hard guard
    if combined_df.shape[0] == 0:
        raise ValueError(
            "No rows left after NaN handling. Loosen thresholds (max_nan_col_frac/max_nan_row_frac) "
            "or inspect inputs for pervasive missingness."
        )



    # Optional scaling
    if args.skip_standardise:
        logger.info("Skipping feature scaling (--skip_standardise set).")
        df_scaled_all = combined_df
    else:
        logger.info("Scaling numeric features using mode='%s', method='%s'", args.scaling_mode, args.scaling_method)
        df_scaled_all = scale_features(
            df=combined_df,
            feature_cols=feature_cols,
            plate_col="Plate_Metadata",
            mode=args.scaling_mode,
            method=args.scaling_method,
            logger=logger,
        )
        logger.info("Scaled combined DataFrame shape: %s", df_scaled_all.shape)
        log_memory_usage(logger, prefix="[After scaling] ")
    # Clean again in case scaling produced NaN/Inf (e.g. zero-variance issues)
    df_scaled_all = clean_nonfinite_features(
        df=df_scaled_all,
        feature_cols=feature_cols,
        logger=logger,
        label="post-scaling",
    )

    # ===== Optional: k-NN baseline on the pre-CLIPn feature space =====
    if args.knn_only:
        logger.info("Running k-NN baseline (pre-CLIPn) with metric='%s', level='%s', k=%d",
                    args.knn_metric, args.knn_level, args.knn_k)

        # Build the matrix at the chosen granularity
        X_knn, meta_knn = aggregate_for_knn(
            df=df_scaled_all,
            feature_cols=feature_cols,
            level=args.knn_level,
            logger=logger,
        )

        # Compute neighbours
        knn_df = run_knn_analysis(
            X=X_knn,
            meta=meta_knn,
            k=args.knn_k,
            metric=args.knn_metric,
            logger=logger,
        )

        # Simple QC summary
        qc_df = simple_knn_qc(
            knn_df=knn_df,
            logger=logger,
        )

        # Save
        knn_dir = Path(args.out) / args.knn_out_subdir
        save_knn_outputs(
            knn_df=knn_df,
            qc_df=qc_df,
            X=X_knn,
            meta=meta_knn,
            out_dir=knn_dir,
            experiment=args.experiment,
            save_full_matrix=args.knn_save_full_matrix,
            metric=args.knn_metric,
            logger=logger,
        )
        if args.knn_only:
            logger.info("k-NN baseline completed; exiting early (--knn_only set).")
            return



    # Encode labels (cpd_id is explicitly not encoded)
    logger.info("Encoding categorical labels for CLIPn compatibility")
    # Ensure key categoricals are strings so they get encoded → can be decoded later
    for _col in ("cpd_type", "Library", "Plate_Metadata", "Well_Metadata"):
        if _col in df_scaled_all.columns:
            df_scaled_all.loc[:, _col] = df_scaled_all[_col].astype("string")

    df_encoded, encoders = encode_labels(df=df_scaled_all.copy(), logger=logger)
    log_memory_usage(logger, prefix="[After encoding] ")

    # Keep a decoded metadata view for later merge
    decoded_meta_df = decode_labels(df=df_encoded.copy(), encoders=encoders, logger=logger)[
        ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    ].reset_index()

    # =========================
    # Mode: reference-only flow
    # =========================
    if args.mode == "reference_only":
        reference_names = args.reference_names
        logger.info("Using reference datasets %s for training; projecting all others.", reference_names)

        # Check that reference names exist in the MultiIndex level
        ds_index_values = set(df_encoded.index.get_level_values("Dataset"))
        missing_refs = [n for n in reference_names if n not in ds_index_values]
        if missing_refs:
            raise ValueError(f"Reference dataset(s) not found in combined_df index: {missing_refs}")

        query_names = [name for name in dataframes if name not in reference_names]
        logger.info("Training on: %s; projecting: %s", reference_names, query_names)

        reference_df = df_encoded.loc[reference_names]
        query_df = df_encoded.loc[query_names] if query_names else pd.DataFrame()

        # Load model path usage policy:
        if args.load_model and args.mode != "integrate_all":
            logger.warning(
                "Loading a pre-trained model is typically used with 'integrate_all'. "
                "Proceeding with 'reference_only' but ensuring encoders align."
            )

        if args.load_model:
            model_files = glob.glob(args.load_model)
            if not model_files:
                raise FileNotFoundError(f"No model files matched pattern: {args.load_model}")
            model_path = model_files[0]
            logger.info("Loading pre-trained CLIPn model from: %s", model_path)
            model = torch_load_compat(model_path=model_path, weights_only=False)


            # Prepare and predict training references
            data_dict, _, _, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(reference_df)
            latent_dict = model.predict(data_dict)

            latent_frames = []
            for i, latent in latent_dict.items():
                name = dataset_key_mapping[i]
                df_latent = pd.DataFrame(latent)
                df_latent.index = pd.MultiIndex.from_product(
                    [[name], range(len(df_latent))], names=["Dataset", "Sample"]
                )
                latent_frames.append(df_latent)
            latent_df = pd.concat(latent_frames)

        else:
            # Train a new model on references
            logger.info("Training new CLIPn model on reference datasets")
            latent_df, cpd_ids, model, dataset_key_mapping = run_clipn_integration(
                df=reference_df,
                logger=logger,
                clipn_param=args.clipn_param,
                output_path=args.out,
                experiment=args.experiment,
                mode=args.mode,
                latent_dim=args.latent_dim,
                lr=args.lr,
                epochs=args.epoch,
                skip_standardise=args.skip_standardise,
            )
            if args.save_model:
                model_path = Path(args.out) / f"{args.experiment}_clipn_model.pt"
                torch.save(obj=model, f=model_path)
                logger.info("Trained CLIPn model saved to: %s", model_path)

        log_memory_usage(logger, prefix="[After CLIPn training] ")

        # Attach metadata for training references
        latent_training_df = latent_df.reset_index()
        training_metadata_df = decoded_meta_df[decoded_meta_df["Dataset"].isin(reference_names)]
        latent_training_df = latent_training_df.merge(
            right=training_metadata_df,
            on=["Dataset", "Sample"],
            how="left",
        )

        # Inject cpd_id from cpd_ids dict for safety (overrides if present)
        assert all(name in cpd_ids for name in latent_training_df["Dataset"].unique()), \
            "Missing cpd_id mappings for some datasets."
        latent_training_df["cpd_id"] = latent_training_df.apply(
            func=lambda row: cpd_ids.get(row["Dataset"], [None])[row["Sample"]]
            if row["Sample"] < len(cpd_ids.get(row["Dataset"], [])) else None,
            axis=1,
        )
        training_output_path = Path(args.out) / "training"
        training_output_path.mkdir(parents=True, exist_ok=True)

        safe_to_csv(df=latent_training_df,
                    path=training_output_path / "training_only_latent.tsv",
                    sep="\t",
                    logger=logger,)

        logger.debug("First 10 cpd_id values:\n%s", latent_training_df["cpd_id"].head(10).to_string(index=False))
        logger.debug("Unique cpd_id values (first 10): %s", latent_training_df["cpd_id"].unique()[:10])

        # Project queries (if any)
        if not query_df.empty:
            logger.info("Projecting query datasets onto reference latent space: %s", query_names)
            # Extend dataset_key_mapping to include query datasets
            max_existing_key = max(dataset_key_mapping.keys(), default=-1)
            new_keys = list(range(max_existing_key + 1, max_existing_key + 1 + len(query_names)))
            if len(new_keys) != len(query_names):
                raise ValueError("Internal error: key/name length mismatch while extending dataset_key_mapping.")
            for new_key, name in zip(new_keys, query_names):
                dataset_key_mapping[new_key] = name


            # Identify a reference encoder to copy from
            try:
                reference_encoder_key = next(
                    k for k, v in dataset_key_mapping.items()
                    if v in reference_names and k in model.model.encoders
                )
            except StopIteration as exc:
                logger.error(
                    "No valid reference_encoder_key found. "
                    "None of the references matched trained encoders."
                )
                raise exc

            extend_model_encoders(model=model, new_keys=new_keys, reference_key=reference_encoder_key, logger=logger)

            # Build model input for queries (drop metadata cols that the model should not see)
            dataset_key_mapping_inv = {v: k for k, v in dataset_key_mapping.items()}
            query_groups = query_df.groupby(level="Dataset", sort=False)
            cols_to_drop = [
                c for c in ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
                if c in query_df.columns
            ]
            query_data_dict_corrected = {
                dataset_key_mapping_inv[name]: group.droplevel("Dataset").drop(columns=cols_to_drop).values
                for name, group in query_groups if name in dataset_key_mapping_inv
            }

            projected_dict = model.predict(query_data_dict_corrected)
            if not projected_dict:
                logger.warning("model.predict() returned an empty dictionary. Check dataset keys and inputs.")
            else:
                logger.debug("Projected %d datasets into latent space: %s", len(projected_dict), list(projected_dict.keys()))

            projected_frames = []
            query_cpd_ids: Dict[str, List[str]] = {}
            for i, latent in projected_dict.items():
                name = dataset_key_mapping[i]
                df_proj = pd.DataFrame(latent)
                df_proj.index = pd.MultiIndex.from_product(
                    [[name], range(len(df_proj))], names=["Dataset", "Sample"]
                )
                projected_frames.append(df_proj)
                # recover cpd_id from original query_df by ordered Sample
                query_cpd_ids[name] = query_df.loc[name]["cpd_id"].tolist()

            latent_query_df = pd.concat(projected_frames).reset_index()
            latent_query_df["cpd_id"] = latent_query_df.apply(
                func=lambda row: query_cpd_ids.get(row["Dataset"], [None])[row["Sample"]],
                axis=1,
            )

            query_output_path = Path(args.out) / "query_only" / f"{args.experiment}_query_only_latent.tsv"
            query_output_path.parent.mkdir(parents=True, exist_ok=True)
            safe_to_csv(df=latent_query_df,
                        path=query_output_path,
                        sep="\t",
                        logger=logger,)
            logger.info("Query-only latent data saved to %s", query_output_path)

            # Merge training + query for downstream combined decode/outputs
            latent_df = pd.concat(
                [latent_df, latent_query_df.set_index(["Dataset", "Sample"])],
                axis=0,
                sort=False,
            )
            cpd_ids.update(query_cpd_ids)

    # ======================
    # Mode: integrate-all flow
    # ======================
    else:
        logger.info("Training and integrating CLIPn on all datasets")
        if args.load_model:
            model_files = glob.glob(args.load_model)
            if not model_files:
                raise FileNotFoundError(f"No model files matched pattern: {args.load_model}")
            model_path = model_files[0]
            logger.info("Loading pre-trained CLIPn model from: %s", model_path)
            model = torch_load_compat(model_path=model_path, weights_only=False)



            # Prepare and predict latent with loaded model
            data_dict, _, _, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df_encoded)
            latent_dict = model.predict(data_dict)

            latent_frames = []
            for i, latent in latent_dict.items():
                name = dataset_key_mapping[i]
                df_latent = pd.DataFrame(latent)
                df_latent.index = pd.MultiIndex.from_product(
                    [[name], range(len(df_latent))], names=["Dataset", "Sample"]
                )
                latent_frames.append(df_latent)
            latent_df = pd.concat(latent_frames)

        else:
            latent_df, cpd_ids, model, dataset_key_mapping = run_clipn_integration(
                df=df_encoded,
                logger=logger,
                clipn_param=args.clipn_param,
                output_path=args.out,
                experiment=args.experiment,
                mode=args.mode,
                latent_dim=args.latent_dim,
                lr=args.lr,
                epochs=args.epoch,
                skip_standardise=args.skip_standardise,
            )
            if args.save_model:
                model_path = Path(args.out) / f"{args.experiment}_clipn_model.pt"
                torch.save(obj=model, f=model_path)
                logger.info("Trained CLIPn model saved to: %s", model_path)

    # =========================
    # Decode + persist artefacts
    # =========================
    latent_df = latent_df.reset_index()
    latent_df = pd.merge(
        left=latent_df,
        right=decoded_meta_df,
        on=["Dataset", "Sample"],
        how="left",
    )

    decoded_df = decode_labels(df=latent_df.copy(), encoders=encoders, logger=logger)

    # Clean up duplicate cpd_id columns if any
    if "cpd_id_x" in decoded_df.columns or "cpd_id_y" in decoded_df.columns:
        decoded_df["cpd_id"] = (
            decoded_df.get("cpd_id_x", pd.Series(dtype=object))
            .combine_first(decoded_df.get("cpd_id_y", pd.Series(dtype=object)))
            .combine_first(decoded_df.get("cpd_id", pd.Series(dtype=object)))
        )
        decoded_df = decoded_df.drop(columns=[c for c in ["cpd_id_x", "cpd_id_y"] if c in decoded_df.columns])

    # Drop rows missing cpd_id (sensible default)
    n_before = decoded_df.shape[0]
    decoded_df = decoded_df[decoded_df["cpd_id"].notna()]
    n_after = decoded_df.shape[0]
    if n_before != n_after:
        logger.warning("Dropped %d rows with missing cpd_id after decoding/merge.", n_before - n_after)

    # Persist decoded outputs (TSV only)
    main_decoded_path = Path(args.out) / f"{args.experiment}_decoded.tsv"
    safe_to_csv(df=decoded_df,
                path=main_decoded_path,
                sep="\t",
                logger=logger,)

    logger.info("Decoded data saved to %s", main_decoded_path)

    post_decoded_path = post_clipn_dir / f"{args.experiment}_decoded.tsv"
    safe_to_csv(df=decoded_df,
                path=post_decoded_path,
                sep="\t",
                logger=logger,)
    logger.info("Decoded data saved to %s", post_decoded_path)


    # Optional compound-level aggregation (from decoded table; categoricals by mode)
    if getattr(args, "aggregate_method", None):
        try:
            df_compound = aggregate_latent_from_decoded(
                decoded_df=decoded_df,
                aggregate=args.aggregate_method,
                logger=logger,
            )
        except Exception:
            logger.exception("Failed to aggregate latent space from decoded table.")
            raise

        agg_path = post_clipn_dir / f"{args.experiment}_CLIPn_latent_aggregated_{args.aggregate_method}.tsv"
        safe_to_csv(
            df=df_compound,
            path=agg_path,
            sep="\t",
            logger=logger,
        )
        logger.info("Aggregated latent space saved to: %s", agg_path)


    # Plate/Well lookup (if present)
    if {"Plate_Metadata", "Well_Metadata"}.issubset(decoded_df.columns):
        plate_well_df = decoded_df[["Dataset", "Sample", "cpd_id", "Plate_Metadata", "Well_Metadata"]].copy()
        plate_well_file = post_clipn_dir / f"{args.experiment}_latent_plate_well_lookup.tsv"
        safe_to_csv(df=plate_well_df,
                    path=plate_well_file,
                    sep="\t",
                    logger=logger,)
        logger.info("Saved Plate/Well metadata to: %s", plate_well_file)
    else:
        logger.warning("Plate_Metadata or Well_Metadata missing in decoded output — skipping plate/well export.")

    # Optional annotation merge
    if args.annotations:
        logger.info("Merging annotations from: %s", args.annotations)
        annot_merge_df = decoded_df.copy()
        # Reconstruct Plate/Well if needed from combined_df (kept in-memory as df_encoded index)
        merge_annotations(
            latent_df_or_path=annot_merge_df,
            annotation_file=args.annotations,
            output_prefix=str(post_clipn_dir / args.experiment),
            logger=logger,
        )

    # Label encoder mappings (TSV)
    try:
        mapping_dir = Path(args.out)
        mapping_dir.mkdir(parents=True, exist_ok=True)
        for column, encoder in encoders.items():
            mapping_path = mapping_dir / f"label_mapping_{column}.tsv"
            mapping_df = pd.DataFrame({column: encoder.classes_, f"{column}_encoded": range(len(encoder.classes_))})
            safe_to_csv(df=mapping_df,
                        path=mapping_path.with_suffix(".tsv"),
                        sep="\t",
                        logger=logger,)
            logger.info("Saved label mapping for %s to %s", column, mapping_path)
        logger.info("CLIPn integration completed.")
        log_memory_usage(logger, prefix="[Mostly finished] ")
    except Exception as exc:
        logger.warning("Failed to save label encoder mappings: %s", exc)

    logger.info("Columns at this stage, encoded: %s", df_encoded.columns.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIPn Integration.")
    parser.add_argument("--datasets_csv", required=True, help="TSV/CSV with columns: 'dataset', 'path'.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--experiment", required=True, help="Experiment name.")
    parser.add_argument(
        "--scaling_mode",
        choices=["all", "per_plate", "none"],
        default="all",
        help="How to scale features.",
    )
    parser.add_argument(
        "--scaling_method",
        choices=["robust", "standard"],
        default="robust",
        help="Scaler to use.",
    )
    parser.add_argument(
        "--mode",
        choices=["reference_only", "integrate_all"],
        required=True,
        help="CLIPn operation mode.",
    )
    parser.add_argument(
        "--clipn_param",
        type=str,
        default="default",
        help="Optional CLIPn parameter for logging only.",
    )
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent space dimensionality.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--epoch", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--save_model", action="store_true", help="Save trained CLIPn model.")
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path or glob to a saved CLIPn model (.pt).",
    )
    parser.add_argument(
        "--reference_names",
        nargs="+",
        default=["reference1", "reference2"],
        help="Datasets to use for training in reference_only mode.",
    )
    parser.add_argument(
        "--aggregate_method",
        choices=["median", "mean", "min", "max"],
        default="median",
        help="Aggregate image-level latent to compound-level.",
    )
    parser.add_argument(
        "--skip_standardise",
        action="store_true",
        help="Skip standardising numeric columns if already scaled.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Optional annotation TSV to merge using Plate/Well.",
    )
    parser.add_argument("--cpu_threads",
                        type=int,
                        default=1,
                        help="Number of CPU threads to use (default: 1)."
                    )


    parser.add_argument(
        "--knn_k",
        type=int,
        default=10,
        help="Number of neighbours per entity.",
    )
    parser.add_argument(
        "--knn_metric",
        choices=["cosine", "euclidean", "correlation"],
        default="cosine",
        help="Distance metric for k-NN.",
    )
    parser.add_argument(
        "--knn_level",
        choices=["compound", "well", "image"],
        default="compound",
        help="Granularity for k-NN.",
    )
    parser.add_argument(
        "--knn_save_full_matrix",
        action="store_true",
        help="Also save the full pairwise distance matrix (guarded to small n).",
    )
    parser.add_argument(
        "--knn_out_subdir",
        type=str,
        default="post_knn",
        help="Subdirectory name for k-NN outputs inside --out.",
    )
    knn_group = parser.add_mutually_exclusive_group()
    knn_group.add_argument(
                        "--knn_only",
                        action="store_true",
                        help="Run k-NN on the pre-CLIPn feature space and exit early."
                    )
    knn_group.add_argument(
                            "--knn_also",
                            action="store_true",
                            help="Run k-NN baseline first, then continue to CLIPn."
                        )



    main(parser.parse_args())
