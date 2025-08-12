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
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.serialization
from clipn.model import CLIPn
from sklearn import set_config
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

random.seed(0); _np.random.seed(0); _torch.manual_seed(0)
if _torch.cuda.is_available(): _torch.cuda.manual_seed_all(0)

from cell_painting.process_data import (
    prepare_data_for_clipn_from_df,
    run_clipn_simple,
    standardise_metadata_columns,
)

# Global timer (for memory log timestamps)
_SCRIPT_START_TIME = time.time()

# Make sklearn return DataFrames
set_config(transform_output="pandas")

# Ensure CLIPn is safe to unpickle
torch.serialization.add_safe_globals([CLIPn])


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


def detect_csv_delimiter(csv_path: str) -> str:
    """
    Detect the delimiter of a small text file (prefer tab if ambiguous).

    Parameters
    ----------
    csv_path : str
        Path to the text file.

    Returns
    -------
    str
        Detected delimiter, one of: '\\t' or ','.
    """
    with open(csv_path, "r", newline="") as handle:
        sample = handle.read(4096)
    has_tab = "\t" in sample
    has_comma = "," in sample
    if has_tab and has_comma:
        return "\t"  # prefer TSV
    if has_tab:
        return "\t"
    if has_comma:
        return ","
    # Default to TSV
    return "\t"


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
    df = pd.read_csv(filepath_or_buffer=path, delimiter=delimiter, index_col=None)

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
    df : pd.DataFrame
        DataFrame whose columns may be label-encoded.
    encoders : dict[str, LabelEncoder]
        Mapping of column names to fitted LabelEncoder objects.
    logger : logging.Logger
        Logger for progress and warnings.

    Returns
    -------
    pd.DataFrame
        DataFrame with decoded columns where possible.
    """
    for col, le in encoders.items():
        if col not in df.columns:
            logger.warning("decode_labels: Column '%s' not found in DataFrame. Skipping.", col)
            continue

        if df[col].isna().all():
            logger.warning("decode_labels: Column '%s' is all-NaN. Skipping decode.", col)
            continue

        if not np.issubdtype(df[col].dtype, np.integer):
            logger.info("decode_labels: Column '%s' is not integer-encoded. Skipping decode.", col)
            continue

        try:
            mask_notna = df[col].notna()
            decoded_vals = df[col].copy()
            decoded_vals.loc[mask_notna] = le.inverse_transform(df.loc[mask_notna, col].astype(int))
            df[col] = decoded_vals
            logger.info("decode_labels: Decoded column '%s'.", col)
        except Exception as exc:
            logger.warning(
                "decode_labels: Could not decode column '%s': %s. "
                "May be due to unseen labels, type errors, or missing encoder classes.",
                col, exc,
            )
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

        # Read annotations as TSV (consistent with project policy)
        annot_df = pd.read_csv(filepath_or_buffer=annotation_file, sep="\t")

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
    logger.info("Starting CLIPn integration pipeline")

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
    if not feature_cols:
        raise ValueError("No numeric feature columns found after harmonisation. Check feature overlap and dtypes.")

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

    # Encode labels (cpd_id is explicitly not encoded)
    logger.info("Encoding categorical labels for CLIPn compatibility")
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
            model = torch.load(f=model_path, weights_only=False)

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
        (training_output_path / "training_only_latent.tsv").write_text("", encoding="utf-8")  # ensure dir exists

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
            for new_key, name in zip(new_keys, query_names, strict=True):
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
            model = torch.load(f=model_path, weights_only=False)

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

    # Optional compound-level aggregation
    if getattr(args, "aggregate_method", None):
        latent_cols = [
            col for col in decoded_df.columns
            if isinstance(col, int) or (isinstance(col, str) and col.isdigit())
        ]
        if not latent_cols:
            logger.error("No latent feature columns found for aggregation. Check column names.")
            raise ValueError("No latent feature columns found for aggregation.")
        df_compound = (
            decoded_df.groupby("cpd_id", as_index=False)[latent_cols]
            .agg(args.aggregate_method)
        )
        for col in ["cpd_type", "Library"]:
            if col in decoded_df.columns:
                first_vals = decoded_df.groupby("cpd_id", as_index=False)[col].first()
                df_compound = pd.merge(left=df_compound, right=first_vals, on="cpd_id", how="left")
        agg_path = post_clipn_dir / f"{args.experiment}_CLIPn_latent_aggregated_{args.aggregate_method}.tsv"
        df_compound.to_csv(path_or_buf=agg_path, sep="\t", index=False)
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

    main(parser.parse_args())
