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
    --impute            : Missing data imputation method: 'none' (default) or 'knn'
    --impute_k          : Number of neighbours for KNN imputation (default: 50).
    --no_plot_loss      : If set, disable plotting and saving the training loss curve and TSV.

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
from cell_painting.process_data import (
    prepare_data_for_clipn_from_df,
    run_clipn_simple,
    standardise_metadata_columns,
    assert_cpd_type_encoded,
    assert_xy_alignment_strict,
    validate_frozen_features_manifest)

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


def save_training_loss(
    *,
    loss_values: Sequence[float] | np.ndarray | "torch.Tensor",
    out_dir: str | Path,
    experiment: str,
    mode: str,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """
    Plot and save CLIPn training loss per epoch.

    Parameters
    ----------
    loss_values : Sequence[float] | numpy.ndarray | torch.Tensor
        The per-epoch loss values returned by the CLIPn trainer.
    out_dir : str | Path
        Base output directory (will write into the 'post_clipn' subfolder).
    experiment : str
        Experiment name used for file naming.
    mode : str
        CLIPn mode context ('reference_only' or 'integrate_all'), used in names.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        (tsv_path, png_path) of the written loss artefacts.

    Notes
    -----
    - Writes a tab-separated file with columns 'epoch' and 'loss'.
    - Saves a PNG line plot using the non-interactive 'Agg' backend
      for safe use on HPC/headless systems.
    """
    # Normalise to a 1-D list of floats
    if hasattr(loss_values, "detach"):  # torch.Tensor
        vals = loss_values.detach().cpu().numpy().tolist()
    elif isinstance(loss_values, np.ndarray):
        vals = loss_values.tolist()
    else:
        vals = list(loss_values)

    # Guard: empty or scalar
    if not isinstance(vals, list) or len(vals) == 0:
        logger.warning("Training loss sequence is empty; nothing to plot/save.")
        post_dir = Path(out_dir) / "post_clipn"
        post_dir.mkdir(parents=True, exist_ok=True)
        empty_tsv = post_dir / f"{experiment}_{mode}_clipn_training_loss.tsv"
        pd.DataFrame(data=[], columns=["epoch", "loss"]).to_csv(empty_tsv, sep="\t", index=False)
        return empty_tsv, post_dir / f"{experiment}_{mode}_clipn_training_loss.png"

    epochs = list(range(1, len(vals) + 1))
    df_loss = pd.DataFrame(data={"epoch": epochs, "loss": vals})

    post_dir = Path(out_dir) / "post_clipn"
    post_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = post_dir / f"{experiment}_{mode}_clipn_training_loss.tsv"
    df_loss.to_csv(tsv_path, sep="\t", index=False)

    png_path = post_dir / f"{experiment}_{mode}_clipn_training_loss.pdf"
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, vals)
    ax.set_title(f"CLIPn training loss — {experiment} [{mode}]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    logger.info("Saved training loss TSV -> %s", tsv_path)
    logger.info("Saved training loss pdf -> %s", png_path)
    return tsv_path, png_path


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


def build_knn_index(
    *,
    X: pd.DataFrame,
    k: int,
    metric: str = "cosine",
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a k-NN index and return neighbour indices and distances.

    Parameters
    ----------
    X : pandas.DataFrame
        Latent matrix with numeric columns.
    k : int
        Number of neighbours to return (excluding self).
    metric : str
        'cosine' or 'euclidean'.
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (indices, distances) arrays of shape (n, k).
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("Not enough rows to build a k-NN graph.")
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(X.values)
    dists, idxs = nn.kneighbors(X.values, return_distance=True)
    idxs_clean = []
    dists_clean = []
    for i in range(n):
        row = [(j, d) for j, d in zip(idxs[i], dists[i]) if j != i]
        row = row[:k]
        idxs_clean.append([j for j, _ in row])
        dists_clean.append([float(d) for _, d in row])
    return np.asarray(idxs_clean, dtype=int), np.asarray(dists_clean, dtype=float)


def precision_at_k(
    *,
    labels: pd.Series,
    nn_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute Precision@k for a single label vector using a fixed neighbour index.

    Parameters
    ----------
    labels : pandas.Series
        Class labels aligned to the rows of the k-NN index.
    nn_indices : numpy.ndarray
        Neighbour indices of shape (n, k).

    Returns
    -------
    numpy.ndarray
        Precision@k values for k=1..K (averaged over queries).
    """
    y = labels.astype(str).to_numpy()
    n, k = nn_indices.shape
    hits_cum = np.zeros(shape=(k,), dtype=float)
    for i in range(n):
        neigh = nn_indices[i]
        same = (y[neigh] == y[i]).astype(float)
        hits_cum += np.cumsum(same) / np.arange(1, k + 1)
    return hits_cum / n


def plot_and_save_precision_curves(
    *,
    out_dir: Path,
    experiment: str,
    mode: str,
    label_name: str,
    prec_curve: np.ndarray,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """
    Save Precision@k TSV and PDF for a given label type.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        (tsv_path, pdf_path).
    """
    k_vals = np.arange(1, len(prec_curve) + 1)
    df = pd.DataFrame({"k": k_vals, "precision": prec_curve})
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"{experiment}_{mode}_precision_at_k_{label_name}.tsv"
    df.to_csv(tsv, sep="\t", index=False)

    pdf = out_dir / f"{experiment}_{mode}_precision_at_k_{label_name}.pdf"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    df.plot(kind="line", x="k", y="precision", ax=ax, legend=False)
    ax.set_xlabel("k")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision@k in latent space — {label_name}")
    ax.grid(visible=True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fname=pdf)
    plt.close(fig)

    logger.info("Saved Precision@k for %s -> %s ; %s", label_name, tsv, pdf)
    return tsv, pdf


def dataset_mixing_entropy(
    *,
    datasets: pd.Series,
    nn_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute neighbour-label entropy for each row using Dataset labels.

    Parameters
    ----------
    datasets : pandas.Series
        Dataset label per row.
    nn_indices : numpy.ndarray
        Neighbour indices of shape (n, k).

    Returns
    -------
    numpy.ndarray
        Entropy values per row (natural log base).
    """
    labs = datasets.astype(str).to_numpy()
    n, _ = nn_indices.shape
    ent = np.zeros(shape=(n,), dtype=float)
    for i in range(n):
        neigh = labs[nn_indices[i]]
        _, counts = np.unique(neigh, return_counts=True)
        probs = counts / counts.sum()
        ent[i] = -np.sum(probs * np.log(probs + 1e-12))
    return ent


def plot_and_save_entropy(
    *,
    ent: np.ndarray,
    out_dir: Path,
    experiment: str,
    mode: str,
    num_datasets: int,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """
    Save dataset-mixing entropy histogram and per-row TSV.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        (tsv_path, pdf_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    norm = ent / math.log(max(2, num_datasets))
    tsv = out_dir / f"{experiment}_{mode}_dataset_mixing_entropy.tsv"
    pd.DataFrame({"entropy": ent, "normalised_entropy": norm}).to_csv(tsv, sep="\t", index=False)

    pdf = out_dir / f"{experiment}_{mode}_dataset_mixing_entropy.pdf"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    pd.Series(norm, name="normalised_entropy").plot(kind="hist", bins=40, ax=ax)
    ax.set_xlabel("Normalised entropy (0..1)")
    ax.set_ylabel("Count")
    ax.set_title("Dataset-mixing entropy (higher = better mixing)")
    fig.tight_layout()
    fig.savefig(fname=pdf)
    plt.close(fig)

    logger.info("Saved mixing entropy -> %s ; %s", tsv, pdf)
    return tsv, pdf


def compute_and_save_silhouette(
    *,
    X: pd.DataFrame,
    labels: pd.Series,
    metric: str,
    out_dir: Path,
    experiment: str,
    mode: str,
    label_name: str,
    logger: logging.Logger,
) -> Path:
    """
    Compute a global silhouette score and write to TSV.

    Parameters
    ----------
    X : pandas.DataFrame
        Latent matrix (numeric).
    labels : pandas.Series
        Labels for silhouette computation.
    metric : str
        'cosine' or 'euclidean'.
    out_dir : pathlib.Path
        Output directory.
    experiment : str
        Experiment name.
    mode : str
        Mode name.
    label_name : str
        Name of the grouping label for the report.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pathlib.Path
        Path to the written TSV.
    """
    valid = labels.fillna("NA").astype(str)
    ok_sizes = (valid.value_counts() > 1).all() and valid.nunique() >= 2
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"{experiment}_{mode}_silhouette_{label_name}.tsv"
    if not ok_sizes:
        pd.DataFrame({"label": [label_name], "silhouette_score": [np.nan], "metric": [metric]}).to_csv(tsv, sep="\t", index=False)
        logger.warning("Silhouette skipped for %s (insufficient class sizes).", label_name)
        return tsv
    score = silhouette_score(X=X.values, labels=valid.to_numpy(), metric=metric)
    pd.DataFrame({"label": [label_name], "silhouette_score": [float(score)], "metric": [metric]}).to_csv(tsv, sep="\t", index=False)
    logger.info("Silhouette (%s): %.4f -> %s", label_name, score, tsv)
    return tsv


def save_latent_variance_report(
    *,
    X: pd.DataFrame,
    out_dir: Path,
    experiment: str,
    mode: str,
    eps: float = 1e-6,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """
    Save per-dimension variance (TSV) and a bar plot (PDF); flag low-variance dims.

    Parameters
    ----------
    X : pandas.DataFrame
        Latent matrix with columns as latent dimensions.
    out_dir : pathlib.Path
        Output directory.
    experiment : str
        Experiment name.
    mode : str
        Mode name.
    eps : float
        Threshold below which a dimension is considered 'dead'.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        (tsv_path, pdf_path).
    """
    var = X.var(axis=0, ddof=1).to_frame(name="variance").sort_values(by="variance", ascending=False)
    var["is_dead_dim"] = var["variance"] < eps

    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"{experiment}_{mode}_latent_variance.tsv"
    var.reset_index(names=["dimension"]).to_csv(tsv, sep="\t", index=False)

    pdf = out_dir / f"{experiment}_{mode}_latent_variance.pdf"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    var.reset_index(drop=True)["variance"].plot(kind="bar", ax=ax)
    ax.set_xlabel("Latent dimension (sorted)")
    ax.set_ylabel("Variance")
    ax.set_title("Latent per-dimension variance (dead dims near zero)")
    fig.tight_layout()
    fig.savefig(fname=pdf)
    plt.close(fig)

    logger.info("Saved latent variance report -> %s ; %s (dead dims: %d)", tsv, pdf, int(var["is_dead_dim"].sum()))
    return tsv, pdf


def wbd_ratio_per_compound(
    *,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    k: int,
    metric: str,
    out_dir: Path,
    experiment: str,
    mode: str,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """
    Compute within/between dispersion ratio (WBDR) per compound and save outputs.

    For each compound with ≥ 2 members:
      - Within: mean distance to k nearest neighbours from the same compound.
      - Between: mean distance to k nearest neighbours from other compounds.
      - Ratio = within / between. Lower is better (< 1 desirable).

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        (tsv_path, pdf_path).
    """
    if "cpd_id" not in meta.columns:
        logger.warning("WBDR skipped: 'cpd_id' not in metadata.")
        dummy_tsv = out_dir / f"{experiment}_{mode}_wbd_ratio.tsv"
        pd.DataFrame(columns=["cpd_id", "wbd_ratio"]).to_csv(dummy_tsv, sep="\t", index=False)
        return dummy_tsv, out_dir / f"{experiment}_{mode}_wbd_ratio.pdf"

    idxs, dists = build_knn_index(X=X, k=max(50, k), metric=metric, logger=logger)
    y = meta["cpd_id"].astype(str).to_numpy()
    ratios = []
    for i in range(X.shape[0]):
        neigh = idxs[i]
        same_mask = (y[neigh] == y[i])
        diff_mask = ~same_mask
        within = float(np.mean(dists[i][same_mask][:k])) if same_mask.any() else np.nan
        between = float(np.mean(dists[i][diff_mask][:k])) if diff_mask.any() else np.nan
        r = np.nan if (np.isnan(within) or np.isnan(between) or between == 0.0) else (within / between)
        ratios.append(r)

    df = pd.DataFrame({"cpd_id": meta["cpd_id"].astype(str), "wbd_ratio": ratios}).dropna()
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"{experiment}_{mode}_wbd_ratio.tsv"
    df.to_csv(tsv, sep="\t", index=False)

    pdf = out_dir / f"{experiment}_{mode}_wbd_ratio.pdf"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    df.plot(kind="hist", y="wbd_ratio", bins=40, ax=ax, legend=False)
    ax.set_xlabel("Within/Between ratio (lower is better)")
    ax.set_ylabel("Count")
    ax.set_title("Compound WBDR in latent space")
    fig.tight_layout()
    fig.savefig(fname=pdf)
    plt.close(fig)

    logger.info("Saved WBDR -> %s ; %s (n=%d)", tsv, pdf, df.shape[0])
    return tsv, pdf


def run_training_diagnostics(
    *,
    decoded_df: pd.DataFrame,
    out_dir: Path,
    experiment: str,
    mode: str,
    level: str = "compound",
    k_nn: int = 15,
    metric: str = "cosine",
    logger: logging.Logger,
) -> None:
    """
    Run post-training diagnostics on the latent space and save TSV/PDF outputs.

    Parameters
    ----------
    decoded_df : pandas.DataFrame
        Decoded latent table (must include latent dims and 'Dataset';
        ideally also 'cpd_id' and 'cpd_type').
    out_dir : pathlib.Path
        Base output directory.
    experiment : str
        Experiment name for file naming.
    mode : str
        'reference_only' or 'integrate_all'.
    level : str
        'compound' (default) or 'image'.
    k_nn : int
        Neighbourhood size for diagnostics.
    metric : str
        'cosine' (default) or 'euclidean'.
    logger : logging.Logger
        Logger instance.
    """
    diag_dir = Path(out_dir) / "training_diagnostics"
    X, meta, _ = extract_latent_and_meta(
        decoded_df=decoded_df,
        level=level,
        aggregate="median",
        logger=logger,
    )
    nn_idx, _ = build_knn_index(X=X, k=k_nn, metric=metric, logger=logger)

    if "cpd_id" in meta.columns:
        p_curve = precision_at_k(labels=meta["cpd_id"], nn_indices=nn_idx)
        plot_and_save_precision_curves(
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="cpd_id", prec_curve=p_curve, logger=logger,
        )
    if "cpd_type" in meta.columns:
        p_curve = precision_at_k(labels=meta["cpd_type"], nn_indices=nn_idx)
        plot_and_save_precision_curves(
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="cpd_type", prec_curve=p_curve, logger=logger,
        )

    if "Dataset" in meta.columns and meta["Dataset"].notna().any():
        ent = dataset_mixing_entropy(datasets=meta["Dataset"], nn_indices=nn_idx)
        plot_and_save_entropy(
            ent=ent, out_dir=diag_dir, experiment=experiment, mode=mode,
            num_datasets=int(meta["Dataset"].nunique(dropna=True)), logger=logger,
        )

    if "cpd_type" in meta.columns:
        compute_and_save_silhouette(
            X=X, labels=meta["cpd_type"], metric=metric,
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="cpd_type", logger=logger,
        )
    if "Dataset" in meta.columns:
        compute_and_save_silhouette(
            X=X, labels=meta["Dataset"], metric=metric,
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="Dataset", logger=logger,
        )

    save_latent_variance_report(
        X=X, out_dir=diag_dir, experiment=experiment, mode=mode, eps=1e-6, logger=logger,
    )

    if "cpd_id" in meta.columns:
        wbd_ratio_per_compound(
            X=X, meta=meta, k=k_nn, metric=metric,
            out_dir=diag_dir, experiment=experiment, mode=mode, logger=logger,
        )

    logger.info("Training diagnostics completed. Outputs in %s", diag_dir)


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
TECHNICAL_FEATURE_BLOCKLIST = {"ImageNumber","Number_Object_Number","ObjectNumber","TableNumber"}

# Columns that are metadata (never model inputs), case-insensitive match by name
METADATA_COL_BLOCKLIST = {
    "cpd_id", "cpd_type", "dataset", "sample", "replicate", "library",
    "plate_metadata", "well_metadata", "plate", "well",
    "chemid", "compound_id", "concentration", "dose", "timepoint", "batch", "site"
}


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
    k: int = 50,
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
    delim_name = "tab" if delimiter == "\t" else "comma"
    logger.info("[%s] Reading %s (delimiter=%s)", name, path, delim_name)
    df = _read_csv_fast(path, delimiter)

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
    common_cols = sorted(_exclude_technical_features(inter_before_block, logger))
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


def encode_labels(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode label columns needed for downstream evaluation, without leaking
    metadata into features. Only 'cpd_type' is encoded; 'cpd_id', 'Library',
    'Plate_Metadata', 'Well_Metadata', and 'Dataset' remain as-is.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with metadata and features.
    logger : logging.Logger
        Logger for progress messages.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        (df_encoded, encoders)
    """
    encoders: Dict[str, LabelEncoder] = {}
    out = df.copy()

    if "cpd_type" in out.columns:
        out.loc[:, "cpd_type"] = out["cpd_type"].astype("string")
        le = LabelEncoder()
        mask = out["cpd_type"].notna()
        if mask.any():
            out.loc[mask, "cpd_type"] = le.fit_transform(out.loc[mask, "cpd_type"])
            encoders["cpd_type"] = le
            logger.info("encode_labels: Encoded 'cpd_type' with %d classes.", len(le.classes_))
        else:
            logger.warning("encode_labels: 'cpd_type' present but all values are NA; left as string.")
    return out, encoders


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
    if torch.cuda.is_available():
        logger.info("GPU detected; set CPU-thread env vars for BLAS/pandas, "
                    "but PyTorch compute will mainly run on GPU.")
    else:
        logger.info("CPU backend: using %d threads.", n)


    logger.info("CPU threads set to %d (from --cpu_threads)", n)
    logger.info("only change threads if you are using Torch cpu backend")
    return n



def select_clipn_features_and_write(
    df: pd.DataFrame,
    out_dir: str | Path,
    experiment: str,
    logger: logging.Logger,
    features_expected: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select the final CLIPn feature set, exclude all metadata (incl. 'cpd_type'),
    enforce column order, and write QC files.

    Rules
    -----
    - Keep only numeric columns.
    - Exclude any column whose lower-cased name is in METADATA_COL_BLOCKLIST.
    - Always exclude 'cpd_type' even if encoded as integers.
    - If `features_expected` is provided (projection), reindex to that exact list
      and raise if any are missing.
    - Write:
        <out_dir>/<experiment>_features_used.tsv   (ordered list of features)
        <out_dir>/<experiment>_columns_full.tsv    (all columns + dtype + is_feature)

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing features + metadata.
    out_dir : str | pathlib.Path
        Output directory for QC files.
    experiment : str
        Name for output files.
    logger : logging.Logger
        Logger.
    features_expected : list[str] | None
        Known-good ordered feature list to enforce (for projection).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (df_features_only, feature_cols)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start with numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Case-insensitive metadata exclusion (incl. cpd_type)
    meta_lower = {m.lower() for m in METADATA_COL_BLOCKLIST}
    def is_meta(col: str) -> bool:
        return col.lower() in meta_lower

    feature_cols = [c for c in numeric_cols if not is_meta(c) and c != "cpd_type"]

    if features_expected is not None:
        missing = [c for c in features_expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} expected feature(s): {missing[:10]} ..."
            )
        feature_cols = list(features_expected)

    if not feature_cols:
        raise ValueError("No usable numeric feature columns after excluding metadata.")

    df_features = df.loc[:, feature_cols].copy()

    # QC manifests
    feats_path = out_dir / f"{experiment}_features_used.tsv"
    cols_path = out_dir / f"{experiment}_columns_full.tsv"

    feats_df = pd.DataFrame(
        {"feature": feature_cols, "dtype": [str(df[c].dtype) for c in feature_cols]}
    )
    safe_to_csv(df=feats_df, path=feats_path, sep="\t", logger=logger)

    cols_df = pd.DataFrame(
        {
            "column": list(df.columns),
            "dtype": [str(df[c].dtype) for c in df.columns],
            "is_feature": [c in feature_cols for c in df.columns],
        }
    )
    safe_to_csv(df=cols_df, path=cols_path, sep="\t", logger=logger)

    logger.info("Feature freeze complete: %d features", len(feature_cols))
    return df_features, feature_cols




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
    skip_standardise: bool = False,
    plot_loss: bool = True,
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

    logger.info("Combined DataFrame shape: %s", df.shape)
    logger.debug("Head of combined DataFrame:\n%s", df.head())

    # Central feature freeze (training path): write QC + remove all metadata
    df_features, feature_cols = select_clipn_features_and_write(
        df=df,
        out_dir=output_path,
        experiment=experiment,
        logger=logger,
        features_expected=None,
    )

    # Get dataset keys + labels (y) from the FULL df (which still has 'cpd_type')
    _, label_dict, label_mappings, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df)

    # Build data_dict (X) from the frozen feature table to ensure no metadata leak
    data_dict: Dict[int, np.ndarray] = {}
    for key, name in dataset_key_mapping.items():
        # keep row order identical to df / label_dict by slicing with the MultiIndex
        X_block = df_features.loc[name].droplevel("Dataset")
        data_dict[key] = X_block.values

    # many sanity checks!!
    # 1) cpd_type must be integer-encoded in the full df used for labels
    assert_cpd_type_encoded(df=df, logger=logger)

    # 2) frozen feature file must not contain metadata or 'cpd_type'
    feature_list_path = Path(output_path) / f"{experiment}_features_used.tsv"
    _ = validate_frozen_features_manifest(feature_list_path=feature_list_path, logger=logger)

    # 3) lengths of X and y must match for every dataset id
    assert_xy_alignment_strict(X=data_dict, y=label_dict, logger=logger)

    logger.info(
        "Prepared data for CLIPn: %d datasets, feature dim=%d, labels per dataset: %s",
        len(data_dict), len(feature_cols), {k: len(v) for k, v in label_dict.items()}
    )

    latent_dict, model, loss = run_clipn_simple(
        data_dict,
        label_dict,
        latent_dim=latent_dim,
        lr=lr,
        epochs=epochs,
    )

    logger.info("CLIPn training completed.")
    # Save loss curve (TSV + PNG) if available
    if plot_loss and loss is not None:
        try:
            save_training_loss(
                loss_values=loss,
                out_dir=output_path,
                experiment=experiment,
                mode=mode,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Failed to save/plot training loss: %s", exc)


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

    plot_loss = not args.no_plot_loss


    # Load + harmonise - lots of logging here
    dataframes, common_cols = load_and_harmonise_datasets(
        datasets_csv=args.datasets_csv,
        logger=logger,
        mode=args.mode,
        audit_dir=Path(args.out) / "feature_audit",
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


    # Optional clean + impute (pre-scaling)
    if args.impute == "median":
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

    elif args.impute == "knn":
        combined_df, dropped_feats = clean_and_impute_features_knn(
            df=combined_df,
            feature_cols=feature_cols,
            logger=logger,
            groupby_cols=["Dataset", "Plate_Metadata"] if "Plate_Metadata" in combined_df.columns else ["Dataset"],
            max_nan_col_frac=0.30,
            max_nan_row_frac=0.80,
            n_neighbors=args.impute_knn_k,
        )
        if dropped_feats:
            feature_cols = [c for c in feature_cols if c not in dropped_feats]

    else:
        logger.info("Imputation disabled (--impute none); skipping NaN drop/impute step.")
        dropped_feats = []



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

    # Final guard: never let technical counters into modelling
    df_scaled_all = df_scaled_all.drop(columns=[c for c in TECHNICAL_FEATURE_BLOCKLIST if c in df_scaled_all.columns], errors="ignore")
    

    feature_audit_dir = Path(args.out) / "feature_audit"
    feature_audit_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(feature_cols, name="feature").to_csv(
        feature_audit_dir / "features_used_after_imputation.tsv", sep="\t", index=False
    )
    logger.info("Final feature column count after cleaning: %d", len(feature_cols))

    # If imputation is off, fail fast if NaNs remain in features
    if args.impute == "none" and df_scaled_all[feature_cols].isna().any().any():
        n_missing = int(df_scaled_all[feature_cols].isna().sum().sum())
        raise ValueError(
            f"{n_missing} missing values remain in feature columns with --impute none. "
            "Enable imputation (--impute median|knn) or pre-clean inputs."
        )

    # ===== Optional: k-NN baseline on the pre-CLIPn feature space =====
    if args.knn_only or args.knn_also:
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
            # Prepare and predict training references (freeze features; re-use expected list if present)
            feature_list_path = Path(args.out) / f"{args.experiment}_features_used.tsv"
            if feature_list_path.exists():
                features_expected = pd.read_csv(feature_list_path, sep="\t")["feature"].tolist()
            else:
                features_expected = None  # will freeze now
            features_expected = validate_frozen_features_manifest(
                feature_list_path=feature_list_path, logger=logger
            )

            ref_features, _ = select_clipn_features_and_write(
                df=reference_df,
                out_dir=args.out,
                experiment=args.experiment,
                logger=logger,
                features_expected=features_expected,
            )
            data_dict, _, _, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(ref_features)


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
                plot_loss=plot_loss, 
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

            # Load frozen training feature order
            feature_list_path = Path(args.out) / f"{args.experiment}_features_used.tsv"
            features_expected = validate_frozen_features_manifest(
                                    feature_list_path=feature_list_path, logger=logger
                                )


            meta_drop = [c for c in query_df.columns if c.lower() in {m.lower() for m in METADATA_COL_BLOCKLIST}]

            query_data_dict_corrected: Dict[int, np.ndarray] = {}
            for name, group in query_groups:
                if name not in dataset_key_mapping_inv:
                    continue
                X = group.droplevel("Dataset").drop(columns=meta_drop, errors="ignore")
                missing = [c for c in features_expected if c not in X.columns]
                if missing:
                    raise ValueError(f"[{name}] Missing {len(missing)} expected feature(s): {missing[:10]} ...")
                X = X.reindex(columns=features_expected)
                query_data_dict_corrected[dataset_key_mapping_inv[name]] = X.values

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


            # Build data_dict (X) from frozen features; labels not needed for prediction
            # Prepare and predict latent with loaded model (manifest-enforced features)
            feature_list_path = Path(args.out) / f"{args.experiment}_features_used.tsv"

            features_expected = validate_frozen_features_manifest(
                feature_list_path=feature_list_path, logger=logger
            )
            df_features, _ = select_clipn_features_and_write(
                df=df_encoded,
                out_dir=args.out,
                experiment=args.experiment,
                logger=logger,
                features_expected=features_expected,
            )

            # Build data_dict using the frozen features but keep the dataset key mapping from the full df
            _, _, _, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df_encoded)
            data_dict = {
                k: df_features.loc[name].droplevel("Dataset").values
                for k, name in dataset_key_mapping.items()
            }

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
                plot_loss=plot_loss,
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
    if not args.no_diagnostics:
        run_training_diagnostics(
                decoded_df=decoded_df,
                out_dir=Path(args.out),
                experiment=args.experiment,
                mode=args.mode,
                level=args.diag_level,
                k_nn=args.diag_k,
                metric=args.diag_metric,
                logger=logger,
            )
        logger.info("Diagnostics completed.")


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
    parser.add_argument(
    "--impute",
    choices=["median", "knn", "none"],
    default="none",
    help="Impute missing values before scaling/modeling. "
         "'median' = per-group median , 'knn' = KNNImputer, "
         "'none' = skip imputation (default)."
    )
    parser.add_argument(
        "--impute_knn_k",
        type=int,
        default=50,
        help="Number of neighbours for KNN imputation (used when --impute knn)."
    )
    parser.add_argument(
                        "--no_plot_loss",
                        action="store_true",
                        help="Disable plotting and saving the training loss curve and TSV.",
                    )

    parser.add_argument(
        "--no_diagnostics",
        action="store_true",
        help="Disable post-training diagnostics (Precision@k, mixing entropy, silhouette, variance, WBDR).",
    )
    parser.add_argument(
        "--diag_level",
        choices=["compound", "image"],
        default="compound",
        help="Granularity for diagnostics ('compound' aggregates by cpd_id).",
    )
    parser.add_argument(
        "--diag_k",
        type=int,
        default=15,
        help="Neighbourhood size k for diagnostics.",
    )
    parser.add_argument(
        "--diag_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for diagnostics.",
    )


    main(parser.parse_args())
