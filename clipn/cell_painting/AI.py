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
