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
            "cpd_type": g["cpd_type"].apply(mode_strict) if "cpd_type" in df.columns else None,
            "Library": g["Library"].apply(mode_strict) if "Library" in df.columns else None,
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
            meta["cpd_id"] = g["cpd_id"].apply(mode_strict).reset_index(drop=True)
        if "cpd_type" in df.columns:
            meta["cpd_type"] = g["cpd_type"].apply(mode_strict).reset_index(drop=True)
        if "Library" in df.columns:
            meta["Library"] = g["Library"].apply(mode_strict).reset_index(drop=True)
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

