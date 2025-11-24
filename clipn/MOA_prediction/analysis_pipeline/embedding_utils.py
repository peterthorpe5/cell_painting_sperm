#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared embedding utilities for CLIPn and CellProfiler workflows.

This module provides:
- automatic detection of feature columns
- automatic detection and preservation of metadata columns
- safe TSV loading
- replicate aggregation
- L2 normalisation
- PCA and UMAP projection

All functions use UK English spelling and PEP-8 compliant docstrings.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence
import logging


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def configure_logging(*, level: int = logging.INFO) -> None:
    """
    Configure basic logging for scripts using this module.

    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO).
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
    )


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def load_tsv_safely(*, path: str | Path) -> pd.DataFrame:
    """
    Load a tab-separated file into a DataFrame with safe defaults.

    Parameters
    ----------
    path : str or Path
        Path to the TSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False)


# --------------------------------------------------------------------------- #
# Metadata and feature handling
# --------------------------------------------------------------------------- #

_DEFAULT_METADATA = {
    "cpd_id",
    "cpd_type",
    "Plate_Metadata",
    "Well_Metadata",
    "Library",
    "Sample",
    "Dataset",
    "plate",
    "well",
    "row",
    "col",
}


def detect_metadata_columns(
    *, df: pd.DataFrame, user_metadata: Optional[List[str]] = None
) -> List[str]:
    """
    Identify metadata columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    user_metadata : list of str, optional
        Additional metadata columns to include.

    Returns
    -------
    list of str
        Columns treated as metadata.
    """
    metadata = set(_DEFAULT_METADATA)
    if user_metadata:
        metadata.update(set(user_metadata))
    present = [c for c in df.columns if c in metadata]
    return present


def numeric_feature_columns(
    *, df: pd.DataFrame, metadata_cols: Sequence[str]
) -> List[str]:
    """
    Identify numeric feature columns excluding metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    metadata_cols : sequence of str
        Columns to exclude.

    Returns
    -------
    list of str
        Columns treated as numeric embedding features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in metadata_cols]
    return feats


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def l2_normalise(*, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalise rows of a matrix.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    eps : float
        Numerical stabiliser.

    Returns
    -------
    np.ndarray
        L2-normalised matrix.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate_replicates(
    *,
    df: pd.DataFrame,
    id_col: str,
    feature_cols: Sequence[str],
    method: str = "median",
) -> pd.DataFrame:
    """
    Aggregate replicate wells per compound.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    id_col : str
        Identifier column.
    feature_cols : sequence of str
        Columns to aggregate.
    method : str
        Aggregation method: "median" or "mean".

    Returns
    -------
    pd.DataFrame
        Aggregated table.
    """
    out_rows = []
    for cid, sub in df.groupby(id_col, sort=False):
        X = sub[feature_cols].to_numpy()
        if method == "median":
            vec = np.median(X, axis=0)
        else:
            vec = np.mean(X, axis=0)

        row = {id_col: cid}
        for col, v in zip(feature_cols, vec):
            row[col] = v

        out_rows.append(row)

    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #

def project_pca(*, X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    PCA projection to 2D or specified dimensions.

    Parameters
    ----------
    X : np.ndarray
        Matrix to project.
    n_components : int
        Number of components (default: 2).

    Returns
    -------
    np.ndarray
        Projected coordinates.
    """
    from sklearn.decomposition import PCA
    model = PCA(n_components=n_components)
    return model.fit_transform(X)


def project_umap(
    *,
    X: np.ndarray,
    n_components: int = 2,
    n_neighbours: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    """
    UMAP projection.

    Parameters
    ----------
    X : np.ndarray
        Matrix to project.
    n_components : int
        Target dimensionality.
    n_neighbours : int
        Number of neighbours in UMAP graph.
    min_dist : float
        Minimum distance between points.
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Projected coordinates.
    """
    try:
        import umap
    except Exception as exc:
        raise ImportError("UMAP is required for this projection.") from exc

    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbours,
        min_dist=min_dist,
        random_state=random_state,
    )
    return model.fit_transform(X)
