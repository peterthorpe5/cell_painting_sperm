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


def mode_strict(series: pd.Series) -> Optional[str]:
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
