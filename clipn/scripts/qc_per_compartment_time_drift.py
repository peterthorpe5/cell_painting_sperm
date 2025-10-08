#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-compartment object-level acquisition-drift QC for CellProfiler outputs.
==========================================================================

Overview
--------
This stand-alone script inspects CellProfiler **object** tables (e.g., Cell,
Cytoplasm, Nuclei) individually to detect acquisition drift/degradation over
the run, *without* collapsing to per-well medians. It uses `ImageNumber` as
a time-like axis and computes for selected features:

- Spearman correlation (rho) vs acquisition order (with p-value).
- Robust slope on per-image medians (binned OLS / optional Huber) with a bootstrap CI.
- Early-vs-late shift via Cliff's delta (effect size).

Outputs are written **per compartment** into subfolders of --out_dir:
- Drift stats TSV (tab-separated; never commas).
- Optional control-only drift stats TSV if --controls_query is given.
- Per-image summary TSV (useful for quick re-plots).
- Scalable plots (hexbin + rolling median) for a small panel of features.

Typical usage
-------------
python qc_per_compartment_time_drift.py \
  --input_dir /path/to/folder \
  --out_dir /path/to/QC \
  --bin_size 150 \
  --max_points_plot 200000 \
  --controls_query 'cpd_type == "control" or Library == "DMSO"'

Notes
-----
- Designed for files like: HepG2CP_Cell.csv.gz, HepG2CP_Cytoplasm.csv.gz,
  HepG2CP_Nuclei.csv.gz. You can broaden patterns with --include_glob.
- Assumes `ImageNumber` exists in object files. If not, use --image_col to
  point to the correct acquisition-order column.
- Plots are limited to a small auto-picked feature panel unless you pass
  --plot_features explicitly.


------
Prepared for object-level QC with per-compartment outputs and UK English.
"""
from __future__ import annotations
import warnings
import argparse
import logging
from pathlib import Path
import sys 
import os
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Optional, Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

# Optional sklearn import (fallback to OLS if unavailable)
try:
    from sklearn.linear_model import HuberRegressor  # type: ignore
except Exception:
    HuberRegressor = None

# ----------------------------- Logging ------------------------------------- #

def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    """
    Configure a logger that writes to console and file.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory where the log file will be written.
    level : str
        Logging level (e.g., "INFO", "DEBUG").

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qc_compartment_drift")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out_dir / "qc_per_compartment_time_drift.log", mode="w")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ----------------------------- Helpers ------------------------------------- #

def list_compartment_files(
    input_dir: Path,
    include_glob: list[str],
) -> dict[str, list[Path]]:
    """
    Find compartment CSVs (gz) and group by inferred compartment.

    Parameters
    ----------
    input_dir : pathlib.Path
        Folder to scan recursively.
    include_glob : list[str]
        Filename patterns to include (e.g., ["*Cell*.csv.gz", "*Cytoplasm*.csv.gz"]).

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Mapping compartment -> list of files.
    """
    mapping: dict[str, list[Path]] = {}
    for pat in include_glob:
        for p in input_dir.rglob(pat):
            comp = infer_compartment_from_name(p.name)
            if comp is None:
                # Skip images/other non-object tables unless user expanded patterns
                continue
            mapping.setdefault(comp, []).append(p)
    # Stable ordering
    for k in mapping:
        mapping[k] = sorted(mapping[k])
    return mapping


def per_image_median_series(
    df: pd.DataFrame,
    feature: str,
    image_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create per-image medians for a feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table.
    feature : str
        Feature column to summarise.
    image_col : str
        Column indicating acquisition order (e.g., 'ImageNumber').

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (x, y) where x are sorted unique image indices (float64),
        and y are corresponding medians (float64), with NaNs dropped.
    """
    sub = df[[image_col, feature]].copy()
    sub[image_col] = ensure_numeric(sub[image_col])
    sub[feature] = ensure_numeric(sub[feature])
    sub = sub.dropna()
    if sub.empty:
        return np.array([]), np.array([])
    med = sub.groupby(image_col, observed=False)[feature].median()
    med = med.dropna()
    if med.empty:
        return np.array([]), np.array([])
    x = med.index.to_numpy(dtype=float)
    y = med.to_numpy(dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def normalise_well_id(well: str) -> Optional[str]:
    """
    Normalise a well ID to 'RowLetter<ColumnInt>' form to match A1..P24 style.

    Examples
    --------
    'A01' -> 'A1', 'a001' -> 'A1', 'H12' -> 'H12'.

    Parameters
    ----------
    well : str
        Raw well identifier.

    Returns
    -------
    Optional[str]
        Normalised well ID or None if unparsable.
    """
    if not isinstance(well, str) or len(well) < 2:
        return None
    row = well[0].upper()
    digits = "".join(ch for ch in well[1:] if ch.isdigit())
    if not digits:
        return None
    try:
        col = int(digits)
    except ValueError:
        return None
    return f"{row}{col}"


def get_controls_subset_by_wells(
    df: pd.DataFrame,
    well_col: str,
    controls_wells: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Subset rows whose well is in a provided list of control wells.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table with a well column.
    well_col : str
        Name of the well column (e.g., 'Well_Metadata').
    controls_wells : list[str]
        List of control wells, e.g., ['A23','B23',...].
    logger : logging.Logger
        Logger.

    Returns
    -------
    pandas.DataFrame
        Subset of df corresponding to control wells (may be empty).
    """
    if well_col not in df.columns:
        logger.warning("Well column '%s' not found; cannot apply controls-by-well filter.", well_col)
        return df.iloc[0:0]

    target = {w for w in (normalise_well_id(w) for w in controls_wells) if w is not None}
    if not target:
        logger.warning("No valid control wells parsed; skipping controls-by-well filter.")
        return df.iloc[0:0]

    wells_norm = df[well_col].astype(str).map(normalise_well_id)
    mask = wells_norm.isin(target)
    out = df.loc[mask].copy()
    logger.info("Controls-by-wells: %d/%d rows retained using %d wells.",
                out.shape[0], df.shape[0], len(target))
    return out

def bin_by_count(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 400,
    min_per_bin: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin (x, y) into ~equal-count bins and take bin medians.

    Parameters
    ----------
    x : numpy.ndarray
        X values (e.g., ImageNumber).
    y : numpy.ndarray
        Y values (per-image medians).
    n_bins : int
        Target number of bins.
    min_per_bin : int
        Minimum observations required to keep a bin.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (xb, yb) arrays of bin midpoints and bin medians.
    """
    n = x.size
    if n == 0:
        return x, y
    k = int(max(5, min(n_bins, n // min_per_bin)))
    qs = np.linspace(0.0, 1.0, k + 1)
    edges = np.quantile(x, qs)
    edges[0] = np.floor(edges[0])
    edges[-1] = np.ceil(edges[-1]) + 1e-9
    xb = []
    yb = []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= min_per_bin:
            xb.append(0.5 * (lo + hi))
            yb.append(np.median(y[mask]))
    if not xb:
        return x, y
    return np.asarray(xb, dtype=float), np.asarray(yb, dtype=float)


def fit_slope(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "ols_binned",
) -> Tuple[float, float, float]:
    """
    Fit a robust slope to (x, y) medians.

    Parameters
    ----------
    x : numpy.ndarray
        X values (binned midpoints or image numbers).
    y : numpy.ndarray
        Y values (binned medians or medians).
    method : {"ols_binned", "huber_binned", "ols"}
        Slope estimator to use.

    Returns
    -------
    tuple[float, float, float]
        (slope, ci_low, ci_high). CI is a light bootstrap over bins (or NaN if too few bins).
    """
    if x.size < 5:
        return (np.nan, np.nan, np.nan)

    # Center x to improve conditioning
    x0 = x - np.median(x)
    X = x0.reshape(-1, 1)

    if method == "huber_binned" and HuberRegressor is not None:
        try:
            hub = HuberRegressor().fit(X, y)
            slope = float(hub.coef_[0])
        except Exception:
            # Fallback to OLS
            slope = float(np.polyfit(x0, y, deg=1)[0])
    elif method in ("ols_binned", "ols"):
        slope = float(np.polyfit(x0, y, deg=1)[0])
    else:
        slope = float(np.polyfit(x0, y, deg=1)[0])

    # Bootstrap CI over bins if we have enough bins
    ci_low = np.nan
    ci_high = np.nan
    if x.size >= 15:
        rng = np.random.default_rng(0)
        B = 300
        slopes = []
        for _ in range(B):
            idx = rng.integers(0, x.size, size=x.size)
            xx = x0[idx]
            yy = y[idx]
            try:
                b = float(np.polyfit(xx, yy, deg=1)[0])
                slopes.append(b)
            except Exception:
                continue
        if slopes:
            ci_low = float(np.quantile(slopes, 0.025))
            ci_high = float(np.quantile(slopes, 0.975))

    return (slope, ci_low, ci_high)


def infer_compartment_from_name(filename: str) -> Optional[str]:
    """
    Guess the compartment name (Cell, Cytoplasm, Nuclei) from a filename.

    Parameters
    ----------
    filename : str
        Basename of the file.

    Returns
    -------
    Optional[str]
        "Cell", "Cytoplasm", or "Nuclei" if detected; otherwise None.
    """
    low = filename.lower()
    if "nuclei" in low:
        return "Nuclei"
    if "cytoplasm" in low:
        return "Cytoplasm"
    if "cell" in low and "image" not in low:
        return "Cell"
    if "mitochondria" in low:
        return "Mitochondria"
    if "acrosome" in low:
        return "Acrosome"
    return None



def read_object_table(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Read a gzipped CSV or TSV, letting pandas infer compression.

    Parameters
    ----------
    path : pathlib.Path
        Input file (e.g., .csv.gz or .tsv.gz).
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with dtypes left mostly as-is.
    """
    # Try comma first (CellProfiler often emits CSV)
    try:
        df = pd.read_csv(path, sep=",", compression="infer", low_memory=False)
        # Heuristic: if only one column due to wrong sep, fallback to TSV
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    except Exception as e:
        logger.warning("CSV read failed for %s (%s); trying TSV.", path, e)
        df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    return df


def ensure_numeric(series: pd.Series) -> pd.Series:
    """
    Coerce a pandas Series to numeric, preserving NaN for invalid entries.

    Parameters
    ----------
    series : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        Numeric series with NaN where coercion failed.
    """
    return pd.to_numeric(series, errors="coerce")


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    pvals : numpy.ndarray
        Array of p-values.
    alpha : float
        Target FDR.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (reject_flags, q_values)
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q_raw = (pvals * n) / ranks
    q_sorted = q_raw[order]
    q_mon = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_mon)
    q[order] = q_mon
    reject = q <= alpha
    return reject, q



def compute_time_trend_smoothed(
    df: pd.DataFrame,
    feature: str,
    image_col: str,
    window: int = 301,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a smoothed time trend for a feature using per-image medians.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table that includes `image_col` and `feature`.
    feature : str
        Feature to model vs acquisition order.
    image_col : str
        Acquisition-order column (e.g., 'ImageNumber').
    window : int
        Window size for centred rolling median smoothing.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Sorted unique image indices (x) and corresponding smoothed medians (y).
    """
    x_img, y_img = per_image_median_series(
        df=df[[image_col, feature]],
        feature=feature,
        image_col=image_col,
    )
    if x_img.size == 0:
        return x_img, y_img
    y_smooth = rolling_median(y=y_img, window=min(window, max(3, x_img.size // 50)))
    return x_img, y_smooth


def compute_plate_residual_map(
    df: pd.DataFrame,
    feature: str,
    image_col: str,
    plate_col: str,
    well_col: str,
    window: int = 301,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Build a per-well residual map after removing the *plate-specific* time trend.

    For each plate:
      1) Compute per-image medians of `feature` and smooth across `image_col`.
      2) Interpolate the smoothed trend at each object's acquisition index.
      3) Residual = feature_value - trend_predicted(ImageNumber).
      4) Aggregate residuals per (plate, well) with the median.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table including `feature`, `image_col`, `plate_col`, `well_col`.
    feature : str
        Feature to process.
    image_col : str
        Acquisition order column.
    plate_col : str
        Plate identifier column.
    well_col : str
        Well identifier column.
    window : int
        Smoothing window for the time trend (centred rolling median).
    logger : logging.Logger, optional
        Logger for diagnostics.

    Returns
    -------
    pandas.DataFrame
        Columns: [plate_col, well_col, 'residual'] with one row per well.
    """
    req = [plate_col, well_col, image_col, feature]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns for residual map: {miss}")

    sub = df[req].dropna().copy()
    if sub.empty:
        return pd.DataFrame(columns=[plate_col, well_col, "residual"])

    sub[image_col] = pd.to_numeric(sub[image_col], errors="coerce")
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub.dropna(subset=[image_col, feature])

    out_blocks: list[pd.DataFrame] = []
    for plate_id, block in sub.groupby(plate_col, observed=False):
        x_tr, y_tr = compute_time_trend_smoothed(
            df=block[[image_col, feature]],
            feature=feature,
            image_col=image_col,
            window=window,
        )
        if x_tr.size == 0 or y_tr.size == 0:
            if logger is not None:
                logger.warning("Residual map: empty trend for plate %s.", plate_id)
            continue

        # Interpolate trend at each object's acquisition index
        xi = block[image_col].to_numpy(dtype=float)
        yi = block[feature].to_numpy(dtype=float)
        # In case of duplicate x_tr, np.interp requires strictly increasing x
        order = np.argsort(x_tr)
        x_tr = x_tr[order]
        y_tr = y_tr[order]
        y_hat = np.interp(x=xi, xp=x_tr, fp=y_tr)  # linear interpolation
        resid = yi - y_hat

        wmed = (
            pd.DataFrame({
                plate_col: block[plate_col].to_numpy(),
                well_col: block[well_col].to_numpy(),
                "residual": resid,
            })
            .groupby([plate_col, well_col], observed=False)["residual"]
            .median()
            .reset_index()
        )
        out_blocks.append(wmed)

    if not out_blocks:
        return pd.DataFrame(columns=[plate_col, well_col, "residual"])

    return pd.concat(out_blocks, axis=0, ignore_index=True)


def per_well_time_quantile(
    df: pd.DataFrame,
    image_col: str,
    plate_col: str,
    well_col: str,
) -> pd.DataFrame:
    """
    Compute per-well acquisition quantile based on the median ImageNumber.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `image_col`, `plate_col`, `well_col`.
    image_col : str
        Acquisition-order column.
    plate_col : str
        Plate identifier.
    well_col : str
        Well identifier.

    Returns
    -------
    pandas.DataFrame
        [plate_col, well_col, 'acq_q'] in [0, 1].
    """
    req = [plate_col, well_col, image_col]
    sub = df[req].dropna()
    if sub.empty:
        return pd.DataFrame(columns=[plate_col, well_col, "acq_q"])
    wmed = (
        sub.groupby([plate_col, well_col], observed=False)[image_col]
        .median()
        .rename("img_median")
        .reset_index()
    )
    wmed["acq_q"] = (
        wmed.groupby(plate_col, observed=False)["img_median"]
        .rank(method="average", pct=True)
        .astype(float)
    )
    return wmed[[plate_col, well_col, "acq_q"]]


def per_well_feature_median_z(
    df: pd.DataFrame,
    feature: str,
    plate_col: str,
    well_col: str,
) -> pd.DataFrame:
    """
    Compute per-well feature median and robust z-score within each plate.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table with `feature`, `plate_col`, `well_col`.
    feature : str
        Feature name to summarise.
    plate_col : str
        Plate identifier.
    well_col : str
        Well identifier.

    Returns
    -------
    pandas.DataFrame
        [plate_col, well_col, 'feat_z'].
    """
    req = [plate_col, well_col, feature]
    sub = df[req].dropna()
    if sub.empty:
        return pd.DataFrame(columns=[plate_col, well_col, "feat_z"])
    wmed = (
        sub.groupby([plate_col, well_col], observed=False)[feature]
        .median()
        .rename("w_med")
        .reset_index()
    )

    def _robust_z(g: pd.DataFrame) -> pd.DataFrame:
        med = g["w_med"].median()
        mad = (g["w_med"] - med).abs().median()
        scale = 1.4826 * mad if mad and mad > 0 else (g["w_med"].std() or 1.0)
        g["feat_z"] = (g["w_med"] - med) / (scale if scale else 1.0)
        return g

    out = wmed.groupby(plate_col, observed=False, group_keys=False).apply(_robust_z)
    return out[[plate_col, well_col, "feat_z"]]



def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta (x vs y) in [-1, 1]; positive => x > y tendency.

    Parameters
    ----------
    x : numpy.ndarray
        Values in group X (e.g., early acquisition).
    y : numpy.ndarray
        Values in group Y (e.g., late acquisition).

    Returns
    -------
    float
        Cliff's delta effect size.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    cx = np.sort(x)
    cy = np.sort(y)
    i = j = more = less = 0
    nx = cx.size
    ny = cy.size
    while i < nx and j < ny:
        if cx[i] > cy[j]:
            more += nx - i
            j += 1
        elif cx[i] < cy[j]:
            less += ny - j
            i += 1
        else:
            v = cx[i]
            tx = ty = 0
            while i < nx and cx[i] == v:
                i += 1
                tx += 1
            while j < ny and cy[j] == v:
                j += 1
                ty += 1
            more += tx * (ny - j)
            less += ty * (nx - i)
    return (more - less) / (nx * ny)


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a centred rolling median.

    Parameters
    ----------
    y : numpy.ndarray
        Input values.
    window : int
        Window size; if <= 1 returns y.

    Returns
    -------
    numpy.ndarray
        Smoothed values.
    """
    if window <= 1 or y.size == 0:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, min_periods=1, center=True).median().to_numpy()


# -------------------------- Feature selection ------------------------------- #
def feature_candidates(header: Iterable[str], compartment: str) -> list[str]:
    """
    Build a conservative default feature panel for an object table.

    Notes
    -----
    CellProfiler object tables typically use names like:
    - Intensity_MeanIntensity_<Channel>
    - Intensity_MedianIntensity_<Channel>
    - Intensity_StdIntensity_<Channel>
    - AreaShape_Area, AreaShape_Compactness, ...
    - Texture_*_<Channel>_<scale>_<angle>_<bins>
    - Granularity_<k>_<Channel>

    They usually do NOT have the 'Mean_<Compartment>_' prefix (that appears in
    image-level summaries). We therefore match on object-style prefixes.

    Parameters
    ----------
    header : Iterable[str]
        Column names in the object table.
    compartment : str
        Compartment name ("Cell", "Cytoplasm", "Nuclei"); not used for filtering
        beyond potential future compartment-specific tweaks.

    Returns
    -------
    list[str]
        A shortlist of informative, object-level columns.
    """
    blacklist_prefixes = (
        "FileName_", "PathName_", "MD5Digest_", "URL_", "Location_CenterMassIntensity_",
        "Group_", "Channel_", "ExecutionTime_", "Image_",  # Image_ are image-level aggregates
    )
    blacklist_exact = {
        "ObjectNumber", "ImageNumber", "TableNumber",
        "Plate", "Plate_Metadata", "Well", "Well_Metadata",
        "Library", "Treatment", "cpd_id", "cpd_type",
        "Column_Metadata", "Field_Metadata", "Row", "Column",
        "ImageName", "ImageId", "ImageSeries",
    }

    wanted_prefixes = (
        "Intensity_MeanIntensity_", "Intensity_MedianIntensity_", "Intensity_StdIntensity_",
        "Intensity_MaxIntensity_", "Intensity_MinIntensity_",
        "AreaShape_", "Texture_", "Granularity_",
        "Neighbors_", "RadialDistribution_",  # optional but sometimes useful
    )

    out: list[str] = []
    for c in header:
        if c in blacklist_exact:
            continue
        if any(c.startswith(bp) for bp in blacklist_prefixes):
            continue
        if any(c.startswith(wp) for wp in wanted_prefixes):
            out.append(c)

    # Keep stable order and cap size to avoid huge default runs
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:200]



# ----------------------------- Core analysis -------------------------------- #
def compute_drift_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    image_col: str,
    early_frac: float = 0.2,
    min_points: int = 2000,
    slope_method: str = "ols_binned",
    n_bins_slope: int = 400,
    min_per_bin: int = 20,
    spearman_mode: str = "images",
    spearman_max_objects: int = 500_000,
) -> pd.DataFrame:
    """
    Compute per-feature drift statistics vs acquisition order at scale.

    Strategy
    --------
    - Build per-image medians to reduce N dramatically.
    - Estimate slope on **binned** medians (OLS or Huber).
    - Compute Spearman on image medians (default) or on objects with a cap.
    - Early/late comparisons use **object-level** values for effect size.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table with at least `image_col` and feature columns.
    feature_cols : list[str]
        Features to evaluate.
    image_col : str
        Column indicating acquisition order (e.g., "ImageNumber").
    early_frac : float
        Fraction for early/late split (e.g., 0.2 => bottom/top 20%).
    min_points : int
        Minimum number of non-NaN object rows required to compute stats.
    slope_method : {"ols_binned", "huber_binned", "ols"}
        Slope estimator used on per-image (optionally binned) medians.
    n_bins_slope : int
        Target number of equal-count bins for slope fitting.
    min_per_bin : int
        Minimum samples required per bin to keep the bin.
    spearman_mode : {"images", "objects"}
        Whether to compute Spearman on per-image medians (default) or on objects.
    spearman_max_objects : int
        If spearman_mode == "objects", a random cap of object rows used for speed.

    Returns
    -------
    pandas.DataFrame
        Rows: features; columns with rho, q, slope, CI, Cliff's delta and counts.
    """
    # Ensure numeric image axis
    img_all = ensure_numeric(df[image_col])
    df = df.loc[img_all.notna()].copy()
    img_all = img_all[img_all.notna()]

    # Early/late thresholds on the image axis
    q_low = img_all.quantile(early_frac)
    q_high = img_all.quantile(1.0 - early_frac)

    records = []
    for feat in feature_cols:
        x_obj = ensure_numeric(df[feat])
        mask = x_obj.notna()
        if mask.sum() < min_points:
            continue

        # Per-image medians (x_img = image numbers, y_img = medians)
        x_img, y_img = per_image_median_series(df.loc[mask, [image_col, feat]], feature=feat, image_col=image_col)
        if x_img.size < 5:
            continue

        # Bin and fit slope on image medians
        xb, yb = bin_by_count(x_img, y_img, n_bins=n_bins_slope, min_per_bin=min_per_bin)
        slope, lo_slope, up_slope = fit_slope(xb, yb, method=slope_method)

        # Spearman correlation
        if spearman_mode == "objects":
            sub = df.loc[mask, [image_col, feat]].dropna()
            if sub.shape[0] > spearman_max_objects:
                sub = sub.sample(n=spearman_max_objects, random_state=0)
            xv = ensure_numeric(sub[feat]).to_numpy()
            iv = ensure_numeric(sub[image_col]).to_numpy()
            if np.nanstd(xv) == 0.0 or np.nanstd(iv) == 0.0:
                rho, pval = (np.nan, np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, pval = spearmanr(iv, xv)
        else:
            if np.nanstd(y_img) == 0.0 or np.nanstd(x_img) == 0.0:
                rho, pval = (np.nan, np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, pval = spearmanr(x_img, y_img)



        # Early vs late effect size (object-level)
        img = img_all.loc[mask]
        xv = x_obj.loc[mask]
        early = xv[img <= q_low].to_numpy()
        late = xv[img >= q_high].to_numpy()
        cd = np.nan
        early_med = np.nan
        late_med = np.nan
        if early.size > 0 and late.size > 0:
            cd = cliffs_delta(early, late)
            early_med = float(np.median(early))
            late_med = float(np.median(late))

        records.append({
            "feature": feat,
            "n_objects": int(mask.sum()),
            "n_images": int(x_img.size),
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "slope_method": slope_method,
            "slope_binned": float(slope),
            "slope_ci_low": float(lo_slope) if np.isfinite(lo_slope) else np.nan,
            "slope_ci_high": float(up_slope) if np.isfinite(up_slope) else np.nan,
            "early_median": early_med,
            "late_median": late_med,
            "cliffs_delta": float(cd),
        })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out

    # FDR across all features tested
    _, qvals = benjamini_hochberg(out["spearman_p"].to_numpy(), alpha=0.05)
    out["spearman_q"] = qvals
    out["drift_flag"] = (np.abs(out["spearman_rho"]) >= 0.10) & (out["spearman_q"] <= 0.01)
    return out.sort_values(["drift_flag", "spearman_q", "spearman_rho"], ascending=[False, True, True])



def find_image_table(input_dir: Path) -> Optional[Path]:
    """
    Locate a single Image table (e.g., '*Image*.csv.gz') under input_dir.

    Parameters
    ----------
    input_dir : pathlib.Path
        Folder to search.

    Returns
    -------
    Optional[pathlib.Path]
        Path to the Image table if found; otherwise None.
    """
    candidates = sorted(list(input_dir.rglob("*Image*.csv.gz")) +
                    list(input_dir.rglob("*Image*.tsv.gz")))
    if not candidates:
        return None
    # If multiple, pick the shortest name as a simple heuristic
    return min(candidates, key=lambda p: len(p.name))


def load_image_metadata(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load the Image table and keep columns useful as metadata.

    Parameters
    ----------
    path : pathlib.Path
        Path to Image.csv.gz (or .tsv.gz).
    logger : logging.Logger
        Logger.

    Returns
    -------
    pandas.DataFrame
        Metadata keyed by ImageNumber (must exist).
    """
    img = read_object_table(path, logger=logger)
    if "ImageNumber" not in img.columns:
        raise KeyError(f"Image table missing 'ImageNumber': {path}")

    # Keep likely metadata columns if present
    keep = ["ImageNumber",
            "Plate", "Plate_Metadata", "Well", "Well_Metadata",
            "Library", "Treatment", "cpd_id", "cpd_type",
            "Column_Metadata", "Field_Metadata", "Row", "Column",
            "ImageName", "ImageId", "ImageSeries"]
    keep = [c for c in keep if c in img.columns]

    # Also keep any column that clearly encodes time/order if present
    for extra in ("Metadata_Time", "AcqTime", "AcquisitionTime"):
        if extra in img.columns and extra not in keep:
            keep.append(extra)

    meta = img[keep].drop_duplicates("ImageNumber", keep="first")
    return meta


def parse_well_id(well: str) -> tuple[float, float]:
    """
    Convert a well like 'A01' or 'H12' to zero-based (row, col).

    Parameters
    ----------
    well : str
        Well identifier.

    Returns
    -------
    tuple[int, int]
        (row_index, col_index) zero-based.
    """
    if not isinstance(well, str) or len(well) < 2:
        return (np.nan, np.nan)
    row_char = well[0].upper()
    row_idx = ord(row_char) - ord("A")
    try:
        col_idx = int(well[1:]) - 1
    except Exception:
        col_idx = np.nan
    return (row_idx, col_idx)


def infer_plate_shape(wells: pd.Series) -> tuple[int, int]:
    """
    Infer plate grid size from observed wells.

    Parameters
    ----------
    wells : pandas.Series
        Well IDs like A01..H12.

    Returns
    -------
    tuple[int, int]
        (n_rows, n_cols)
    """
    rc = wells.dropna().astype(str).map(parse_well_id)
    rows = [r for r, c in rc if pd.notna(r)]
    cols = [c for r, c in rc if pd.notna(c)]
    if not rows or not cols:
        return (8, 12)  # sensible default
    return (int(max(rows) + 1), int(max(cols) + 1))


def compute_plate_delta_robust(
    df: pd.DataFrame,
    feature: str,
    image_col: str,
    plate_col: str,
    well_col: str,
    early_frac: float = 0.2,
    min_images_per_side: int = 1,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Compute per-plate, per-well early–late median delta robustly.

    This version collapses to per-image medians per well first, then performs an
    early/late split using plate-specific quantiles of the acquisition axis. For
    each well, it uses whatever early/late images that well actually has (with a
    minimum count per side), avoiding the all-NaN 'delta' problem seen when wells
    occur only in one half of the run.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table containing the feature, plate, well and image columns.
    feature : str
        Feature to summarise (must be numeric).
    image_col : str
        Acquisition-order column (e.g., 'ImageNumber').
    plate_col : str
        Plate identifier column (e.g., 'Plate_Metadata').
    well_col : str
        Well identifier column (e.g., 'Well_Metadata', like 'A01' or 'A1').
    early_frac : float
        Fraction for the lower tail; the upper tail uses (1 - early_frac).
    min_images_per_side : int
        Minimum number of per-well images required on each side to compute medians.
    logger : logging.Logger, optional
        Logger for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Columns: [plate_col, well_col, early_median, late_median, delta, n_img_early, n_img_late].
        Rows with insufficient images on either side have NaN delta.
    """
    req = [plate_col, well_col, image_col, feature]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns for plate delta: {miss}")

    sub = df[req].dropna(subset=[image_col, feature, plate_col, well_col]).copy()
    sub[image_col] = pd.to_numeric(sub[image_col], errors="coerce")
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub.dropna(subset=[image_col, feature])

    if sub.empty:
        return pd.DataFrame(columns=[plate_col, well_col, "early_median", "late_median",
                                     "delta", "n_img_early", "n_img_late"])

    # Per-image medians per (plate, well)
    img_med = (
        sub.groupby([plate_col, well_col, image_col], observed=False)[feature]
        .median()
        .rename("value")
        .reset_index()
    )

    # Compute plate-specific early/late thresholds
    thr = (
        img_med.groupby(plate_col, observed=False)[image_col]
        .quantile([early_frac, 1.0 - early_frac])
        .unstack(level=-1)
        .rename(columns={early_frac: "thr_low", 1.0 - early_frac: "thr_high"})
        .reset_index()
    )

    img_med = img_med.merge(thr, on=plate_col, how="left")

    # Split within each plate using plate thresholds, then summarise per well
    early = img_med.loc[img_med[image_col] <= img_med["thr_low"]]
    late = img_med.loc[img_med[image_col] >= img_med["thr_high"]]

    e_med = (
        early.groupby([plate_col, well_col], observed=False)["value"]
        .agg(["median", "count"])
        .rename(columns={"median": "early_median", "count": "n_img_early"})
    )
    l_med = (
        late.groupby([plate_col, well_col], observed=False)["value"]
        .agg(["median", "count"])
        .rename(columns={"median": "late_median", "count": "n_img_late"})
    )

    out = e_med.join(l_med, how="outer")
    # Enforce a minimal image count per side
    ok = (out["n_img_early"].fillna(0) >= min_images_per_side) & (
        out["n_img_late"].fillna(0) >= min_images_per_side
    )
    out.loc[~ok, ["early_median", "late_median"]] = np.nan
    out["delta"] = out["late_median"] - out["early_median"]
    out = out.reset_index()

    if logger is not None:
        n_total = out.shape[0]
        n_valid = int(out["delta"].notna().sum())
        logger.info(
            "Plate delta coverage: %d/%d wells with valid early and late medians.",
            n_valid, n_total
        )

    return out


def plot_plate_heatmap(
    plate_df: pd.DataFrame,
    plate: str,
    well_col: str,
    value_col: str,
    out_pdf: Path,
) -> None:
    """
    Render a plate heatmap for the given plate.

    Improvements:
    - Symmetric scaling based on robust percentiles to avoid degenerate colourbars.
    - Clear warning and early return if all values are NaN.
    - Grid labels preserved (rows A..P, columns 1..24 for 384-well; adaptively inferred).

    Parameters
    ----------
    plate_df : pandas.DataFrame
        Rows for a single plate with well IDs and a numeric value to plot.
    plate : str
        Plate identifier for title/filename.
    well_col : str
        Well ID column (e.g., 'A01').
    value_col : str
        Column holding the value to plot (e.g., 'delta').
    out_pdf : pathlib.Path
        Output path for the PDF.
    """
    if plate_df.empty or value_col not in plate_df.columns:
        return

    vals = pd.to_numeric(plate_df[value_col], errors="coerce")
    if vals.notna().sum() == 0:
        return

    n_rows, n_cols = infer_plate_shape(plate_df[well_col])
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)

    for _, row in plate_df.iterrows():
        r, c = parse_well_id(str(row[well_col]))
        if pd.isna(r) or pd.isna(c) or r >= n_rows or c >= n_cols:
            continue
        v = row[value_col]
        try:
            grid[int(r), int(c)] = float(v)
        except Exception:
            continue

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return
    low = np.nanpercentile(finite, 2)
    high = np.nanpercentile(finite, 98)

    if low >= 0:  # non-centred metric (e.g., 0..1 quantiles)
        vmin, vmax = low, high
    else:        # centred metric (delta/residual/z) -> symmetric scaling
        vmax = max(abs(low), abs(high))
        vmin = -vmax

    fig = plt.figure(figsize=(max(6, n_cols * 0.4), max(4, n_rows * 0.4)))
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(f"Plate {plate}: {value_col} per well")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(i + 1) for i in range(n_cols)], rotation=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([chr(ord("A") + i) for i in range(n_rows)])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=180)
    plt.close(fig)




def per_image_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    image_col: str,
) -> pd.DataFrame:
    """
    Summarise selected features per image (median and IQR), plus object counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table.
    feature_cols : list[str]
        Feature columns to summarise.
    image_col : str
        Acquisition-order column.

    Returns
    -------
    pandas.DataFrame
        Per-image medians/IQR and object counts for selected features.
    """
    agg_dict = {}
    for c in feature_cols:
        agg_dict[c] = ["median", "quantile", "quantile"]
    # Compute once then rename
    grouped = df.groupby(image_col, observed=False)
    med = grouped[feature_cols].median()
    q1 = grouped[feature_cols].quantile(0.25)
    q3 = grouped[feature_cols].quantile(0.75)
    cnt = grouped.size().rename("object_count")
    out = med.rename(columns=lambda x: f"{x}__median").join([
        q1.rename(columns=lambda x: f"{x}__q1"),
        q3.rename(columns=lambda x: f"{x}__q3"),
        cnt,
    ])
    out.reset_index(inplace=True)
    return out


# ----------------------------- Plotting ------------------------------------- #


def per_image_iqr_series(
    df: pd.DataFrame,
    feature: str,
    image_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-image IQR (Q3 - Q1) for a given feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table containing the feature and `image_col`.
    feature : str
        Feature for which to compute the per-image IQR.
    image_col : str
        Column indicating acquisition order (e.g., 'ImageNumber').

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (x_img, iqr_img), where x_img are sorted unique image indices and
        iqr_img are the corresponding per-image IQR values. NaNs are dropped.
    """
    sub = df[[image_col, feature]].copy()
    sub[image_col] = ensure_numeric(sub[image_col])
    sub[feature] = ensure_numeric(sub[feature])
    sub = sub.dropna()
    if sub.empty:
        return np.array([]), np.array([])

    grouped = sub.groupby(image_col, observed=False)[feature]
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = (q3 - q1).dropna()
    if iqr.empty:
        return np.array([]), np.array([])

    x = iqr.index.to_numpy(dtype=float)
    y = iqr.to_numpy(dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def make_equal_count_bins(
    x: np.ndarray,
    target_per_bin: int = 100,
    min_per_bin: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create equal-count time bins on a 1D array of image indices.

    Parameters
    ----------
    x : numpy.ndarray
        Array of image indices (float or int), typically sorted `ImageNumber`s.
    target_per_bin : int
        Target number of images per bin.
    min_per_bin : int
        Minimum number of images allowed in a bin; small tail bins are merged.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (bin_ids, bin_edges, bin_counts)
        bin_ids : integer bin assignment for each x element (1..K).
        bin_edges : edges in x-units (length K+1).
        bin_counts : number of images per bin (length K).
    """
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Determine initial number of bins
    n_bins = max(1, int(np.ceil(x.size / float(target_per_bin))))
    n_bins = min(n_bins, max(1, x.size // max(min_per_bin, 1)))

    # Quantile edges (equal-count by construction)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs)

    # Assign, guarding inclusivity at the top edge
    bin_ids = np.searchsorted(edges, x, side="right")
    bin_ids = np.clip(bin_ids, 1, n_bins)

    # Merge tiny bins with neighbours
    counts = np.array([np.sum(bin_ids == k) for k in range(1, n_bins + 1)])
    # Sweep from left to right, merging bins with < min_per_bin into the previous
    for k in range(1, n_bins + 1):
        if counts[k - 1] < min_per_bin and n_bins > 1:
            # Merge k into k-1 if possible, else into k+1
            if k > 1:
                bin_ids[bin_ids == k] = k - 1
                counts[k - 2] += counts[k - 1]
                counts[k - 1] = 0
            elif k < n_bins:
                bin_ids[bin_ids == k] = k + 1
                counts[k] += counts[k - 1]
                counts[k - 1] = 0

    # Re-label bins to be consecutive 1..K after merges
    unique_bins = sorted(set(bin_ids.tolist()))
    remap = {b: i + 1 for i, b in enumerate(unique_bins)}
    bin_ids = np.array([remap[b] for b in bin_ids], dtype=int)
    K = len(unique_bins)
    bin_counts = np.array([np.sum(bin_ids == k) for k in range(1, K + 1)])

    # Recompute edges for pretty labelling (min..max of x in each final bin)
    edges = np.zeros(K + 1, dtype=float)
    edges[0] = float(x.min())
    for k in range(1, K):
        # boundary between k and k+1
        left = x[bin_ids == k]
        right = x[bin_ids == (k + 1)]
        if left.size and right.size:
            edges[k] = 0.5 * (left.max() + right.min())
        else:
            edges[k] = edges[k - 1]
    edges[K] = float(x.max())

    return bin_ids, edges, bin_counts


def plot_time_binned_boxpanel_feature(
    df: pd.DataFrame,
    image_col: str,
    feature: str,
    out_pdf: Path,
    target_images_per_bin: int = 100,
    min_images_per_bin: int = 8,
    stats_row: Optional[pd.Series] = None,
    plate_col: Optional[str] = None,
    early_frac: float = 0.2,
    show_iqr_panel: bool = True,
) -> None:
    """
    FastQC-style time-binned boxplot panel for a single feature.

    Top row: boxplots of per-image medians across equal-count acquisition bins,
    with a line through bin medians (trend), early/late horizontal guides,
    and plate boundary markers.

    Bottom row (optional): boxplots of per-image IQRs across the same bins,
    showing how within-image variability evolves over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table for this compartment. Must contain `image_col`
        and the specified `feature`. If `plate_col` is provided, it should be
        present per object (the function aggregates per image).
    image_col : str
        Column indicating acquisition order (e.g., 'ImageNumber').
    feature : str
        Feature to summarise and plot.
    out_pdf : pathlib.Path
        Output PDF path for the figure.
    target_images_per_bin : int
        Target number of images per time bin (default: 100).
    min_images_per_bin : int
        Minimum number of images allowed in any bin; tiny bins are merged.
    stats_row : Optional[pandas.Series]
        Optional per-feature stats to annotate (keys: 'spearman_rho',
        'spearman_q', 'early_median', 'late_median', 'cliffs_delta').
        If absent, early/late lines are computed from per-image medians.
    plate_col : Optional[str]
        Optional plate identifier column; if given, plate boundary markers and
        labels are drawn based on the dominant plate per bin.
    early_frac : float
        Fraction for early/late split of images (for guides) if stats are not provided.
    show_iqr_panel : bool
        Whether to include the IQR boxplot row (default: True).

    Returns
    -------
    None
        Saves a PDF to `out_pdf`.
    """
    # --- Build per-image summaries ---
    x_med, y_med = per_image_median_series(df=df, feature=feature, image_col=image_col)
    x_iqr, y_iqr = per_image_iqr_series(df=df, feature=feature, image_col=image_col)

    if x_med.size < 5:
        return  # too few images to make a useful panel

    # --- Create equal-count bins based on images available for medians ---
    bin_ids, edges, bin_counts = make_equal_count_bins(
        x=x_med,
        target_per_bin=target_images_per_bin,
        min_per_bin=min_images_per_bin,
    )
    K = int(bin_ids.max()) if bin_ids.size else 0
    positions = np.arange(1, K + 1, dtype=float)

    # Group per-image medians by bin
    bins_med: List[np.ndarray] = [y_med[bin_ids == k] for k in range(1, K + 1)]
    # Bin medians (for the thin trend line)
    bin_meds = np.array([np.median(v) if v.size else np.nan for v in bins_med])

    # Build matching bins for IQRs using the same edges (map x_iqr to bin via edges)
    bins_iqr: List[np.ndarray] = []
    if show_iqr_panel and x_iqr.size:
        # Assign iqr images to bins by edges
        iqr_ids = np.searchsorted(edges, x_iqr, side="right")
        iqr_ids = np.clip(iqr_ids, 1, K)
        bins_iqr = [y_iqr[iqr_ids == k] for k in range(1, K + 1)]

    # --- Early/Late guides (prefer object-level stats if provided) ---
    early_line = np.nan
    late_line = np.nan
    rho = np.nan
    qval = np.nan
    cliffs = np.nan

    if stats_row is not None:
        rho = float(stats_row.get("spearman_rho", np.nan))
        qval = float(stats_row.get("spearman_q", np.nan))
        cliffs = float(stats_row.get("cliffs_delta", np.nan))
        early_line = float(stats_row.get("early_median", np.nan))
        late_line = float(stats_row.get("late_median", np.nan))

    if not np.isfinite(early_line) or not np.isfinite(late_line):
        # Fallback: compute from per-image medians
        q_low = np.quantile(x_med, early_frac)
        q_high = np.quantile(x_med, 1.0 - early_frac)
        early_vals = y_med[x_med <= q_low]
        late_vals = y_med[x_med >= q_high]
        if early_vals.size:
            early_line = float(np.median(early_vals))
        if late_vals.size:
            late_line = float(np.median(late_vals))

    # --- Plate boundary detection (optional) ---
    plate_boundaries: List[int] = []
    plate_labels: Dict[int, str] = {}
    if plate_col is not None and plate_col in df.columns:
        # Map each image to a plate label
        img_plate = (
            df[[image_col, plate_col]]
            .dropna()
            .groupby(image_col, observed=False)[plate_col]
            .agg(lambda s: s.iloc[0])
        )
        # Dominant plate per bin
        bin_plates: List[Optional[str]] = []
        for k in range(1, K + 1):
            imgs_k = x_med[bin_ids == k]
            if imgs_k.size == 0:
                bin_plates.append(None)
                continue
            plate_vals = img_plate.reindex(imgs_k).astype(str).tolist()
            if len(plate_vals) == 0:
                bin_plates.append(None)
            else:
                most = Counter(plate_vals).most_common(1)[0][0]
                bin_plates.append(most)
        # Boundaries where dominant plate changes between bins
        for k in range(1, K):
            if bin_plates[k - 1] != bin_plates[k]:
                plate_boundaries.append(k + 0)  # vertical line at k + 0.5 later
        # Labels centred over contiguous runs
        start = 1
        current = bin_plates[0] if bin_plates else None
        for k in range(2, K + 2):
            nxt = bin_plates[k - 1] if (k - 1) < K else None
            if nxt != current:
                mid = (start + (k - 1)) / 2.0
                if current is not None:
                    plate_labels[int(np.round(mid * 10))] = str(current)  # keyed by rough position
                start = k
                current = nxt

    # --- Plot ---
    nrows = 2 if show_iqr_panel else 1
    fig = plt.figure(figsize=(max(8.0, K * 0.25), 3.8 * nrows))
    axes = fig.subplots(nrows=nrows, ncols=1, sharex=True)

    if nrows == 1:
        axes = [axes]  # make iterable

    # Top: per-image medians
    ax0 = axes[0]
    ax0.boxplot(
        x=bins_med,
        positions=positions,
        widths=0.7,
        manage_ticks=False,
        showfliers=False,
    )
    # Thin line through bin medians
    ax0.plot(positions, bin_meds, linewidth=1.5)

    # Early/Late horizontal guides
    if np.isfinite(early_line):
        ax0.axhline(y=early_line, linestyle="--", linewidth=1)
    if np.isfinite(late_line):
        ax0.axhline(y=late_line, linestyle="--", linewidth=1)

    ax0.set_ylabel(f"{feature}\n(per-image median)")
    ax0.set_title("Time-binned boxplots (equal-count bins)")

    # Plate boundaries
    for k in plate_boundaries:
        ax0.axvline(x=k + 0.5, linestyle=":", linewidth=1)

    # Bottom: per-image IQRs
    if show_iqr_panel:
        ax1 = axes[1]
        ax1.boxplot(
            x=bins_iqr,
            positions=positions,
            widths=0.7,
            manage_ticks=False,
            showfliers=False,
        )
        ax1.set_ylabel(f"{feature}\n(per-image IQR)")

        # Carry plate boundaries to lower panel too
        for k in plate_boundaries:
            ax1.axvline(x=k + 0.5, linestyle=":", linewidth=1)

    # X axis: bin indices + n per bin
    axes[-1].set_xlabel("Acquisition (binned by ImageNumber)")
    axes[-1].set_xticks(positions)
    # Show bin counts as labels: e.g., 'n=98'
    xticklabels = [f"n={int(n)}" for n in bin_counts]
    axes[-1].set_xticklabels(xticklabels, rotation=90)

    # Add a compact annotation (ρ, q, Δmedian from early/late lines, δ)
    delta_median = np.nan
    if np.isfinite(early_line) and np.isfinite(late_line):
        delta_median = late_line - early_line

    annot = _format_drift_annotation(
        rho=rho,
        q=qval,
        slope=np.nan,  # not shown here (this panel is non-parametric)
        delta_median=delta_median,
        cliffs_delta=cliffs,
    )
    if annot:
        axes[0].text(
            0.01,
            0.99,
            annot,
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, linewidth=0.0),
        )

    # Plate labels over spans (if any)
    if plate_labels:
        for key, lab in plate_labels.items():
            x_pos = key / 10.0
            axes[0].text(
                x_pos,
                1.02,
                lab,
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=180)
    plt.close(fig)


def plot_feature_vs_time_hexbin_with_trend(
    df: pd.DataFrame,
    image_col: str,
    feature: str,
    out_pdf: Path,
    rolling_window: int = 301,
    max_points_plot: int = 200_000,
    stats_row: Optional[pd.Series] = None,
    early_frac: float = 0.2,
) -> None:
    """
    Plot object-level values vs acquisition order as a hexbin, with a rolling
    median (non-linear shape), a linear trend line fitted on per-image medians,
    a light 95% CI ribbon on the trend slope, early/late horizontal guides, and
    a compact annotation of key drift metrics (rho, q, slope, median delta, delta).

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table containing `image_col` and `feature`.
    image_col : str
        Column indicating acquisition order (e.g., "ImageNumber").
    feature : str
        Feature to plot on the y-axis.
    out_pdf : pathlib.Path
        Output path for the PDF figure.
    rolling_window : int
        Window (in samples) for the rolling median overlay on object values.
    max_points_plot : int
        Maximum number of object rows to sample for the hexbin (to keep files light).
    stats_row : Optional[pandas.Series]
        Optional row from the per-feature stats table providing:
        'spearman_rho', 'spearman_q', 'slope_binned', 'slope_ci_low',
        'slope_ci_high', 'early_median', 'late_median', 'cliffs_delta'.
        If absent, the function computes reasonable fallbacks for plotting.
    early_frac : float
        Fraction defining the early/late split for horizontal guides (default 0.2).

    Returns
    -------
    None
        Saves a PDF to `out_pdf`.
    """
    # ---- Prepare object-level subsample for hexbin + rolling median ----
    sub = df[[image_col, feature]].dropna()
    if sub.empty:
        return
    if sub.shape[0] > max_points_plot:
        sub = sub.sample(n=max_points_plot, random_state=0)

    x = ensure_numeric(sub[image_col]).to_numpy()
    y = ensure_numeric(sub[feature]).to_numpy()
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Rolling median over object values for shape (non-linear drift visible)
    y_roll = rolling_median(y=y, window=min(rolling_window, max(3, len(y) // 50)))

    # ---- Build per-image medians series for trend line anchor ----
    # Use just the two columns for speed/memory
    x_img, y_img = per_image_median_series(
        df=sub[[image_col, feature]],
        feature=feature,
        image_col=image_col,
    )

    # If too few images, plotting a trend adds no value
    have_trend = x_img.size >= 5

    # Prepare trend slope and CI (prefer the stats table if supplied)
    slope = np.nan
    lo = np.nan
    hi = np.nan
    rho = np.nan
    qval = np.nan
    early_med = np.nan
    late_med = np.nan
    cliffs = np.nan

    if stats_row is not None:
        slope = float(stats_row.get("slope_binned", np.nan))
        lo = float(stats_row.get("slope_ci_low", np.nan))
        hi = float(stats_row.get("slope_ci_high", np.nan))
        rho = float(stats_row.get("spearman_rho", np.nan))
        qval = float(stats_row.get("spearman_q", np.nan))
        early_med = float(stats_row.get("early_median", np.nan))
        late_med = float(stats_row.get("late_median", np.nan))
        cliffs = float(stats_row.get("cliffs_delta", np.nan))

    # Fallback: estimate slope quickly if missing
    if have_trend and (not np.isfinite(slope)):
        xb, yb = bin_by_count(x=x_img, y=y_img, n_bins=400, min_per_bin=20)
        if xb.size >= 5:
            # Centre x to keep intercept stable; fit OLS for plotting
            x0 = xb - np.median(xb)
            coef = np.polyfit(x=x0, y=yb, deg=1)
            slope = float(coef[0])

    # Anchor the line at the median of per-image medians (robust, interpretable)
    x_c = np.median(x_img) if have_trend else np.median(x)
    y_c = np.median(y_img) if have_trend else np.median(y)

    # Build line coordinates across the observed x-range
    x_line = np.linspace(float(x.min()), float(x.max()), num=200)
    y_line = None
    y_lo = None
    y_hi = None
    if np.isfinite(slope):
        y_line = y_c + slope * (x_line - x_c)
        # CI ribbon from slope bounds if available
        if np.isfinite(lo) and np.isfinite(hi):
            y_lo = y_c + lo * (x_line - x_c)
            y_hi = y_c + hi * (x_line - x_c)

    # If early/late medians are missing, estimate them for guides
    if not np.isfinite(early_med) or not np.isfinite(late_med):
        q_low = np.quantile(x, early_frac)
        q_high = np.quantile(x, 1.0 - early_frac)
        early_vals = y[x <= q_low]
        late_vals = y[x >= q_high]
        if early_vals.size > 0:
            early_med = float(np.median(early_vals))
        if late_vals.size > 0:
            late_med = float(np.median(late_vals))

    # ---- Plot ----
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)

    hb = ax.hexbin(x, y, gridsize=80, mincnt=1)  # default colormap; no explicit colours
    ax.plot(x, y_roll, linewidth=2, label="Rolling median")

    # Trend line and its CI ribbon
    if y_line is not None:
        ax.plot(x_line, y_line, linewidth=2, linestyle="-", label="Linear trend (per-image medians)")
        if (y_lo is not None) and (y_hi is not None):
            ax.fill_between(x_line, y_lo, y_hi, alpha=0.2, label="Trend 95% CI")

    # Early / Late horizontal guides
    if np.isfinite(early_med):
        ax.axhline(y=early_med, linestyle="--", linewidth=1, label="Early median")
    if np.isfinite(late_med):
        ax.axhline(y=late_med, linestyle="--", linewidth=1, label="Late median")

    ax.set_xlabel(image_col)
    ax.set_ylabel(feature)
    ax.set_title("Acquisition drift: hexbin + rolling median + trend")

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Object count")

    # Compact metrics annotation
    annot = _format_drift_annotation(
        rho=rho,
        q=qval,
        slope=slope,
        delta_median=(late_med - early_med) if (np.isfinite(late_med) and np.isfinite(early_med)) else np.nan,
        cliffs_delta=cliffs,
    )
    if annot:
        ax.text(
            0.01,
            0.99,
            annot,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, linewidth=0.0),
        )

    # Keep legend compact; avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        # Deduplicate by label while preserving order
        seen = set()
        keep = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                keep.append((h, l))
        ax.legend(
            handles=[h for h, _ in keep],
            labels=[l for _, l in keep],
            loc="lower right",
            fontsize=8,
            frameon=True,
        )

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=180)
    plt.close(fig)


def _format_drift_annotation(
    rho: float,
    q: float,
    slope: float,
    delta_median: float,
    cliffs_delta: float,
) -> str:
    """
    Format a compact, readable annotation of drift metrics.

    Parameters
    ----------
    rho : float
        Spearman rank correlation (per-image medians vs ImageNumber).
    q : float
        FDR-adjusted q-value (Benjamini–Hochberg) for the Spearman test.
    slope : float
        Linear slope per unit of ImageNumber (per-image medians, binned fit).
    delta_median : float
        Late - Early median difference (object-level).
    cliffs_delta : float
        Cliff's delta effect size (object-level), in [-1, 1].

    Returns
    -------
    str
        Multi-token string like 'ρ = -0.21; q = 3×10⁻⁵; slope = -8.8e-5 / img; Δmedian = -0.36; δ = -0.22'.
        Empty string if all metrics are NaN.
    """
    parts = []

    if np.isfinite(rho):
        parts.append(f"ρ = {rho:.2f}")

    if np.isfinite(q):
        parts.append(f"q = {format_sci(q)}")

    if np.isfinite(slope):
        parts.append(f"slope = {slope:.3g} / img")

    if np.isfinite(delta_median):
        parts.append(f"Δmedian = {delta_median:.3g}")

    if np.isfinite(cliffs_delta):
        parts.append(f"δ = {cliffs_delta:.2f}")

    return "; ".join(parts)


def format_sci(x: float, sigfigs: int = 1) -> str:
    """
    Format a small p/q-value in scientific notation using a multiplication sign
    and superscript exponent (e.g., 3×10⁻⁵), falling back to fixed-point when
    appropriate.

    Parameters
    ----------
    x : float
        Value to format.
    sigfigs : int
        Number of significant figures for the mantissa (default 1 → '3×10⁻⁵').

    Returns
    -------
    str
        Nicely formatted value, or 'NA' if not finite.
    """
    if not np.isfinite(x):
        return "NA"
    if x == 0.0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    # Use a thin space around the multiplication sign for readability
    if abs(exp) >= 3:  # scientific format
        mant_str = f"{mant:.{sigfigs}g}"
        exp_str = str(exp).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{mant_str}×10{exp_str}"
    # Otherwise show in fixed or compact
    return f"{x:.3g}"



def plot_feature_vs_time_hexbin(
    df: pd.DataFrame,
    image_col: str,
    feature: str,
    out_pdf: Path,
    rolling_window: int = 301,
    max_points_plot: int = 200_000,
) -> None:
    """
    Hexbin scatter of a single feature vs acquisition order with rolling median.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table.
    image_col : str
        Acquisition-order column.
    feature : str
        Feature to plot.
    out_pdf : pathlib.Path
        Output pdf path.
    rolling_window : int
        Window in samples for rolling median overlay.
    max_points_plot : int
        Max random sample size for plotting to limit file size.

    Returns
    -------
    None
    """
    sub = df[[image_col, feature]].dropna()
    if sub.empty:
        return
    if sub.shape[0] > max_points_plot:
        sub = sub.sample(n=max_points_plot, random_state=0)

    x = ensure_numeric(sub[image_col]).to_numpy()
    y = ensure_numeric(sub[feature]).to_numpy()
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y_med = rolling_median(y, window=min(rolling_window, max(3, len(y)//50)))

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    hb = ax.hexbin(x, y, gridsize=80, mincnt=1)
    ax.plot(x, y_med, linewidth=2)
    ax.set_xlabel(image_col)
    ax.set_ylabel(feature)
    ax.set_title("Acquisition drift (hexbin + rolling median)")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=180)
    plt.close(fig)


# ------------------------------ Driver -------------------------------------- #


def run_for_compartment(
    files: list[Path],
    compartment: str,
    out_dir: Path,
    image_col: str,
    feature_list: Optional[list[str]],
    plot_features: Optional[list[str]],
    bin_size: int,
    max_points_plot: int,
    controls_query: Optional[str],
    plate_col: str,
    well_col: str,
    heatmap_features: Optional[str],
    image_meta: Optional[pd.DataFrame],
    slope_method: str,
    n_bins_slope: int,
    min_per_bin: int,
    spearman_mode: str,
    spearman_max_objects: int,
    controls_wells: list[str],
    heatmap_mode: str,
    early_frac: float,
    min_images_per_side: int,
    min_wells_for_delta: int,
    resid_window: int,
    logger: logging.Logger,
) -> None:
    """
    Execute object-level acquisition-drift QC for a single compartment.

    This function loads all matching object tables for the specified compartment,
    merges Image-level metadata (if provided), selects an informative set of
    object-level features, computes drift statistics against acquisition order,
    writes TSV outputs, generates scalable plots (hexbin + rolling median), and
    renders per-plate heatmaps of early–late deltas.

    Parameters
    ----------
    files : list[pathlib.Path]
        File paths belonging to this compartment (e.g., Cell, Cytoplasm, Nuclei).
    compartment : str
        Compartment name ("Cell", "Cytoplasm", or "Nuclei").
    out_dir : pathlib.Path
        Root directory to write outputs. A subfolder named after the compartment
        will be created inside.
    image_col : str
        Column indicating acquisition order (e.g., "ImageNumber").
    feature_list : Optional[list[str]]
        If provided, the exact feature columns to evaluate. If None, features are
        auto-selected from the table header using `feature_candidates(...)`.
    plot_features : Optional[list[str]]
        Optional explicit list of features to plot. If None, the top drifting
        features by FDR are used (fallback to the first few numeric features).
    bin_size : int
        Reserved for future binned-distribution plots (not used by hexbin plots).
    max_points_plot : int
        Maximum number of randomly sampled object rows to plot per feature,
        to control output size.
    controls_query : Optional[str]
        Pandas query string to subset control-only analysis (e.g.,
        '(Library == "DMSO") or (cpd_type == "DMSO")'). If None, control-only
        stats are skipped.
    plate_col : str
        Column name for plate identifier (e.g., "Plate_Metadata").
    well_col : str
        Column name for well identifier in 'A01'..'H12' style (e.g., "Well_Metadata").
    heatmap_features : Optional[str]
        Comma-separated list of features to include in plate heatmaps. If None,
        the top three drifting features by FDR are used (fallback to first three).
    image_meta : Optional[pandas.DataFrame]
        Image-level metadata keyed by "ImageNumber" to merge onto object rows.
        May include columns such as Plate/Well/Library/cpd_type.
    logger : logging.Logger
        Logger instance for progress and diagnostics.

    Returns
    -------
    None
        Results are written to disk under `out_dir/compartment`.
    """
    comp_dir = out_dir / compartment
    comp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s: %d file(s) ===", compartment, len(files))

    # Load and concatenate all object tables for this compartment
    frames: list[pd.DataFrame] = []
    for path in files:
        df = read_object_table(path, logger=logger)
        if image_col not in df.columns:
            logger.warning(
                "[%s] missing '%s'; skipping file: %s",
                compartment, image_col, path
            )
            continue
        frames.append(df)

    if not frames:
        logger.warning("No valid tables for %s; skipping.", compartment)
        return

    data = pd.concat(frames, axis=0, ignore_index=True)
    n_rows, n_cols = data.shape
    logger.info("%s: concatenated table shape = %s", compartment, (n_rows, n_cols))

    # Merge Image-level metadata (Plate/Well/controls) if available
    if image_meta is not None:
        if "ImageNumber" not in data.columns:
            logger.warning(
                "%s: object table missing ImageNumber; cannot merge metadata.",
                compartment
            )
        else:
            before_cols = data.shape[1]
            data = data.merge(
                image_meta,
                on="ImageNumber",
                how="left",
                validate="many_to_one"
            )
            logger.info(
                "%s: merged Image metadata columns: +%d cols (now %d).",
                compartment, data.shape[1] - before_cols, data.shape[1]
            )

    # Select features (auto or user-provided), then keep numeric only
    if feature_list is None:
        feats = feature_candidates(header=data.columns, compartment=compartment)
        logger.info("%s: auto-selected %d feature(s) for drift testing.", compartment, len(feats))
    else:
        feats = [c for c in feature_list if c in data.columns]
        logger.info(
            "%s: user-specified %d/%d feature(s) present.",
            compartment, len(feats), len(feature_list)
        )

    if not feats:
        logger.warning("%s: no features selected; skipping.", compartment)
        return

    num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(data[c])]
    if not num_feats:
        logger.warning("%s: no numeric features after filtering; skipping.", compartment)
        return

    # Compute drift statistics (object-level) and write TSV
    stats = compute_drift_stats(
        df=data,
        feature_cols=num_feats,
        image_col=image_col,
        early_frac=0.2,
        min_points=2000,
        slope_method=slope_method,
        n_bins_slope=n_bins_slope,
        min_per_bin=min_per_bin,
        spearman_mode=spearman_mode,
        spearman_max_objects=spearman_max_objects,
    )

    stats_path = comp_dir / "qc_feature_drift_raw.tsv"
    stats.to_csv(stats_path, sep="\t", index=False)
    logger.info("%s: wrote drift stats -> %s", compartment, stats_path)


    # Optional: control-only analysis (query first, then wells fallback)
    ctrl_df = None
    if controls_query and all(c in data.columns for c in ["Library", "cpd_type"]):
        try:
            tmp = data.query(controls_query)
            if tmp.shape[0] >= 2000:
                ctrl_df = tmp
            else:
                logger.warning(
                    "%s: controls query returned %d rows (<2000); will try wells fallback.",
                    compartment, tmp.shape[0]
                )
        except Exception as exc:
            logger.error("%s: controls query failed (%s). Falling back to wells list. Error: %s",
                         compartment, controls_query, exc)
    # fallback by wells as you already had



    if ctrl_df is None:
        ctrl_df = get_controls_subset_by_wells(
            df=data,
            well_col=well_col,
            controls_wells=controls_wells,
            logger=logger,
        )

    if ctrl_df is not None and ctrl_df.shape[0] >= 2000:
        stats_ctrl = compute_drift_stats(
            df=ctrl_df,
            feature_cols=num_feats,
            image_col=image_col,
            early_frac=0.2,
            min_points=500,
            slope_method=slope_method,
            n_bins_slope=n_bins_slope,
            min_per_bin=min_per_bin,
            spearman_mode=spearman_mode,
            spearman_max_objects=spearman_max_objects,
        )
        stats_ctrl_path = comp_dir / "qc_feature_drift_controls.tsv"
        stats_ctrl.to_csv(stats_ctrl_path, sep="\t", index=False)
        logger.info("%s: wrote control-only drift stats -> %s", compartment, stats_ctrl_path)
    else:
        logger.warning("%s: no adequate control subset available; skipping control-only stats.", compartment)


    # Per-image summary (medians/IQRs) for a compact panel, then write TSV
    per_img = per_image_summary(
        df=data,
        feature_cols=num_feats[:20],
        image_col=image_col,
    )
    per_img_path = comp_dir / "qc_per_image_summary.tsv"
    per_img.to_csv(per_img_path, sep="\t", index=False)
    logger.info("%s: wrote per-image summary -> %s", compartment, per_img_path)

    # Feature shortlist for plots
    if plot_features:
        pfeats = [c for c in plot_features if c in data.columns]
    else:
        pfeats = (
            stats.nsmallest(8, "spearman_q")["feature"].tolist()
            if not stats.empty else num_feats[:8]
        )



    # Hexbin plots with rolling median (scalable for big N)
    plots_dir = comp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for feat in pfeats:
        stats_row = None
        if 'feature' in stats.columns:
            hit = stats.loc[stats['feature'] == feat]
            if not hit.empty:
                stats_row = hit.iloc[0]

        out_pdf = plots_dir / f"{feat}.hexbin.pdf"
        plot_feature_vs_time_hexbin_with_trend(
            df=data,
            image_col=image_col,
            feature=feat,
            out_pdf=out_pdf,
            rolling_window=301,
            max_points_plot=max_points_plot,
            stats_row=stats_row,
            early_frac=0.2,
        )
    logger.info("%s: wrote %d plot(s) to %s", compartment, len(pfeats), plots_dir)
    # Boxpanel plots (FastQC-style)
    box_dir = comp_dir / "boxpanels"
    box_dir.mkdir(exist_ok=True)

    for feat in pfeats:
        row = None
        if 'feature' in stats.columns:
            hit = stats.loc[stats['feature'] == feat]
            if not hit.empty:
                row = hit.iloc[0]

        out_pdf = box_dir / f"{feat}.time_binned_boxpanel.pdf"
        plot_time_binned_boxpanel_feature(
            df=data,
            image_col=image_col,
            feature=feat,
            out_pdf=out_pdf,
            target_images_per_bin=100,
            min_images_per_bin=8,
            stats_row=row,
            plate_col=plate_col if plate_col in data.columns else None,
            early_frac=0.2,
            show_iqr_panel=True,
        )
    logger.info("%s: wrote %d boxpanel plot(s) to %s", compartment, len(pfeats), box_dir)

    # Plate heatmaps: choose features explicitly or fall back to top three drifters
    if heatmap_features:
        heat_feats = [
            s.strip() for s in heatmap_features.split(",")
            if s.strip() and s.strip() in data.columns
        ]
    else:
        heat_feats = (
            stats.nsmallest(3, "spearman_q")["feature"].tolist()
            if not stats.empty else num_feats[:3]
        )

    # Compute maps and render heatmaps (if Plate/Well present)
    if plate_col in data.columns and well_col in data.columns:
        hm_dir = comp_dir / "heatmaps"
        hm_dir.mkdir(exist_ok=True)
        pdf_count = 0

        for feat in heat_feats:
            made_any = 0

            # 1) Delta (auto or explicit)
            if heatmap_mode in ("delta", "auto"):
                delta_df = compute_plate_delta_robust(
                    df=data,
                    feature=feat,
                    image_col=image_col,
                    plate_col=plate_col,
                    well_col=well_col,
                    early_frac=early_frac,
                    min_images_per_side=min_images_per_side,
                    logger=logger,
                )
                delta_df.to_csv(hm_dir / f"{feat}.plate_delta.tsv", sep="\t", index=False)
                non_null = int(delta_df["delta"].notna().sum()) if "delta" in delta_df else 0
                if non_null >= min_wells_for_delta:
                    for plate_id, plate_block in delta_df.groupby(plate_col, observed=False):
                        pdf = hm_dir / f"{feat}.plate_{plate_id}.delta.pdf"
                        plot_plate_heatmap(
                            plate_df=plate_block,
                            plate=str(plate_id),
                            well_col=well_col,
                            value_col="delta",
                            out_pdf=pdf,
                        )
                        made_any += 1

            # 2) Residuals (fallback or explicit)
            if made_any == 0 and heatmap_mode in ("residual", "auto"):
                resid_df = compute_plate_residual_map(
                    df=data,
                    feature=feat,
                    image_col=image_col,
                    plate_col=plate_col,
                    well_col=well_col,
                    window=resid_window,
                    logger=logger,
                )
                resid_df.to_csv(hm_dir / f"{feat}.plate_residual.tsv", sep="\t", index=False)
                for plate_id, plate_block in resid_df.groupby(plate_col, observed=False):
                    pdf = hm_dir / f"{feat}.plate_{plate_id}.residual.pdf"
                    plot_plate_heatmap(
                        plate_df=plate_block.rename(columns={"residual": "delta"}),
                        plate=str(plate_id),
                        well_col=well_col,
                        value_col="delta",
                        out_pdf=pdf,
                    )
                    made_any += 1

            # 3) Diagnostics if still nothing (or explicitly requested)
            if made_any == 0 and heatmap_mode in ("time_quantile", "well_median_z", "auto"):
                tq_df = per_well_time_quantile(
                    df=data,
                    image_col=image_col,
                    plate_col=plate_col,
                    well_col=well_col,
                )
                for plate_id, plate_block in tq_df.groupby(plate_col, observed=False):
                    pdf = hm_dir / f"{feat}.plate_{plate_id}.acq_time_quantile.pdf"
                    plot_plate_heatmap(
                        plate_df=plate_block.rename(columns={"acq_q": "delta"}),
                        plate=str(plate_id),
                        well_col=well_col,
                        value_col="delta",
                        out_pdf=pdf,
                    )
                    made_any += 1

                wz_df = per_well_feature_median_z(
                    df=data,
                    feature=feat,
                    plate_col=plate_col,
                    well_col=well_col,
                )
                for plate_id, plate_block in wz_df.groupby(plate_col, observed=False):
                    pdf = hm_dir / f"{feat}.plate_{plate_id}.well_median_z.pdf"
                    plot_plate_heatmap(
                        plate_df=plate_block.rename(columns={"feat_z": "delta"}),
                        plate=str(plate_id),
                        well_col=well_col,
                        value_col="delta",
                        out_pdf=pdf,
                    )
                    made_any += 1

            pdf_count += made_any

        logger.info(
            "%s: plate heatmap PDFs created: %d (mode=%s). Output -> %s",
            compartment, pdf_count, heatmap_mode, hm_dir
        )
    else:
        logger.warning(
            "%s: plate/well columns not found (%s, %s); skipping heatmaps.",
            compartment, plate_col, well_col
        )



def main(args: argparse.Namespace) -> None:
    """
    Entry point: iterate per-compartment files and run drift QC.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    None
    """
    out_dir = Path(args.out_dir).resolve()
    logger = setup_logger(out_dir=out_dir, level=args.log_level)

    logger.info("Python Version: %s", sys.version_info)
    logger.info("Command-line Arguments: %s", " ".join(sys.argv))
    include_glob = args.include_glob

    # Define input_dir first
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Image table discovery + load
    image_meta = None
    image_path = find_image_table(input_dir)
    if image_path is None:
        logger.warning("No Image table found under %s. Proceeding without merged metadata.", input_dir)
    else:
        logger.info("Using Image table: %s", image_path)
        image_meta = load_image_metadata(image_path, logger=logger)

    if args.controls_wells_tsv:
        controls_wells = pd.read_csv(args.controls_wells_tsv, sep="\t", header=None)[0].astype(str).tolist()
    else:
        controls_wells = [w.strip() for w in args.controls_wells.split(",") if w.strip()]



    include_glob = args.include_glob  # default already set above

    files_by_comp = list_compartment_files(input_dir=input_dir, include_glob=include_glob)
    if not files_by_comp:
        logger.error("No matching compartment files found under %s with patterns: %s",
                     input_dir, include_glob)
        return

    feature_list: Optional[list[str]] = None
    if args.feature_list_tsv:
        fl_path = Path(args.feature_list_tsv)
        feature_list = pd.read_csv(fl_path, sep="\t", header=None)[0].tolist()

    plot_features: Optional[list[str]] = None
    if args.plot_features:
        plot_features = [s.strip() for s in args.plot_features.split(",") if s.strip()]

    for comp, files in files_by_comp.items():
        run_for_compartment(
            files=files,
            compartment=comp,
            out_dir=out_dir,
            image_col=args.image_col,
            feature_list=feature_list,
            plot_features=plot_features,
            bin_size=args.bin_size,
            max_points_plot=args.max_points_plot,
            controls_query=args.controls_query,
            plate_col=args.plate_col,
            well_col=args.well_col,
            heatmap_features=args.heatmap_features,
            image_meta=image_meta,
            slope_method=args.slope_method,
            n_bins_slope=args.n_bins_slope,
            min_per_bin=args.min_per_bin,
            spearman_mode=args.spearman_mode,
            spearman_max_objects=args.spearman_max_objects,       # always pass; it's None if not found
            controls_wells=controls_wells,
            heatmap_mode=args.heatmap_mode,
            early_frac=args.early_frac,
            min_images_per_side=args.min_images_per_side,
            min_wells_for_delta=args.min_wells_for_delta,
            resid_window=args.resid_window,
            logger=logger,
        )

    logger.info("Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stand-alone per-compartment drift QC for CellProfiler object files."
    )

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing CellProfiler object CSV/TSV files "
             "(e.g., *Cell.csv.gz, *Cytoplasm.csv.gz, *Nuclei.csv.gz).",
    )
    io.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write QC outputs (TSVs and plots).",
    )
    io.add_argument(
        "--include_glob",
        nargs="+",
        default=[
            "*[Nn]uclei*.csv.gz", "*[Nn]uclei*.tsv.gz",
            "*[Cc]ell*.csv.gz", "*[Cc]ell*.tsv.gz",
            "*[Cc]ytoplasm*.csv.gz", "*[Cc]ytoplasm*.tsv.gz",
            "*[Mm]itochondria*.csv.gz", "*[Mm]itochondria*.tsv.gz",
            "*[Aa]crosome*.csv.gz", "*[Aa]crosome*.tsv.gz",
        ],
        help=("Filename patterns to include. Defaults cover Cell/Cytoplasm/Nuclei/"
              "Mitochondria/Acrosome (csv/tsv, case-agnostic via [Aa])."),
    )

    # Columns / metadata
    cols = parser.add_argument_group("Column mapping")
    cols.add_argument(
        "--image_col",
        default="ImageNumber",
        help="Column indicating acquisition order (default: ImageNumber).",
    )
    cols.add_argument(
        "--plate_col",
        default="Plate_Metadata",
        help="Column holding plate identifier (default: Plate_Metadata).",
    )
    cols.add_argument(
        "--well_col",
        default="Well_Metadata",
        help="Column holding well ID like A01, H12 (default: Well_Metadata).",
    )

    # Feature selection & plotting
    feat = parser.add_argument_group("Feature selection and plotting")
    feat.add_argument(
        "--feature_list_tsv",
        default=None,
        help="Optional TSV (one column, no header) listing exact feature columns to test.",
    )
    feat.add_argument(
        "--plot_features",
        default=None,
        help="Optional comma-separated list of features to plot. Defaults to top drifters.",
    )
    feat.add_argument(
        "--bin_size",
        type=int,
        default=150,
        help="Reserved for future binned-distribution plots (default: 150).",
    )
    feat.add_argument(
        "--max_points_plot",
        type=int,
        default=200_000,
        help="Maximum random-subsampled points for plotting per feature (default: 200000).",
    )
    feat.add_argument(
        "--heatmap_features",
        default=None,
        help="Comma-separated list of features to plot as plate heatmaps.",
    )

    # Drift modelling
    drift = parser.add_argument_group("Drift modelling")
    drift.add_argument(
        "--slope_method",
        default="ols_binned",
        choices=["ols_binned", "huber_binned", "ols"],
        help="Slope estimator on per-image medians (default: ols_binned).",
    )
    drift.add_argument(
        "--n_bins_slope",
        type=int,
        default=400,
        help="Target number of equal-count bins for slope fitting (default: 400).",
    )
    drift.add_argument(
        "--min_per_bin",
        type=int,
        default=20,
        help="Minimum observations required per bin (default: 20).",
    )
    drift.add_argument(
        "--spearman_mode",
        default="images",
        choices=["images", "objects"],
        help="Compute Spearman on per-image medians or on objects (default: images).",
    )
    drift.add_argument(
        "--spearman_max_objects",
        type=int,
        default=500_000,
        help="If 'objects', random cap of rows for Spearman (default: 500000).",
    )

    # Controls handling
    controls = parser.add_argument_group("Controls handling")
    controls.add_argument(
        "--controls_query",
        default=None,
        help=("Pandas query string to subset control objects (e.g., 'Library == \"DMSO\"'). "
              "If required columns are absent, this is skipped."),
    )
    controls.add_argument(
        "--controls_wells",
        default="A23,B23,C23,D23,E23,F23,G23,H23,I23,J23,K23,L23,M23,N23,O23,P23",
        help=("Comma-separated list of control wells (case-insensitive). "
              "Used if --controls_query is missing/invalid or yields too few rows."),
    )
    controls.add_argument(
        "--controls_wells_tsv",
        default=None,
        help=("Optional TSV (one well per line, no header) to define control wells. "
              "Overrides --controls_wells if provided."),
    )

    # Heatmap modes & robustness
    hm = parser.add_argument_group("Heatmap modes and robustness")
    hm.add_argument(
        "--heatmap_mode",
        choices=["auto", "delta", "residual", "time_quantile", "well_median_z"],
        default="auto",
        help=("Statistic to plot for plate heatmaps. "
              "'auto' tries delta then falls back to residual; "
              "diagnostics include time_quantile and well_median_z."),
    )
    hm.add_argument(
        "--early_frac",
        type=float,
        default=0.2,
        help="Early/late split fraction for delta maps (default: 0.2).",
    )
    hm.add_argument(
        "--min_images_per_side",
        type=int,
        default=1,
        help="Minimum per-well image count required on each side for delta (default: 1).",
    )
    hm.add_argument(
        "--min_wells_for_delta",
        type=int,
        default=10,
        help=("Minimum number of wells with valid early/late medians to accept "
              "delta heatmaps before falling back (default: 10)."),
    )
    hm.add_argument(
        "--resid_window",
        type=int,
        default=301,
        help="Rolling-median window for residual time trend smoothing (default: 301).",
    )

    # Logging
    log = parser.add_argument_group("Logging")
    log.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    main(parser.parse_args())
