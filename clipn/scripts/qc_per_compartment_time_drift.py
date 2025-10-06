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
- Theil–Sen robust slope with 95% confidence interval.
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

Author
------
Prepared for object-level QC with per-compartment outputs and UK English.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, theilslopes


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
) -> pd.DataFrame:
    """
    Compute per-feature drift statistics vs acquisition order.

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
        Minimum number of non-NaN points required to compute stats.

    Returns
    -------
    pandas.DataFrame
        Rows: features; columns with rho, p, slope, CI, Cliff's delta and counts.
    """
    img = ensure_numeric(df[image_col])
    valid_img = img.notna()
    df = df.loc[valid_img].copy()
    img = img[valid_img]

    # Early/late thresholds
    q_low = img.quantile(early_frac)
    q_high = img.quantile(1.0 - early_frac)

    records = []
    for feat in feature_cols:
        x = ensure_numeric(df[feat])
        mask = x.notna()
        if mask.sum() < min_points:
            continue
        xv = x[mask].to_numpy()
        iv = img[mask].to_numpy()

        # Spearman
        rho, pval = spearmanr(iv, xv)

        # Theil–Sen slope (robust)
        # theilslopes returns slope, intercept, lo_slope, up_slope
        slope, _, lo_slope, up_slope = theilslopes(y=xv, x=iv)

        # Early vs late Cliff's delta
        early = xv[iv <= q_low]
        late = xv[iv >= q_high]
        cd = np.nan
        if early.size > 0 and late.size > 0:
            cd = cliffs_delta(early, late)

        records.append({
            "feature": feat,
            "n_objects": int(xv.size),
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "theil_sen_slope": float(slope),
            "theil_sen_ci_low": float(lo_slope),
            "theil_sen_ci_high": float(up_slope),
            "early_median": float(np.median(early)) if early.size else np.nan,
            "late_median": float(np.median(late)) if late.size else np.nan,
            "cliffs_delta": float(cd),
        })
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    # FDR across all features tested
    reject, q = benjamini_hochberg(out["spearman_p"].to_numpy(), alpha=0.05)
    out["spearman_q"] = q
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
    candidates = sorted(input_dir.rglob("*Image*.csv.gz"))
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


def parse_well_id(well: str) -> tuple[int, int]:
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


def compute_plate_delta(
    df: pd.DataFrame,
    feature: str,
    image_col: str,
    plate_col: str,
    well_col: str,
    early_frac: float = 0.2,
) -> pd.DataFrame:
    """
    Compute per-plate, per-well early-late median delta for a feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Object-level table with feature, plate, well, image columns.
    feature : str
        Feature to summarise.
    image_col : str
        Acquisition-order column.
    plate_col : str
        Plate identifier column.
    well_col : str
        Well identifier column (e.g., A01).
    early_frac : float
        Fraction for early/late split boundaries.

    Returns
    -------
    pandas.DataFrame
        Columns: plate, well, early_median, late_median, delta
    """
    sub = df[[plate_col, well_col, image_col, feature]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=[plate_col, well_col, "early_median", "late_median", "delta"])

    # Split thresholds *globally* by acquisition order
    q_low = sub[image_col].quantile(early_frac)
    q_high = sub[image_col].quantile(1.0 - early_frac)

    early = sub.loc[sub[image_col] <= q_low]
    late = sub.loc[sub[image_col] >= q_high]

    gcols = [plate_col, well_col]
    e_med = early.groupby(gcols, observed=False)[feature].median().rename("early_median")
    l_med = late.groupby(gcols, observed=False)[feature].median().rename("late_median")
    out = e_med.to_frame().join(l_med, how="outer")
    out["delta"] = out["late_median"] - out["early_median"]
    out.reset_index(inplace=True)
    return out


def plot_plate_heatmap(
    plate_df: pd.DataFrame,
    plate: str,
    well_col: str,
    value_col: str,
    out_png: Path,
) -> None:
    """
    Render a plate heatmap (imshow) for the given plate.

    Parameters
    ----------
    plate_df : pandas.DataFrame
        Rows for a single plate with well IDs and value column.
    plate : str
        Plate identifier (used in title/filename).
    well_col : str
        Well ID column (e.g., A01).
    value_col : str
        Column in plate_df holding the value to plot (e.g., 'delta').
    out_png : pathlib.Path
        Output path for PNG.

    Returns
    -------
    None
    """
    if plate_df.empty:
        return

    n_rows, n_cols = infer_plate_shape(plate_df[well_col])
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)

    for _, row in plate_df.iterrows():
        r, c = parse_well_id(str(row[well_col]))
        if pd.isna(r) or pd.isna(c) or r >= n_rows or c >= n_cols:
            continue
        grid[int(r), int(c)] = row[value_col]

    fig = plt.figure(figsize=(max(6, n_cols * 0.4), max(4, n_rows * 0.4)))
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, aspect="auto")  # use default colormap
    ax.set_title(f"Plate {plate}: {value_col} per well")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(i + 1) for i in range(n_cols)], rotation=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([chr(ord("A") + i) for i in range(n_rows)])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
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

def plot_feature_vs_time_hexbin(
    df: pd.DataFrame,
    image_col: str,
    feature: str,
    out_png: Path,
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
    out_png : pathlib.Path
        Output PNG path.
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
    fig.savefig(out_png, dpi=180)
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
    logger: logging.Logger,
) -> None:


    """
    Execute QC for a specific compartment across its files.

    Parameters
    ----------
    files : list[pathlib.Path]
        Files belonging to this compartment.
    compartment : str
        Compartment name ("Cell", "Cytoplasm", "Nuclei").
    out_dir : pathlib.Path
        Root output directory. Compartment subfolder will be created inside.
    image_col : str
        Acquisition-order column name.
    feature_list : Optional[list[str]]
        If provided, the exact feature columns to evaluate; otherwise auto-pick.
    plot_features : Optional[list[str]]
        Optional explicit list of features to plot; otherwise pick a small auto-panel.
    bin_size : int
        Plotting helper not used in hexbin (kept for future binned violin option).
    max_points_plot : int
        Maximum number of points for plotting (random subsample).
    controls_query : Optional[str]
        Pandas query string to subset controls-only analysis.
    logger : logging.Logger
        Logger.

    Returns
    -------
    None
    """
    comp_dir = out_dir / compartment
    comp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s: %d file(s) ===", compartment, len(files))

    # Load and concatenate all files for this compartment
    frames = []
    for f in files:
        df = read_object_table(f, logger=logger)
        if image_col not in df.columns:
            logger.warning("[%s] missing '%s'; skipping file: %s", compartment, image_col, f)
            continue
        frames.append(df)
    if not frames:
        logger.warning("No valid tables for %s; skipping.", compartment)
        return
    data = pd.concat(frames, axis=0, ignore_index=True)
    n_rows, n_cols = data.shape
    logger.info("%s: concatenated table shape = %s", compartment, (n_rows, n_cols))

        # Merge Image metadata (Plate/Well/controls) if available
    if image_meta is not None:
        if "ImageNumber" not in data.columns:
            logger.warning("%s: object table missing ImageNumber; cannot merge metadata.", compartment)
        else:
            before = data.shape[1]
            data = data.merge(image_meta, on="ImageNumber", how="left", validate="many_to_one")
            logger.info("%s: merged Image metadata columns: +%d cols (now %d).",
                        compartment, data.shape[1] - before, data.shape[1])


    # Select features
    if feature_list is None:
        feats = feature_candidates(data.columns, compartment=compartment)
        logger.info("%s: auto-selected %d feature(s) for drift testing.", compartment, len(feats))
    else:
        feats = [c for c in feature_list if c in data.columns]
        logger.info("%s: user-specified %d/%d feature(s) present.", compartment, len(feats), len(feature_list))
    if not feats:
        logger.warning("%s: no features selected; skipping.", compartment)
        return

    # Keep only numeric features to avoid dtype issues
    num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(data[c])]
    if not num_feats:
        logger.warning("%s: no numeric features after filtering; skipping.", compartment)
        return
    
    feature_cols=num_feats

    # Compute full drift stats
    stats = compute_drift_stats(
        df=data,
        feature_cols=feats,
        image_col=image_col,
        early_frac=0.2,
        min_points=2000,
    )
    stats_path = comp_dir / "qc_feature_drift_raw.tsv"
    stats.to_csv(stats_path, sep="\t", index=False)
    logger.info("%s: wrote drift stats -> %s", compartment, stats_path)

    # Controls-only analysis (optional)
    if controls_query:
        try:
            ctrl = data.query(controls_query)
            if ctrl.shape[0] >= 2000:
                stats_ctrl = compute_drift_stats(
                    df=ctrl,
                    feature_cols=feats,
                    image_col=image_col,
                    early_frac=0.2,
                    min_points=500,
                )
                stats_ctrl_path = comp_dir / "qc_feature_drift_controls.tsv"
                stats_ctrl.to_csv(stats_ctrl_path, sep="\t", index=False)
                logger.info("%s: wrote control-only drift stats -> %s", compartment, stats_ctrl_path)
            else:
                logger.warning("%s: controls subset too small (%d rows); skipping control-only stats.",
                               compartment, ctrl.shape[0])
        except Exception as e:
            logger.error("%s: controls query failed (%s): %s", compartment, controls_query, e)

    # Per-image summary
    per_img = per_image_summary(data, feature_cols=feats[:20], image_col=image_col)
    per_img_path = comp_dir / "qc_per_image_summary.tsv"
    per_img.to_csv(per_img_path, sep="\t", index=False)
    logger.info("%s: wrote per-image summary -> %s", compartment, per_img_path)

    # Plots: small set (explicit or auto-pick a handful)
    if plot_features:
        pfeats = [c for c in plot_features if c in data.columns]
    else:
        # Auto-pick: top 8 drifting features by q-value (or fallback to first 8)
        if not stats.empty:
            pfeats = stats.nsmallest(8, "spearman_q")["feature"].tolist()
        else:
            pfeats = feats[:8]
    plots_dir = comp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for feat in pfeats:
        out_png = plots_dir / f"{feat}.hexbin.pdf"
        plot_feature_vs_time_hexbin(
            df=data,
            image_col=image_col,
            feature=feat,
            out_png=out_png,
            rolling_window=301,
            max_points_plot=max_points_plot,
        )
        # Determine features for heatmaps
    if heatmap_features:
        heat_feats = [s.strip() for s in heatmap_features.split(",") if s.strip() and s.strip() in data.columns]
    else:
        heat_feats = stats.nsmallest(3, "spearman_q")["feature"].tolist() if not stats.empty else feats[:3]

    # Compute and save per-plate early-late deltas + heatmaps
    if plate_col in data.columns and well_col in data.columns:
        hm_dir = comp_dir / "heatmaps"
        hm_dir.mkdir(exist_ok=True)
        for feat in heat_feats:
            delta_df = compute_plate_delta(
                df=data,
                feature=feat,
                image_col=image_col,
                plate_col=plate_col,
                well_col=well_col,
                early_frac=0.2,
            )
            delta_path = hm_dir / f"{feat}.plate_delta.tsv"
            delta_df.to_csv(delta_path, sep="\t", index=False)

            # One PNG per plate
            for plate_id, plate_block in delta_df.groupby(plate_col, observed=False):
                png = hm_dir / f"{feat}.plate_{plate_id}.pdf"
                plot_plate_heatmap(
                    plate_df=plate_block,
                    plate=str(plate_id),
                    well_col=well_col,
                    value_col="delta",
                    out_png=png,
                )
        logger.info("%s: wrote plate heatmaps for %d feature(s) -> %s", compartment, len(heat_feats), hm_dir)
    else:
        logger.warning("%s: plate/well columns not found (%s, %s); skipping heatmaps.",
                       compartment, plate_col, well_col)

    logger.info("%s: wrote %d plot(s) to %s", compartment, len(pfeats), plots_dir)


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

    image_path = find_image_table(input_dir)
    if image_path is None:
        logger.warning("No Image table found under %s. Proceeding without merged metadata.", input_dir)
    else:
        logger.info("Using Image table: %s", image_path)
        image_meta = load_image_metadata(image_path, logger=logger)


    # Normalise a bare 'DMSO' controls_query into a valid boolean expression
    if args.controls_query and args.controls_query.strip().upper() == "DMSO":
        args.controls_query = '(Library == "DMSO") or (cpd_type == "DMSO")'


    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    include_glob = args.include_glob or [
        "*Cell*.csv.gz",
        "*Cytoplasm*.csv.gz",
        "*Nuclei*.csv.gz",
    ]
    files_by_comp = list_compartment_files(input_dir=input_dir, include_glob=include_glob)
    if not files_by_comp:
        logger.error("No matching compartment files found under %s with patterns: %s",
                     input_dir, include_glob)
        return

    # Optional explicit feature list (read from file)
    feature_list: Optional[list[str]] = None
    if args.feature_list_tsv:
        fl_path = Path(args.feature_list_tsv)
        feature_list = pd.read_csv(fl_path, sep="\t", header=None)[0].tolist()

    plot_features: Optional[list[str]] = None
    if args.plot_features:
        plot_features = [s.strip() for s in args.plot_features.split(",") if s.strip()]

    # Plate heatmaps of early-late delta for selected features
    # Decide which features to heatmap: explicit list, else top 3 drift features
    heat_feats = []
    if plot_features and (len(plot_features) > 0):
        heat_feats = plot_features[:3]
    elif not stats.empty:
        heat_feats = stats.nsmallest(3, "spearman_q")["feature"].tolist()

    # Allow explicit override via --heatmap_features (comma-separated)
    # This requires passing args.heatmap_features into run_for_compartment or grabbing from closure



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
            image_meta=image_meta if image_path else None,
            logger=logger,
        )

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stand-alone per-compartment drift QC for CellProfiler object files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing CellProfiler object CSVs (e.g., *Cell.csv.gz, *Cytoplasm.csv.gz, *Nuclei.csv.gz).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write QC outputs (TSVs and plots).",
    )
    parser.add_argument(
        "--image_col",
        default="ImageNumber",
        help="Column indicating acquisition order (default: ImageNumber).",
    )
    parser.add_argument(
        "--include_glob",
        nargs="+",
        default=None,
        help="Glob(s) to include (default: '*Cell*.csv.gz' '*Cytoplasm*.csv.gz' '*Nuclei*.csv.gz').",
    )
    parser.add_argument(
        "--feature_list_tsv",
        default=None,
        help="Optional TSV (one column, no header) listing exact feature columns to test.",
    )
    parser.add_argument(
        "--plot_features",
        default=None,
        help="Optional comma-separated list of features to plot. Defaults to top drifters.",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=150,
        help="Reserved for future binned-distribution plots (default: 150).",
    )
    parser.add_argument(
        "--max_points_plot",
        type=int,
        default=200000,
        help="Maximum random-subsampled points for plotting per feature (default: 200000).",
    )
    parser.add_argument(
        "--controls_query",
        default='(Library == "DMSO") or (cpd_type == "DMSO")',
        help="Pandas query string to subset control objects (e.g., 'Library == \"DMSO\"').",
    )


    parser.add_argument(
        "--plate_col",
        default="Plate_Metadata",
        help="Column holding plate identifier (default: Plate_Metadata).",
    )
    parser.add_argument(
        "--well_col",
        default="Well_Metadata",
        help="Column holding well ID like A01, H12 (default: Well_Metadata).",
    )
    parser.add_argument(
        "--heatmap_features",
        default=None,
        help="Comma-separated list of features to plot as plate heatmaps (early-late median delta).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    main(parser.parse_args())
