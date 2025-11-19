#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-plate acrosome reaction analysis with dose–response curves.

Overview
--------
This script performs the following steps:

1. Discover and load CellProfiler tables from a given directory:
       - *Image.csv   (per-image measurements)
       - *Acrosome.csv (object-level acrosome features; currently unused)

   The Image table is expected to contain:
       - Plate_Metadata
       - Well_Metadata
       - ImageNumber
       - an AR count column (default 'AR_count')
       - a total cell count column (default 'Cell_count')

2. Load library metadata (PTOD) with:
       - 'DDD'              -> cpd_id
       - 'PTODCONCVALUE'    -> conc
       - 'Plate ID'
       - Row/Col and/or Well

3. Load a controls table with:
       - 'source_plate_barcode' -> plate_id
       - 'source_well'          -> well_id
       - 'cpd_id', 'drugname', 'info' (DMSO identified here)

4. Harmonise well identifiers to 'A01' format and merge everything so each
   image has cpd_id, conc and cpd_type in {'DMSO', 'POS', 'TEST'}.

5. Aggregate per well:
       - total_cells (sum of Cell_count over images)
       - total_AR_cells (sum of AR_count over images)
       - AR_pct_well

6. Apply QC:
       - drop wells with total_cells < min_cells_per_well
       - among DMSO wells, flag AR% outliers using an IQR rule
       - within each (cpd_id, conc) group, flag replicate wells as AR% outliers
         using a MAD-based rule

7. Pool counts per (cpd_id, conc) and run two-sided Fisher's exact tests
   versus pooled DMSO controls:
       odds ratio, p-value, delta AR%, FDR-adjusted q-value.

8. Optionally fit a four-parameter logistic dose–response curve per compound
   and estimate EC50.

9. Write outputs:
       - acrosome_per_well_qc.tsv
       - acrosome_fisher_per_dose.tsv
       - acrosome_drc_fits.tsv (if enabled)
       - plate heatmap, volcano, top-hit barplot, dose–response plots
       - HTML report linking everything.

Notes
-----
- The script is written for a *single assay plate* at a time.
- All comparisons are against pooled DMSO wells on that plate.
- P-values are from two-sided Fisher's exact tests; decreased AR and increased
  AR are both detected.
"""

from __future__ import annotations

import argparse
import logging
import math
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import fisher_exact


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging and argument parsing
# ---------------------------------------------------------------------------


def configure_logging(*, verbosity: int) -> None:
    """
    Configure the root logger for the script.

    Parameters
    ----------
    verbosity : int
        Verbosity level (0=warning, 1=info, 2=debug).
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.debug("Logging configured with level %s", logging.getLevelName(level))

    cmd_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    LOGGER.info("Command line: %s", cmd_line)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Single-plate acrosome reaction analysis using per-image counts, "
            "with Fisher tests and dose–response curves."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--cp_dir",
        dest="cp_dir",
        default="raw",
        help=(
            "Directory containing CellProfiler output tables, expected to "
            "include '*_Image.csv' and '*_Acrosome.csv'."
        ),
    )
    parser.add_argument(
        "--library_metadata",
        dest="library_metadata",
        default="sperm_painting_PTOD_metadata.tsv",
        help=(
            "Plate map / PTOD metadata file with DDD, PTODCONCVALUE, Plate ID, "
            "Row/Col and/or Well."
        ),
    )
    parser.add_argument(
        "--controls",
        dest="controls",
        default="sperm_painting_controls_20221019.csv",
        help="Controls file listing DMSO and other controls.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default="acrosome_results",
        help="Directory to write TSVs, plots and HTML report.",
    )
    parser.add_argument(
        "--image_plate_col",
        dest="image_plate_col",
        default="Plate_Metadata",
        help="Plate identifier column in the Image table.",
    )
    parser.add_argument(
        "--image_well_col",
        dest="image_well_col",
        default="Well_Metadata",
        help="Well identifier column in the Image table.",
    )
    parser.add_argument(
        "--image_cell_count_col",
        dest="image_cell_count_col",
        default="Count_SpermCells",
        help="Total cell count column in the Image table:Count_SpermCells.",
    )
    parser.add_argument(
        "--image_ar_count_col",
        dest="image_ar_count_col",
        default="Count_Acrosome",
        help="Acrosome-reacted cell count column in the Image table:Count_Acrosome.",
    )
    parser.add_argument(
        "--min_cells_per_well",
        dest="min_cells_per_well",
        type=int,
        default=50,
        help="Minimum cell count per well for inclusion.",
    )
    parser.add_argument(
        "--dmso_iqr_multiplier",
        dest="dmso_iqr_multiplier",
        type=float,
        default=3.0,
        help=(
            "IQR multiplier for defining AR%% outliers amongst DMSO wells. "
            "Only DMSO wells are auto-dropped as AR%% outliers."
        ),
    )
    parser.add_argument(
        "--replicate_mad_multiplier",
        dest="replicate_mad_multiplier",
        type=float,
        default=5.0,
        help=(
            "MAD multiplier for flagging within-compound–dose replicate "
            "AR%% outliers."
        ),
    )
    parser.add_argument(
        "--fit_dose_response",
        dest="fit_dose_response",
        action="store_true",
        help="If set, attempt four-parameter logistic fits and EC50 estimation.",
    )
    parser.add_argument(
        "--min_drc_points",
        dest="min_drc_points",
        type=int,
        default=3,
        help=(
            "Minimum number of distinct doses required to fit a "
            "dose–response curve for a compound."
        ),
    )
    parser.add_argument(
        "--verbosity",
        dest="verbosity",
        type=int,
        default=1,
        help="Verbosity level: 0=warnings, 1=info, 2=debug.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Well ID normalisation and metadata loading
# ---------------------------------------------------------------------------


def standardise_well_from_row_col(
    *,
    row: str,
    col: int,
) -> str:
    """
    Build a CellProfiler-style well ID from row and column.

    Examples
    --------
    row='A', col=1   -> 'A01'
    row='B', col=12  -> 'B12'

    Parameters
    ----------
    row : str
        Row letter, for example 'A', 'B'.
    col : int
        Column number, 1-based.

    Returns
    -------
    str
        Well identifier in 'A01' format.
    """
    row = str(row).strip().upper()
    col_int = int(col)
    return f"{row}{col_int:02d}"


def standardise_well_string(
    *,
    well: str,
) -> str:
    """
    Normalise a raw well string to 'A01' format where possible.

    This handles common cases such as 'A1', 'A01', 'A001'.

    Parameters
    ----------
    well : str
        Raw well identifier.

    Returns
    -------
    str
        Normalised well identifier, or the original string if parsing fails.
    """
    s = str(well).strip().upper()
    if not s:
        return s

    row = s[0]
    digits = "".join(ch for ch in s[1:] if ch.isdigit())
    if not digits:
        return s

    col_int = int(digits)
    return f"{row}{col_int:02d}"


def load_compound_metadata(
    *,
    metadata_path: Path,
) -> pd.DataFrame:
    """
    Load library metadata for compounds and doses (PTOD).

    The input is expected to contain at least:
        - 'DDD'              -> cpd_id
        - 'PTODCONCVALUE'    -> conc
        - 'Plate ID'
        - Row/Col and/or Well

    Parameters
    ----------
    metadata_path : Path
        Path to the PTOD metadata file.

    Returns
    -------
    pandas.DataFrame
        Metadata with columns:
        - plate_id
        - well_id
        - cpd_id
        - conc
        plus any additional chemistry columns.
    """
    LOGGER.info("Loading library metadata from '%s'", metadata_path)
    df = pd.read_csv(metadata_path, sep=None, engine="python")
    LOGGER.info("Loaded library metadata with %d rows and %d columns", *df.shape)

    # Build well_id
    if {"Row", "Col"}.issubset(df.columns):
        LOGGER.info("Building well_id from 'Row' and 'Col' columns.")
        df["well_id"] = [
            standardise_well_from_row_col(row=r, col=c)
            for r, c in zip(df["Row"], df["Col"])
        ]
    elif "Well" in df.columns:
        LOGGER.info("Normalising well_id from 'Well' column.")
        df["well_id"] = df["Well"].map(lambda w: standardise_well_string(well=w))
    else:
        raise KeyError(
            "Metadata must contain either ('Row', 'Col') or 'Well' columns "
            "to derive well identifiers."
        )

    rename_map = {
        "Plate ID": "plate_id",
        "DDD": "cpd_id",
        "PTODCONCVALUE": "conc",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
        else:
            LOGGER.warning(
                "Expected metadata column '%s' not found; '%s' will be missing.",
                old,
                new,
            )

    return df


def load_controls(
    *,
    controls_path: Path,
) -> pd.DataFrame:
    """
    Load controls table and identify DMSO and other controls.

    The input is expected to contain:
        - 'source_plate_barcode' -> plate_id
        - 'source_well'          -> well_id
        - 'cpd_id', 'drugname', 'info' (one or more may be present)

    Any row where any of (cpd_id, drugname, info) contains 'dmso'
    (case-insensitive) is labelled as 'DMSO'; all others are 'POS'.

    Parameters
    ----------
    controls_path : Path
        Path to the controls CSV.

    Returns
    -------
    pandas.DataFrame
        Controls with columns:
        - plate_id
        - well_id
        - control_type ('DMSO' or 'POS')
        - cpd_id (if present in the input)
    """
    LOGGER.info("Loading controls from '%s'", controls_path)
    df = pd.read_csv(controls_path, sep=None, engine="python")
    LOGGER.info("Loaded controls table with %d rows and %d columns", *df.shape)

    col_map = {
        "source_plate_barcode": "plate_id",
        "source_well": "well_raw",
    }
    df.rename(columns=col_map, inplace=True)

    df["well_id"] = df["well_raw"].map(lambda w: standardise_well_string(well=w))

    def _row_is_dmso(row: pd.Series) -> bool:
        candidates: List[str] = []
        for col in ("cpd_id", "drugname", "info"):
            if col in row and pd.notna(row[col]):
                candidates.append(str(row[col]).lower())
        return any("dmso" in text for text in candidates)

    df["control_type"] = np.where(
        df.apply(_row_is_dmso, axis=1),
        "DMSO",
        "POS",
    )

    keep_cols = ["plate_id", "well_id", "control_type"]
    if "cpd_id" in df.columns:
        keep_cols.append("cpd_id")

    df = df[keep_cols].copy()
    LOGGER.info(
        "Controls summary: %d DMSO wells, %d other controls.",
        (df["control_type"] == "DMSO").sum(),
        (df["control_type"] == "POS").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# CellProfiler discovery and per-image counts
# ---------------------------------------------------------------------------


def discover_cp_files(
    *,
    cp_dir: Path,
) -> Tuple[Path, Path | None]:
    """
    Discover CellProfiler Image and Acrosome tables in a directory.

    Parameters
    ----------
    cp_dir : Path
        Directory containing CellProfiler *.csv outputs.

    Returns
    -------
    tuple
        (image_path, acrosome_path_or_none)

    Raises
    ------
    FileNotFoundError
        If no Image table is found.
    """
    image_files = sorted(cp_dir.glob("*_Image.csv"))
    acrosome_files = sorted(cp_dir.glob("*_Acrosome.csv"))

    if not image_files:
        raise FileNotFoundError(
            f"No '*_Image.csv' file found in directory '{cp_dir}'."
        )
    if len(image_files) > 1:
        LOGGER.warning(
            "Multiple Image tables found; using the first: %s", image_files[0]
        )

    image_path = image_files[0]
    acrosome_path = acrosome_files[0] if acrosome_files else None

    LOGGER.info("Using Image table: '%s'", image_path)
    if acrosome_path is not None:
        LOGGER.info("Found Acrosome table: '%s' (currently unused).", acrosome_path)
    else:
        LOGGER.info("No '*_Acrosome.csv' file found; continuing without it.")

    return image_path, acrosome_path


def load_image_counts(
    *,
    image_path: Path,
    plate_col: str,
    well_col: str,
    cell_count_col: str,
    ar_count_col: str,
) -> pd.DataFrame:
    """
    Load per-image counts (total cells and AR-reacted cells) from Image table.

    Parameters
    ----------
    image_path : Path
        Path to the CellProfiler Image table.
    plate_col : str
        Name of the plate metadata column.
    well_col : str
        Name of the well metadata column.
    cell_count_col : str
        Name of the total cell count column.
    ar_count_col : str
        Name of the acrosome-reacted cell count column.

    Returns
    -------
    pandas.DataFrame
        Per-image table with columns:
        - ImageNumber
        - plate_id
        - well_id
        - Cell_count
        - AR_count
    """
    LOGGER.info("Loading Image table from '%s'", image_path)
    df = pd.read_csv(image_path, sep=None, engine="python")
    LOGGER.info("Loaded Image table with %d rows and %d columns", *df.shape)

    required = {plate_col, well_col, cell_count_col, ar_count_col, "ImageNumber"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"Image table is missing required columns: {sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "ImageNumber": df["ImageNumber"].astype(int),
            "plate_id": df[plate_col].astype(str),
            "well_id": df[well_col].map(lambda w: standardise_well_string(well=w)),
            "Cell_count": df[cell_count_col].astype(int),
            "AR_count": df[ar_count_col].astype(int),
        }
    )

    LOGGER.debug(
        "Example rows from per-image counts:\n%s",
        out.head().to_string(index=False),
    )
    return out


def merge_image_with_metadata(
    *,
    df_image: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_ctrl: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge per-image counts with library metadata and control annotations.

    Parameters
    ----------
    df_image : pandas.DataFrame
        Per-image counts with plate_id, well_id, Cell_count, AR_count.
    df_meta : pandas.DataFrame
        Library metadata with plate_id, well_id, cpd_id, conc.
    df_ctrl : pandas.DataFrame or None
        Controls table with plate_id, well_id and control_type.

    Returns
    -------
    pandas.DataFrame
        Per-image table with added:
        - cpd_id
        - conc
        - cpd_type ('DMSO', 'POS', 'TEST').
    """
    df = df_image.copy()

    LOGGER.info("Merging per-image counts with library metadata.")
    df = df.merge(
        df_meta,
        how="left",
        on=["plate_id", "well_id"],
        suffixes=("", "_lib"),
    )

    if df["cpd_id"].isna().any():
        n_missing = df["cpd_id"].isna().sum()
        LOGGER.warning(
            "After merging library metadata, %d image rows have missing cpd_id.",
            n_missing,
        )

    df["cpd_type"] = "TEST"
    if df_ctrl is not None and not df_ctrl.empty:
        LOGGER.info("Merging control annotations into per-image table.")
        df = df.merge(
            df_ctrl,
            how="left",
            on=["well_id"],  # "plate_id",  WE SHOULD LOOK AT THIS!!!
            suffixes=("", "_ctrl"),
        )
        df["cpd_type"] = df["control_type"].fillna(df["cpd_type"])
        df.drop(columns=["control_type"], inplace=True, errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Per-well aggregation and QC
# ---------------------------------------------------------------------------


def aggregate_per_well_from_counts(
    *,
    df_image: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-image counts into per-well AR counts and percentages.

    Parameters
    ----------
    df_image : pandas.DataFrame
        Per-image table containing:
        - plate_id
        - well_id
        - cpd_id
        - cpd_type
        - conc
        - Cell_count
        - AR_count

    Returns
    -------
    pandas.DataFrame
        Per-well summary with columns:
        - plate_id
        - well_id
        - cpd_id
        - cpd_type
        - conc
        - n_images
        - total_cells
        - total_AR_cells
        - AR_pct_well
    """
    required = {
        "plate_id",
        "well_id",
        "cpd_id",
        "cpd_type",
        "conc",
        "Cell_count",
        "AR_count",
    }
    missing = required.difference(df_image.columns)
    if missing:
        raise KeyError(
            "Missing required columns for per-well aggregation: "
            f"{sorted(missing)}"
        )

    LOGGER.info("Aggregating per-image counts into per-well summaries.")

    group_cols = ["plate_id", "well_id", "cpd_id", "cpd_type", "conc"]
    agg = (
        df_image.groupby(group_cols, dropna=False)
        .agg(
            n_images=("ImageNumber", "nunique"),
            total_cells=("Cell_count", "sum"),
            total_AR_cells=("AR_count", "sum"),
        )
        .reset_index()
    )
    agg["AR_pct_well"] = agg["total_AR_cells"] / agg["total_cells"] * 100.0

    LOGGER.info(
        "Per-well aggregation produced %d wells. Example:\n%s",
        agg.shape[0],
        agg.head().to_string(index=False),
    )
    return agg


def flag_wells_for_qc(
    *,
    df_well: pd.DataFrame,
    min_cells_per_well: int,
    dmso_iqr_multiplier: float,
    replicate_mad_multiplier: float,
) -> pd.DataFrame:
    """
    Apply QC rules to per-well summaries.

    QC rules
    --------
    1. Drop any well with total_cells < min_cells_per_well.

    2. Among DMSO wells only, flag AR_pct_well outliers using an IQR rule:
           outlier if AR_pct < Q1 - k*IQR or AR_pct > Q3 + k*IQR
       where k = dmso_iqr_multiplier.

    3. Within each compound–dose group (cpd_id, conc), flag replicate wells
       that are AR% outliers using a median absolute deviation (MAD) rule:
           outlier if |AR_pct - median| > k * MAD_scaled,
       where k = replicate_mad_multiplier and MAD_scaled = 1.4826 * MAD.

    Parameters
    ----------
    df_well : pandas.DataFrame
        Per-well summary table.
    min_cells_per_well : int
        Minimum number of cells required for a well to be considered reliable.
    dmso_iqr_multiplier : float
        IQR multiplier for defining DMSO AR% outliers.
    replicate_mad_multiplier : float
        MAD multiplier for defining within-compound–dose outliers.

    Returns
    -------
    pandas.DataFrame
        Copy of df_well with additional columns:
        - is_dmso : bool
        - qc_reason : comma-separated reasons (empty if none)
        - qc_keep : bool (True if well passes QC).
    """
    df = df_well.copy()
    df["is_dmso"] = df["cpd_type"] == "DMSO"
    df["qc_reasons"] = [[] for _ in range(df.shape[0])]

    # Rule 1: low cell count
    low_mask = df["total_cells"] < min_cells_per_well
    for idx in df.index[low_mask]:
        df.at[idx, "qc_reasons"].append("LOW_CELL_COUNT")
    LOGGER.info("Wells failing low cell count: %d", low_mask.sum())

    # Rule 2: DMSO AR% outliers
    dmso_df = df[df["is_dmso"] & ~low_mask]
    if dmso_df.empty:
        LOGGER.warning("No DMSO wells available for AR%% outlier QC.")
    else:
        q1 = dmso_df["AR_pct_well"].quantile(0.25)
        q3 = dmso_df["AR_pct_well"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - dmso_iqr_multiplier * iqr
        upper = q3 + dmso_iqr_multiplier * iqr
        LOGGER.info(
            "DMSO AR%% IQR thresholds: Q1=%.3f, Q3=%.3f, IQR=%.3f, "
            "lower=%.3f, upper=%.3f",
            q1,
            q3,
            iqr,
            lower,
            upper,
        )

        dmso_out_mask = df["is_dmso"] & ~low_mask & (
            (df["AR_pct_well"] < lower) | (df["AR_pct_well"] > upper)
        )
        for idx in df.index[dmso_out_mask]:
            df.at[idx, "qc_reasons"].append("DMSO_AR_OUTLIER")
        LOGGER.info("DMSO AR%% outlier wells: %d", dmso_out_mask.sum())

    # Rule 3: within compound–dose MAD-based replicate QC
    LOGGER.info("Applying replicate QC within (cpd_id, conc) groups.")
    for (cpd_id, conc), group in df.groupby(["cpd_id", "conc"]):
        if group.shape[0] <= 2:
            continue
        values = group["AR_pct_well"].values
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            continue
        mad_scaled = 1.4826 * mad
        threshold = replicate_mad_multiplier * mad_scaled
        out_mask_local = np.abs(values - median) > threshold
        if out_mask_local.any():
            LOGGER.debug(
                "Replicate outliers for cpd_id=%s, conc=%s: %d of %d",
                cpd_id,
                conc,
                out_mask_local.sum(),
                group.shape[0],
            )
            for idx in group.index[out_mask_local]:
                df.at[idx, "qc_reasons"].append("REPLICATE_AR_OUTLIER")

    df["qc_reason"] = df["qc_reasons"].apply(
        lambda reasons: ",".join(sorted(set(reasons)))
    )
    df["qc_keep"] = df["qc_reason"] == ""
    LOGGER.info(
        "QC summary: %d wells kept, %d wells dropped.",
        df["qc_keep"].sum(),
        (~df["qc_keep"]).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Fisher tests and FDR
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    *,
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Apply Benjamini–Hochberg FDR correction to an array of p-values.

    Parameters
    ----------
    p_values : numpy.ndarray
        Array of raw p-values.

    Returns
    -------
    numpy.ndarray
        Array of adjusted p-values (q-values).
    """
    n = len(p_values)
    if n == 0:
        return p_values

    order = np.argsort(p_values)
    ranked_p = p_values[order]
    ranks = np.arange(1, n + 1, dtype=float)

    bh_values = ranked_p * n / ranks
    bh_values = np.minimum.accumulate(bh_values[::-1])[::-1]
    bh_values = np.clip(bh_values, 0.0, 1.0)

    q_values = np.empty_like(bh_values)
    q_values[order] = bh_values
    return q_values


def run_fisher_per_compound_dose(
    *,
    df_well_qc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run Fisher's exact test per compound–dose versus pooled DMSO controls.

    Parameters
    ----------
    df_well_qc : pandas.DataFrame
        Per-well table with QC flags applied and 'qc_keep' column.

    Returns
    -------
    pandas.DataFrame
        Per-compound–dose table with:
        - plate_id
        - cpd_id
        - conc
        - cpd_type
        - n_wells
        - total_cells_compound
        - total_AR_compound
        - AR_pct_compound
        - total_cells_dmso
        - total_AR_dmso
        - AR_pct_dmso
        - delta_AR_pct
        - odds_ratio
        - p_value
        - q_value
    """
    df = df_well_qc[df_well_qc["qc_keep"]].copy()
    if df.empty:
        raise ValueError("No wells remain after QC; cannot run Fisher tests.")

    plates = df["plate_id"].unique()
    if len(plates) != 1:
        LOGGER.warning(
            "Multiple plate_ids found (%s); proceeding but pooling all.",
            plates,
        )

    dmso_mask = df["cpd_type"] == "DMSO"
    dmso_df = df[dmso_mask]
    if dmso_df.empty:
        raise ValueError("No DMSO wells found after QC; cannot perform comparisons.")

    total_cells_dmso = int(dmso_df["total_cells"].sum())
    total_AR_dmso = int(dmso_df["total_AR_cells"].sum())
    AR_pct_dmso = (total_AR_dmso / total_cells_dmso) * 100.0
    LOGGER.info(
        "Pooled DMSO counts: AR=%d, non-AR=%d, AR%%=%.3f",
        total_AR_dmso,
        total_cells_dmso - total_AR_dmso,
        AR_pct_dmso,
    )

    test_df = df[~dmso_mask].copy()
    if test_df.empty:
        raise ValueError("No non-DMSO wells found after QC.")

    combos = (
        test_df[["cpd_id", "conc", "cpd_type"]]
        .drop_duplicates()
        .sort_values(["cpd_id", "conc"])
    )

    results: List[Dict[str, object]] = []
    p_vals: List[float] = []

    for _, combo in combos.iterrows():
        cpd_id = combo["cpd_id"]
        conc = combo["conc"]
        cpd_type = combo["cpd_type"]

        subset = test_df[(test_df["cpd_id"] == cpd_id) & (test_df["conc"] == conc)]
        if subset.empty:
            continue

        total_cells_compound = int(subset["total_cells"].sum())
        total_AR_compound = int(subset["total_AR_cells"].sum())
        AR_pct_compound = (total_AR_compound / total_cells_compound) * 100.0
        delta_AR_pct = AR_pct_compound - AR_pct_dmso

        table = np.array(
            [
                [total_AR_compound, total_cells_compound - total_AR_compound],
                [total_AR_dmso, total_cells_dmso - total_AR_dmso],
            ]
        )

        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        p_vals.append(p_value)

        results.append(
            {
                "plate_id": subset["plate_id"].iloc[0],
                "cpd_id": cpd_id,
                "conc": conc,
                "cpd_type": cpd_type,
                "n_wells": int(subset.shape[0]),
                "total_cells_compound": total_cells_compound,
                "total_AR_compound": total_AR_compound,
                "AR_pct_compound": AR_pct_compound,
                "total_cells_dmso": total_cells_dmso,
                "total_AR_dmso": total_AR_dmso,
                "AR_pct_dmso": AR_pct_dmso,
                "delta_AR_pct": delta_AR_pct,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No compound–dose combinations to test.")

    q_vals = benjamini_hochberg(p_values=np.asarray(p_vals, dtype=float))
    results_df["q_value"] = q_vals

    LOGGER.info(
        "Fisher testing completed for %d compound–dose combinations.",
        results_df.shape[0],
    )
    return results_df


# ---------------------------------------------------------------------------
# Dose–response fitting (4-parameter logistic)
# ---------------------------------------------------------------------------


def four_param_logistic(
    conc: np.ndarray,
    bottom: float,
    top: float,
    log_ec50: float,
    hill: float,
) -> np.ndarray:
    """
    Four-parameter logistic function on log10 concentration.

    Parameters
    ----------
    conc : numpy.ndarray
        Concentrations (strictly positive).
    bottom : float
        Response at very low concentration.
    top : float
        Response at very high concentration.
    log_ec50 : float
        log10(EC50).
    hill : float
        Hill slope (steepness).

    Returns
    -------
    numpy.ndarray
        Response values at the given concentrations.
    """
    x = np.log10(conc)
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((log_ec50 - x) * hill))


def fit_dose_response_per_compound(
    *,
    fisher_df: pd.DataFrame,
    min_points: int,
) -> pd.DataFrame:
    """
    Fit four-parameter logistic curves per compound and estimate EC50.

    Fits are based on Fisher results aggregated by compound–dose. The
    response is AR_pct_compound; the DMSO AR% is used as an approximate
    baseline and informs initial parameter guesses.

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound–dose Fisher results.
    min_points : int
        Minimum number of distinct doses required to attempt a fit.

    Returns
    -------
    pandas.DataFrame
        One row per compound with:
        - cpd_id
        - n_doses
        - ec50
        - log10_ec50
        - hill
        - bottom
        - top
        - max_delta_AR_pct
        - fit_success (bool)
    """
    LOGGER.info(
        "Fitting four-parameter logistic curves for compounds with at least "
        "%d distinct doses.",
        min_points,
    )

    records: List[Dict[str, object]] = []

    for cpd_id, group in fisher_df.groupby("cpd_id"):
        g = group.copy()
        g = g[g["conc"] > 0].sort_values("conc")
        if g["conc"].nunique() < min_points:
            LOGGER.debug(
                "Skipping compound %s: insufficient distinct dose levels (%d).",
                cpd_id,
                g["conc"].nunique(),
            )
            continue

        conc = g["conc"].values.astype(float)
        resp = g["AR_pct_compound"].values.astype(float)

        dmso_level = float(g["AR_pct_dmso"].iloc[0])
        bottom0 = dmso_level
        top0 = float(resp.max())
        if math.isclose(top0, bottom0):
            top0 = bottom0 + 5.0
        log_ec50_0 = float(np.median(np.log10(conc)))
        hill0 = 1.0

        p0 = [bottom0, top0, log_ec50_0, hill0]
        bounds = (
            [0.0, 0.0, -12.0, 0.1],
            [100.0, 100.0, 2.0, 5.0],
        )

        fit_success = False
        ec50 = np.nan
        log_ec50 = np.nan
        hill = np.nan
        bottom = np.nan
        top = np.nan

        try:
            params, _ = curve_fit(
                four_param_logistic,
                conc,
                resp,
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
            bottom, top, log_ec50, hill = params
            ec50 = 10.0 ** log_ec50
            fit_success = True
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Dose–response fit failed for %s: %s", cpd_id, err)

        max_delta = float((resp - dmso_level).max())

        records.append(
            {
                "cpd_id": cpd_id,
                "n_doses": int(g["conc"].nunique()),
                "ec50": ec50,
                "log10_ec50": log_ec50,
                "hill": hill,
                "bottom": bottom,
                "top": top,
                "max_delta_AR_pct": max_delta,
                "fit_success": bool(fit_success),
            }
        )

    drc_df = pd.DataFrame(records)
    LOGGER.info(
        "Dose–response fitting attempted for %d compounds; %d fits succeeded.",
        drc_df.shape[0],
        drc_df["fit_success"].sum() if not drc_df.empty else 0,
    )
    return drc_df


# ---------------------------------------------------------------------------
# Plotting and HTML report (unchanged structure)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotting and HTML report
# ---------------------------------------------------------------------------


def plot_plate_heatmap(
    *,
    df_well_qc: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot a plate-layout heatmap of AR% per well.

    This function assumes well_id is in 'A01' format; rows are letters,
    columns are integers.

    Parameters
    ----------
    df_well_qc : pandas.DataFrame
        Per-well table (QC-passed wells preferred).
    output_path : Path
        Path to save the PNG file.
    """
    df = df_well_qc.copy()
    if df.empty:
        LOGGER.warning("No wells available for plate heatmap.")
        return

    def _parse_well(w: str) -> Tuple[str, int]:
        s = str(w).strip().upper()
        if not s or len(s) < 2:
            return "", -1
        row = s[0]
        try:
            col = int(s[1:])
        except ValueError:
            return "", -1
        return row, col

    df["well_row"], df["well_col"] = zip(
        *df["well_id"].map(lambda w: _parse_well(w))
    )

    if (df["well_row"] == "").any() or (df["well_col"] < 0).any():
        LOGGER.warning(
            "Could not parse some well IDs into row/column; skipping heatmap."
        )
        return

    rows = sorted(df["well_row"].unique())
    cols = sorted(df["well_col"].unique())
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: j for j, c in enumerate(cols)}

    heat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for _, r in df.iterrows():
        i = row_index[r["well_row"]]
        j = col_index[r["well_col"]]
        heat[i, j] = r["AR_pct_well"]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heat, aspect="auto", origin="upper")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plate_id = str(df["plate_id"].iloc[0])
    ax.set_title(f"Plate {plate_id} – AR% per well (QC-kept)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AR% per well")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved plate heatmap to '%s'", output_path)


def plot_inducing_compounds_barplot(
    *,
    fisher_df: pd.DataFrame,
    output_path: Path,
    q_threshold: float = 0.05,
    min_delta: float = 0.0,
    top_n: int = 30,
) -> None:
    """
    Plot a barplot of compounds that only induce AR (no decreasing doses).

    A compound is included if:
      - For that cpd_id, all delta_AR_pct >= min_delta (no decreases).
      - At least one dose has q_value <= q_threshold and delta_AR_pct > min_delta.

    For each qualifying compound, the bar represents the dose with the
    largest delta_AR_pct that meets the q_threshold criterion.

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound–dose Fisher results.
    output_path : Path
        Path to save the PNG figure.
    q_threshold : float, optional
        Maximum FDR q-value for a dose to be considered significant.
    min_delta : float, optional
        Minimum delta_AR_pct considered as an increase (usually 0.0).
    top_n : int, optional
        Maximum number of compounds to display (sorted by max delta_AR_pct).
    """
    if fisher_df.empty:
        LOGGER.warning(
            "Fisher results table is empty; skipping inducing-compound barplot."
        )
        return

    records: List[Dict[str, object]] = []

    for cpd_id, group in fisher_df.groupby("cpd_id"):
        g = group.copy()

        # 1) Exclude compounds that ever decrease AR below min_delta.
        if (g["delta_AR_pct"] < min_delta).any():
            continue

        # 2) Require at least one significant positive dose.
        sig = g[(g["q_value"] <= q_threshold) & (g["delta_AR_pct"] > min_delta)]
        if sig.empty:
            continue

        # 3) Pick the best (largest) delta_AR_pct among significant doses.
        best = sig.loc[sig["delta_AR_pct"].idxmax()]

        records.append(
            {
                "cpd_id": best["cpd_id"],
                "best_conc": best["conc"],
                "best_delta_AR_pct": best["delta_AR_pct"],
                "best_q_value": best["q_value"],
                "cpd_type": best["cpd_type"],
            }
        )

    if not records:
        LOGGER.info(
            "No compounds met the 'only induce AR' criteria; "
            "no inducing-compound barplot produced."
        )
        return

    df = pd.DataFrame(records)
    df = df.sort_values("best_delta_AR_pct", ascending=True).tail(top_n)

    df["label"] = (
        df["cpd_id"].astype(str)
        + " @ "
        + df["best_conc"].astype(str)
        + " (q="
        + df["best_q_value"].map(lambda x: f"{x:.2g}")
        + ")"
    )

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(df))))
    ax.barh(df["label"], df["best_delta_AR_pct"])

    ax.set_xlabel("Max ΔAR% vs DMSO (significant, non-decreasing compound)")
    ax.set_ylabel("Compound @ best conc (q)")
    ax.set_title(
        f"Compounds that only induce AR (q ≤ {q_threshold}, n={len(df)})"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved inducing-compounds barplot to '%s'", output_path)


def plot_volcano_interactive(
    *,
    fisher_df: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    """
    Plot an interactive volcano of Fisher results with hover metadata.

    The plot is saved as a self-contained HTML file using Plotly. Hovering
    over each point shows compound metadata including:
      - cpd_id
      - conc
      - cpd_type
      - AR_pct_compound
      - AR_pct_dmso
      - delta_AR_pct
      - p_value
      - q_value

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound–dose Fisher results.
    output_path : Path
        Path to save the HTML file.

    Returns
    -------
    Path or None
        Path to the generated HTML file if successful, otherwise None.
    """
    try:
        import plotly.express as px  # type: ignore[import]
    except ImportError:
        LOGGER.warning(
            "Plotly is not installed; skipping interactive volcano plot. "
            "Install plotly to enable this feature."
        )
        return None

    df = fisher_df.copy()

    df["log2_or"] = np.nan
    valid = df["odds_ratio"] > 0
    df.loc[valid, "log2_or"] = np.log2(df.loc[valid, "odds_ratio"])
    df["neg_log10_q"] = -np.log10(df["q_value"].clip(lower=1e-300))

    hover_cols = [
        "cpd_id",
        "conc",
        "AR_pct_compound",
        "AR_pct_dmso",
        "delta_AR_pct",
        "odds_ratio",
        "p_value",
        "q_value",
        "SMILES",
        "Series",
        "Parent Molecular Formula",
        "Parent Molecular Mass",
        "Full Molecular Formula Salt",
    ]

    for col in hover_cols:
        if col not in df.columns:
            LOGGER.warning(
                "Column '%s' missing from Fisher results; "
                "interactive volcano hover will omit it.",
                col,
            )

    fig = px.scatter(
        df,
        x="log2_or",
        y="neg_log10_q",
        hover_name="cpd_id",
        hover_data={col: True for col in hover_cols if col in df.columns},
        labels={
            "log2_or": "log2(odds ratio, compound–dose vs DMSO)",
            "neg_log10_q": "-log10(FDR q-value)",
        },
        title="Acrosome reaction – interactive Fisher volcano plot",
    )

    fig.update_layout(
        template="simple_white",
    )
    fig.add_vline(x=0.0, line_width=1.0, line_dash="dash")

    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
    )
    LOGGER.info("Saved interactive volcano plot to '%s'", output_path)
    return output_path


def plot_volcano(
    *,
    fisher_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot a volcano-style summary of Fisher test results.

    x-axis: log2(odds_ratio)
    y-axis: -log10(q_value)

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound–dose Fisher results.
    output_path : Path
        Path to save the PNG figure.
    """
    df = fisher_df.copy()

    df["log2_or"] = np.nan
    valid = df["odds_ratio"] > 0
    df.loc[valid, "log2_or"] = np.log2(df.loc[valid, "odds_ratio"])
    df["neg_log10_q"] = -np.log10(df["q_value"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["log2_or"], df["neg_log10_q"], s=20, alpha=0.7)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0)

    ax.set_xlabel("log2(odds ratio, compound–dose vs DMSO)")
    ax.set_ylabel("-log10(FDR q-value)")
    ax.set_title("Acrosome reaction – Fisher per compound–dose")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved volcano plot to '%s'", output_path)


def plot_top_delta_barplot(
    *,
    fisher_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """
    Plot a barplot of delta AR% for the top-N compound–dose hits.

    The top hits are chosen by smallest q-value.

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound–dose Fisher results.
    output_path : Path
        Path to save the PNG figure.
    top_n : int, optional
        Number of top combinations to display.
    """
    df = fisher_df.sort_values("q_value", ascending=True).head(top_n).copy()
    if df.empty:
        LOGGER.warning("No entries available for top hits barplot.")
        return

    df["label"] = (
        df["cpd_id"].astype(str) + " @ " + df["conc"].astype(str)
    )
    df = df.sort_values("delta_AR_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(df))))
    ax.barh(df["label"], df["delta_AR_pct"])

    ax.set_xlabel("Delta AR% (compound–dose − DMSO)")
    ax.set_ylabel("Compound @ conc")
    ax.set_title(f"Top {len(df)} compound–dose hits by FDR")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved top hits barplot to '%s'", output_path)


def plot_dose_response_examples(
    *,
    fisher_df: pd.DataFrame,
    drc_df: pd.DataFrame,
    output_dir: Path,
    max_compounds: int = 12,
) -> List[Path]:
    """
    Plot dose–response curves for selected compounds.

    Compounds are selected as the subset with successful fits, ordered by
    smallest EC50 (most potent), up to max_compounds entries.

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound–dose Fisher results.
    drc_df : pandas.DataFrame
        Per-compound dose–response fit summary.
    output_dir : Path
        Directory to save PNG plots.
    max_compounds : int, optional
        Maximum number of compounds to plot.

    Returns
    -------
    list of Path
        Paths to the generated PNG files.
    """
    if drc_df.empty:
        LOGGER.info("No successful dose–response fits; skipping DRC plots.")
        return []

    subset = drc_df[drc_df["fit_success"]].dropna(subset=["ec50"])
    subset = subset.sort_values("ec50").head(max_compounds)
    if subset.empty:
        LOGGER.info("No compounds with finite EC50 for DRC plotting.")
        return []

    paths: List[Path] = []

    for _, row in subset.iterrows():
        cpd_id = row["cpd_id"]
        g = fisher_df[fisher_df["cpd_id"] == cpd_id].copy()
        g = g[g["conc"] > 0].sort_values("conc")
        if g.empty:
            continue

        conc = g["conc"].values.astype(float)
        resp = g["AR_pct_compound"].values.astype(float)

        # Use stored parameters
        bottom = row["bottom"]
        top = row["top"]
        log_ec50 = row["log10_ec50"]
        hill = row["hill"]

        x_plot = np.linspace(conc.min(), conc.max(), 200)
        y_fit = four_param_logistic(x_plot, bottom, top, log_ec50, hill)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(conc, resp, s=30, alpha=0.8, label="Observed")
        ax.plot(x_plot, y_fit, linewidth=1.5, label="4PL fit")

        ax.set_xscale("log")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("AR%")
        ax.set_title(
            f"{cpd_id} – EC50={row['ec50']:.3g}, "
            f"max ΔAR%={row['max_delta_AR_pct']:.1f}"
        )
        ax.legend()

        fig.tight_layout()
        out_path = output_dir / f"drc_{cpd_id}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        paths.append(out_path)
        LOGGER.info("Saved dose–response plot for %s to '%s'", cpd_id, out_path)

    return paths


def tsv_preview_html(
    *,
    path: Path,
    max_rows: int = 100,
    title: str | None = None,
) -> str:
    """
    Render a TSV file as an HTML table snippet.

    Only the first `max_rows` rows are shown to keep the report compact.
    If the file does not exist or cannot be read, a short message is
    returned instead of a table.

    Parameters
    ----------
    path : Path
        Path to the TSV file.
    max_rows : int, optional
        Maximum number of rows to display.
    title : str, optional
        Optional title to place above the table.

    Returns
    -------
    str
        HTML snippet containing an optional title and a table, or a short
        message if the file is unavailable.
    """
    if not path.exists():
        return (
            f"<p><strong>{path.name}</strong> not found; "
            "no preview available.</p>"
        )

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as err:  # noqa: BLE001
        return (
            f"<p>Could not load <strong>{path.name}</strong> for preview "
            f"(error: {err}).</p>"
        )

    n_rows, n_cols = df.shape
    if n_rows > max_rows:
        df_preview = df.head(max_rows).copy()
        note = (
            f"<p>Showing first {max_rows} of {n_rows} rows "
            f"({n_cols} columns).</p>"
        )
    else:
        df_preview = df
        note = f"<p>Showing all {n_rows} rows ({n_cols} columns).</p>"

    table_html = df_preview.to_html(
        index=False,
        escape=True,
        border=0,
        classes="data-table",
    )

    heading = f"<h3>{title}</h3>" if title is not None else ""
    wrapped = (
        f"{heading}{note}"
        "<div style='max-height:400px; overflow:auto; border:1px solid #ccc; "
        "padding:0.5em; margin-bottom:1.5em;'>"
        f"{table_html}"
        "</div>"
    )
    return wrapped



def write_html_report(
    *,
    output_dir: Path,
    per_well_tsv: Path,
    fisher_tsv: Path,
    drc_tsv: Path | None,
    plate_heatmap_png: Path | None,
    volcano_png: Path | None,
    barplot_png: Path | None,
    drc_pngs: List[Path],
    volcano_interactive_html: Path | None = None,
    inducing_barplot_png: Path | None = None,
) -> Path:
    """
    Write a simple HTML report linking TSVs and embedding plots and previews.

    The report includes:
      - Download links for all main TSV outputs.
      - Embedded, scrollable HTML previews of the TSV tables
        (first N rows only).
      - Static PNG plots and optional interactive volcano link.
      - Optional dose–response curve PNGs.

    Parameters
    ----------
    output_dir : Path
        Output directory for the HTML file.
    per_well_tsv : Path
        Path to per-well QC table.
    fisher_tsv : Path
        Path to Fisher results table.
    drc_tsv : Path or None
        Path to dose–response fit summary, if present.
    plate_heatmap_png : Path or None
        Plate heatmap PNG.
    volcano_png : Path or None
        Volcano plot PNG.
    barplot_png : Path or None
        Top hits barplot PNG.
    drc_pngs : list of Path
        List of individual DRC plot PNGs.
    volcano_interactive_html : Path or None, optional
        Path to interactive volcano HTML, if generated.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    html_path = output_dir / "acrosome_dose_response_report.html"
    LOGGER.info("Writing HTML report to '%s'", html_path)

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<title>Acrosome reaction – single-plate dose–response report</title>")
    parts.append(
        "<style>"
        "body { font-family: Arial, sans-serif; margin: 1.5em; }"
        "h1, h2, h3 { color: #333333; }"
        "img { max-width: 100%; height: auto; margin-bottom: 1em; }"
        "code { background-color: #f0f0f0; padding: 0.1em 0.3em; }"
        "ul { margin-top: 0.2em; }"
        ".data-table { border-collapse: collapse; font-size: 0.85em; }"
        ".data-table th, .data-table td { "
        "  border: 1px solid #ddd; padding: 0.25em 0.5em; "
        "}"
        ".data-table th { background-color: #f7f7f7; }"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<h1>Acrosome reaction vs DMSO – single-plate analysis</h1>")

    # Download links
    parts.append("<h2>Data tables</h2>")
    parts.append("<ul>")
    parts.append(
        f"<li>Per-well summary (QC-aware): "
        f"<a href='{per_well_tsv.name}'>{per_well_tsv.name}</a></li>"
    )
    parts.append(
        f"<li>Fisher per compound–dose: "
        f"<a href='{fisher_tsv.name}'>{fisher_tsv.name}</a></li>"
    )
    if drc_tsv is not None and drc_tsv.exists():
        parts.append(
            f"<li>Dose–response fit summary: "
            f"<a href='{drc_tsv.name}'>{drc_tsv.name}</a></li>"
        )
    parts.append("</ul>")

    # Embedded previews of TSVs
    parts.append("<h2>Table previews</h2>")
    parts.append(
        tsv_preview_html(
            path=per_well_tsv,
            max_rows=100,
            title="Per-well summary (QC-aware)",
        )
    )
    parts.append(
        tsv_preview_html(
            path=fisher_tsv,
            max_rows=100,
            title="Fisher per compound–dose",
        )
    )
    if drc_tsv is not None and drc_tsv.exists():
        parts.append(
            tsv_preview_html(
                path=drc_tsv,
                max_rows=100,
                title="Dose–response fit summary",
            )
        )

    # Plate heatmap
    if plate_heatmap_png is not None and plate_heatmap_png.exists():
        parts.append("<h2>Plate layout – AR% per well (QC-kept)</h2>")
        parts.append(f"<img src='{plate_heatmap_png.name}' alt='Plate heatmap'>")

    # Interactive volcano
    if (
        volcano_interactive_html is not None
        and volcano_interactive_html.exists()
    ):
        parts.append("<h2>Interactive volcano plot</h2>")
        parts.append(
            "<p>You can explore compound-level results by hovering over points "
            "in the interactive volcano plot:</p>"
        )
        parts.append(
            f"<p><a href='{volcano_interactive_html.name}'>"
            "Open interactive Fisher volcano (HTML)</a></p>"
        )

    # Static volcano
    if volcano_png is not None and volcano_png.exists():
        parts.append("<h2>Fisher results – volcano plot</h2>")
        parts.append(f"<img src='{volcano_png.name}' alt='Volcano plot'>")

    # Top hits barplot
    if barplot_png is not None and barplot_png.exists():
        parts.append("<h2>Top compound–dose hits by FDR</h2>")
        parts.append(f"<img src='{barplot_png.name}' alt='Top hits barplot'>")
        
        if inducing_barplot_png is not None and inducing_barplot_png.exists():
        parts.append("<h2>Compounds that only induce AR</h2>")
        parts.append(
            "<p>Compounds shown here have no doses that reduce AR below the "
            "DMSO level, and at least one significant dose (FDR ≤ 0.05) that "
            "increases AR. Bars reflect the dose with the largest ΔAR%%.</p>"
        )
        parts.append(
            f"<img src='{inducing_barplot_png.name}' "
            f"alt='Compounds that only induce AR'>"
        )

    # DRC curves
    if drc_pngs:
        parts.append("<h2>Example dose–response curves</h2>")
        for png in drc_pngs:
            parts.append(f"<h3>{png.stem}</h3>")
            parts.append(f"<img src='{png.name}' alt='{png.stem}'>")

    # Footer text
    parts.append(
        "<p>All p-values are from two-sided Fisher's exact tests comparing each "
        "compound–dose to pooled DMSO wells on this plate. "
        "False discovery rate is controlled using the Benjamini–Hochberg "
        "procedure. Dose–response curves, where available, are fitted using a "
        "four-parameter logistic function on log<sub>10</sub>-concentration.</p>"
    )

    parts.append("</body></html>")

    html_content = "\n".join(parts)
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for command-line execution.
    """
    args = parse_args()
    configure_logging(verbosity=args.verbosity)

    cp_dir = Path(args.cp_dir)
    lib_path = Path(args.library_metadata)
    ctrl_path = Path(args.controls)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover and load CellProfiler Image table
    image_path, _ = discover_cp_files(cp_dir=cp_dir)
    df_image = load_image_counts(
        image_path=image_path,
        plate_col=args.image_plate_col,
        well_col=args.image_well_col,
        cell_count_col=args.image_cell_count_col,
        ar_count_col=args.image_ar_count_col,
    )

    # 2. Load metadata and controls
    df_meta = load_compound_metadata(metadata_path=lib_path)
    df_ctrl = load_controls(controls_path=ctrl_path)

    # 3. Merge per-image counts with metadata
    df_image = merge_image_with_metadata(
        df_image=df_image,
        df_meta=df_meta,
        df_ctrl=df_ctrl,
    )

    # 4. Aggregate per well
    df_well = aggregate_per_well_from_counts(df_image=df_image)

    # 5. QC
    df_well_qc = flag_wells_for_qc(
        df_well=df_well,
        min_cells_per_well=args.min_cells_per_well,
        dmso_iqr_multiplier=args.dmso_iqr_multiplier,
        replicate_mad_multiplier=args.replicate_mad_multiplier,
    )

        # DEBUG: how many DMSO wells before/after QC?
    LOGGER.info(
        "cpd_type counts (all wells):\n%s",
        df_well_qc["cpd_type"].value_counts(dropna=False),
    )
    LOGGER.info(
        "DMSO wells total: %d",
        (df_well_qc["cpd_type"] == "DMSO").sum(),
    )
    LOGGER.info(
        "DMSO wells with qc_keep=True: %d",
        df_well_qc.query("cpd_type == 'DMSO' and qc_keep").shape[0],
    )

    per_well_tsv = output_dir / "acrosome_per_well_qc.tsv"
    df_well_qc.to_csv(per_well_tsv, sep="\t", index=False)
    LOGGER.info("Wrote per-well QC table to '%s'", per_well_tsv)

    # 6. Fisher tests per compound–dose
    fisher_df = run_fisher_per_compound_dose(df_well_qc=df_well_qc)

    # Enrich Fisher results with chemistry metadata (one row per cpd_id)
    chem_cols = [
        "SMILES",
        "Series",
        "Parent Molecular Formula",
        "Parent Molecular Mass",
        "Full Molecular Formula Salt",
    ]
    available_chem_cols = [c for c in chem_cols if c in df_meta.columns]
    if available_chem_cols:
        LOGGER.info(
            "Merging chemical metadata into Fisher table for columns: %s",
            ", ".join(available_chem_cols),
        )
        chem_meta = (
            df_meta[["cpd_id"] + available_chem_cols]
            .drop_duplicates(subset=["cpd_id"])
        )
        n_before = fisher_df.shape[0]
        fisher_df = fisher_df.merge(
            chem_meta,
            on="cpd_id",
            how="left",
        )
        n_missing = fisher_df["cpd_id"].isna().sum()
        LOGGER.debug(
            "After merging chemistry metadata: %d rows, %d with missing cpd_id.",
            n_before,
            n_missing,
        )
    else:
        LOGGER.warning(
            "None of the requested chemistry columns are present in metadata; "
            "interactive hover will not include SMILES / formulae."
        )

    fisher_tsv = output_dir / "acrosome_fisher_per_dose.tsv"
    fisher_df.to_csv(fisher_tsv, sep="\t", index=False)
    LOGGER.info("Wrote Fisher results to '%s'", fisher_tsv)



    # 7. Optional dose–response fitting
    drc_tsv: Path | None = None
    drc_df: pd.DataFrame | None = None
    if args.fit_dose_response:
        drc_df = fit_dose_response_per_compound(
            fisher_df=fisher_df,
            min_points=args.min_drc_points,
        )
        drc_tsv = output_dir / "acrosome_drc_fits.tsv"
        drc_df.to_csv(drc_tsv, sep="\t", index=False)
        LOGGER.info("Wrote dose–response fits to '%s'", drc_tsv)


    # 8. Plots
    plate_heatmap_png = output_dir / "plate_AR_pct_heatmap.png"
    plot_plate_heatmap(
        df_well_qc=df_well_qc[df_well_qc["qc_keep"]],
        output_path=plate_heatmap_png,
    )

    volcano_png = output_dir / "fisher_volcano.png"
    plot_volcano(fisher_df=fisher_df, output_path=volcano_png)

    volcano_interactive_html: Path | None = None
    volcano_interactive_html = plot_volcano_interactive(
        fisher_df=fisher_df,
        output_path=output_dir / "fisher_volcano_interactive.html",
    )
    if volcano_interactive_html is None:
        LOGGER.warning("Interactive volcano HTML was not generated.")
    else:
        LOGGER.info(
            "Interactive volcano HTML path: %s (exists=%s)",
            volcano_interactive_html,
            volcano_interactive_html.exists(),
        )


    barplot_png = output_dir / "top_delta_AR_pct_barplot.png"
    plot_top_delta_barplot(fisher_df=fisher_df, output_path=barplot_png, top_n=30)

    drc_pngs: List[Path] = []
    if args.fit_dose_response and drc_df is not None:
        drc_pngs = plot_dose_response_examples(
            fisher_df=fisher_df,
            drc_df=drc_df,
            output_dir=output_dir,
            max_compounds=12,
        )

    barplot_png = output_dir / "top_delta_AR_pct_barplot.png"
    plot_top_delta_barplot(
        fisher_df=fisher_df,
        output_path=barplot_png,
        top_n=30,
    )

    inducing_barplot_png = output_dir / "inducing_compounds_barplot.png"
    plot_inducing_compounds_barplot(
        fisher_df=fisher_df,
        output_path=inducing_barplot_png,
        q_threshold=0.05,
        min_delta=0.0,
        top_n=30,
    )


    # 9. HTML report
    html_report = write_html_report(
        output_dir=output_dir,
        per_well_tsv=per_well_tsv,
        fisher_tsv=fisher_tsv,
        drc_tsv=drc_tsv,
        plate_heatmap_png=plate_heatmap_png,
        volcano_png=volcano_png,
        barplot_png=barplot_png,
        drc_pngs=drc_pngs,
        volcano_interactive_html=volcano_interactive_html,
        inducing_barplot_png=inducing_barplot_png,
    )



    LOGGER.info("HTML report written to '%s'", html_report)

    LOGGER.info("Analysis complete.")


if __name__ == "__main__":
    main()
