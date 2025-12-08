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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import fisher_exact


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging and argument parsing
# ---------------------------------------------------------------------------

def configure_logging(*, verbosity: int, output_dir: str) -> None:
    """
    Configure logging for both console and file outputs.

    Parameters
    ----------
    verbosity : int
        Verbosity level (0=warning, 1=info, 2=debug).
    output_dir : str
        Directory where the log file will be written.
    """
    # Select log level
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    # Make sure directory exists
    os.makedirs(output_dir, exist_ok=True)
    logfile = os.path.join(output_dir, "acrosome_analysis.log")

    # Remove any pre-existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging to BOTH console + file
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),                  # Console
            logging.FileHandler(logfile, mode="w")    # Log file
        ],
    )

    LOGGER.debug("Logging configured with level %s", logging.getLevelName(level))

    # Add command-line info to log
    cmd_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    LOGGER.info("Command line: %s", cmd_line)
    LOGGER.info("Logging to file: %s", logfile)



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
        default=None,
        help="Optional controls file (DMSO etc.). Omit to skip controls.",
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
            "IQR multiplier for defining AR% outliers amongst DMSO wells. "
            "Only DMSO wells are auto-dropped as AR% outliers."
        ),
    )
    parser.add_argument(
        "--replicate_mad_multiplier",
        dest="replicate_mad_multiplier",
        type=float,
        default=5.0,
        help=(
            "MAD multiplier for flagging within-compound–dose replicate "
            "AR% outliers."
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

    parser.add_argument(
        "--motility_csv",
        type=str,
        required=False,
        default=None,
        help="Optional motility data CSV/TSV containing HA/PM/TM/VCL metrics."
    )


    return parser.parse_args()




# ---------------------------------------------------------------------------
# QC functions
# ---------------------------------------------------------------------------

def load_object_table(*, path: Path) -> pd.DataFrame:
    """
    Load any CellProfiler object-level table (.csv or .csv.gz).

    Parameters
    ----------
    path : Path
        Path to the object table.

    Returns
    -------
    pandas.DataFrame
        Raw object-level table.
    """
    LOGGER.info("Loading object-level table: %s", path)

    compression = "gzip" if str(path).endswith(".gz") else None

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        compression=compression,
    )

    LOGGER.info(
        "Loaded object-level table with %d rows and %d columns from %s",
        df.shape[0], df.shape[1], path
    )

    return df



def object_level_qc(
    *,
    df_image: pd.DataFrame,
    df_acrosome: pd.DataFrame | None,
    df_sperm: pd.DataFrame | None,
    df_nuclei: pd.DataFrame | None,
) -> Dict[str, pd.DataFrame]:
    """
    Perform object-level QC using available CellProfiler object tables.

    Missing tables (SpermCells, FilteredNuclei) are handled gracefully
    and assigned zero-count columns.
    """

    qc_outputs = {}
    image_num_col = "ImageNumber"

    # -------------------------------------------------------------
    # 1. BUILD PER-IMAGE OBJECT COUNTS
    # -------------------------------------------------------------
    counts = []

    # Acrosome table (most important)
    if df_acrosome is not None:
        tmp = df_acrosome[[image_num_col]].copy()
        tmp["acrosome_objects"] = 1
        counts.append(tmp)
    else:
        LOGGER.warning("No Acrosome table provided – object QC limited.")

    # SpermCells table (optional)
    if df_sperm is not None:
        tmp = df_sperm[[image_num_col]].copy()
        tmp["sperm_objects"] = 1
        counts.append(tmp)

    # FilteredNuclei table (optional)
    if df_nuclei is not None:
        tmp = df_nuclei[[image_num_col]].copy()
        tmp["nuclei_objects"] = 1
        counts.append(tmp)

    if not counts:
        LOGGER.warning("No object tables provided – skipping object-level QC.")
        return {}

    df_counts = pd.concat(counts, ignore_index=True)

    # group by ImageNumber
    df_image_qc = (
        df_counts.groupby(image_num_col)
        .sum()
        .reset_index()
    )

    # Ensure missing columns are created as zero
    for col in ["acrosome_objects", "sperm_objects", "nuclei_objects"]:
        if col not in df_image_qc.columns:
            df_image_qc[col] = 0

    # merge ImageNumber -> plate/well
    df_image_qc = df_image_qc.merge(
        df_image[["ImageNumber", "plate_id", "well_id"]],
        on="ImageNumber",
        how="left",
    )

    qc_outputs["image_qc"] = df_image_qc

    # -------------------------------------------------------------
    # 2. WELL-LEVEL QC SUMMARY
    # -------------------------------------------------------------
    df_well_qc = (
        df_image_qc.groupby(["plate_id", "well_id"])
        .agg(
            images=("ImageNumber", "nunique"),
            acrosome_objects=("acrosome_objects", "sum"),
            sperm_objects=("sperm_objects", "sum"),
            nuclei_objects=("nuclei_objects", "sum"),
        )
        .reset_index()
    )

    df_well_qc["objects_total"] = (
        df_well_qc["acrosome_objects"]
        + df_well_qc["sperm_objects"]
        + df_well_qc["nuclei_objects"]
    )

    qc_outputs["well_qc"] = df_well_qc

    # -------------------------------------------------------------
    # 3. FLAG PROBLEM WELLS
    # -------------------------------------------------------------
    flagged = df_well_qc[
        (df_well_qc["objects_total"] < 50)
        | (df_well_qc["images"] < 2)
        | (df_well_qc["objects_total"] > 50_000)
    ].copy()

    qc_outputs["flagged_wells"] = flagged

    return qc_outputs


def intensity_qc(
    *,
    df_acrosome: pd.DataFrame | None,
    df_image: pd.DataFrame,
    intensity_col: str = None,
) -> pd.DataFrame | None:
    """
    Perform intensity-based QC using the acrosome object table.

    Parameters
    ----------
    df_acrosome : pandas.DataFrame or None
        The Acrosome object table, minimally containing:
            ImageNumber, <intensity_col>
        If None, returns None.
    df_image : pandas.DataFrame
        The Image table (for plate_id, well_id)
    intensity_col : str or None
        Column representing acrosome intensity. If None, attempts to detect one.

    Returns
    -------
    pandas.DataFrame or None
        Well-level intensity QC table, or None if unavailable.
    """
    if df_acrosome is None:
        LOGGER.warning("Intensity QC skipped: no Acrosome table available.")
        return None

    # Auto-detect an intensity column if user did not provide one
    if intensity_col is None:
        candidates = [c for c in df_acrosome.columns if "Intensity" in c]
        if not candidates:
            LOGGER.warning("No intensity column found in Acrosome table.")
            return None
        intensity_col = candidates[0]
        LOGGER.info("Using intensity column '%s' for intensity QC.", intensity_col)

    df = df_acrosome[[ "ImageNumber", intensity_col ]].copy()

    # Merge plate + well info
    df = df.merge(
        df_image[["ImageNumber", "plate_id", "well_id"]],
        on="ImageNumber",
        how="left",
    )

    # Compute well-level stats
    df_qc = (
        df.groupby(["plate_id", "well_id"])[intensity_col]
        .agg(
            median_intensity="median",
            mean_intensity="mean",
            sd_intensity="std",
            mad_intensity=lambda x: (x - x.median()).abs().median(),
            n_objects="count",
        )
        .reset_index()
    )

    # CV (coefficient of variation)
    df_qc["cv_intensity"] = df_qc["sd_intensity"] / df_qc["mean_intensity"]

    # Flag outliers
    df_qc["is_low_intensity"] = df_qc["median_intensity"] < df_qc["median_intensity"].median() * 0.4
    df_qc["is_high_variability"] = df_qc["cv_intensity"] > 1.0

    return df_qc


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def write_qc_pdf(
    *,
    output_path: Path,
    image_qc: pd.DataFrame,
    well_qc: pd.DataFrame,
    intensity_qc: pd.DataFrame | None = None,
):
    """
    Produce a multi-page PDF QC report for the plate.

    Parameters
    ----------
    output_path : Path
        Destination PDF path.
    image_qc : pandas.DataFrame
        Per-image QC summary.
    well_qc : pandas.DataFrame
        Per-well object QC.
    intensity_qc : pandas.DataFrame or None
        Optional intensity QC table.
    """
    with PdfPages(output_path) as pdf:

        # PAGE 1 — OBJECT COUNTS
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(image_qc["acrosome_objects"], bins=50)
        ax.set_title("Object counts per image (Acrosome)")
        ax.set_xlabel("Objects per image")
        ax.set_ylabel("Frequency")
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 2 — TOTAL OBJECTS BY WELL
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(well_qc["objects_total"], bins=50)
        ax.set_title("Total objects per well")
        ax.set_xlabel("Objects")
        ax.set_ylabel("Number of wells")
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 3 — INTENSITY QC, if available
        if intensity_qc is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(intensity_qc["median_intensity"], bins=50)
            ax.set_title("Median acrosome intensity")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Wells")
            pdf.savefig(fig)
            plt.close(fig)

        # PAGE 4 — FLAGGED WELLS SUMMARY
        flagged = well_qc[
            (well_qc["objects_total"] < 50)
            | (well_qc["images"] < 2)
            | (well_qc["objects_total"] > 50000)
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        txt = flagged.to_string(index=False)
        ax.text(0, 1, "Flagged Wells:\n" + txt, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

    LOGGER.info("QC PDF written: %s", output_path)


def load_motility_file(path: str) -> pd.DataFrame:
    """
    Load optional motility dataset and keep only the required columns.
    Missing cpd_id values are removed.
    """

    df = pd.read_csv(path, sep=None, engine="python")  # auto-detect delimiter

    required_cols = [
        "cpd_id",
        "HA_value", "HA_pc",
        "PM_value", "PM_pc",
        "TM_value", "TM_pc",
        "VCL_median_value", "VCL_median_pc",
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Motility file missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna(subset=["cpd_id"])  # removes DMSO rows like cpd_id = NA

    return df



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
    Load and harmonise PTOD compound metadata.

    Supports multiple well-identifier formats, but prefers a pre-built
    'well_id' column (as produced by merge_scp_metadata.py).

    Parameters
    ----------
    metadata_path : Path
        Path to the merged metadata file.

    Returns
    -------
    pandas.DataFrame
        Metadata with harmonised:
            - plate_id
            - well_id
            - cpd_id
            - conc
    """
    LOGGER.info("Loading library metadata from '%s'", metadata_path)

    df = pd.read_csv(metadata_path, sep=None, engine="python")
    LOGGER.info(
        "Loaded library metadata with %d rows and %d columns",
        df.shape[0], df.shape[1]
    )

    # ---------------------------------------------------------
    # NORMALISE COLUMN NAMES (strip + lowercase)
    # ---------------------------------------------------------
    df.columns = [c.strip() for c in df.columns]

    cols_lower = {c.lower(): c for c in df.columns}

    # ---------------------------------------------------------
    # CASE 1 — Metadata already includes 'well_id'
    # ---------------------------------------------------------
    if "well_id" in cols_lower:
        well_col = cols_lower["well_id"]
        LOGGER.info("Using existing 'well_id' column from merged metadata: %s", well_col)
        df["well_id"] = df[well_col].astype(str).str.upper().str.strip()

    # ---------------------------------------------------------
    # CASE 2 — Has Row + Col
    # ---------------------------------------------------------
    elif "row" in cols_lower and "col" in cols_lower:
        LOGGER.info("Building 'well_id' from ROW/COL columns.")
        df["well_id"] = [
            standardise_well_from_row_col(row=r, col=c)

            for r, c in zip(df[cols_lower["row"]], df[cols_lower["col"]])
        ]

    # ---------------------------------------------------------
    # CASE 3 — Has Well (A01 format)
    # ---------------------------------------------------------
    elif "well" in cols_lower:
        LOGGER.info("Normalising 'well_id' from 'Well' column.")
        df["well_id"] = df[cols_lower["well"]].map(
            lambda w: standardise_well_string(well=w)
        )

    # ---------------------------------------------------------
    # CASE 4 — PTODWELLREFERENCE (A001 → A01)
    # ---------------------------------------------------------
    elif "ptodwellreference" in cols_lower:
        LOGGER.info("Converting PTODWELLREFERENCE → A01 format.")
        col = cols_lower["ptodwellreference"]
        vals = df[col].astype(str).str.upper().str.strip()
        df["well_id"] = (
            vals.str[0] +
            vals.str[1:].astype(int).astype(str).str.zfill(2)
        )

    # ---------------------------------------------------------
    # FINAL — No usable well column
    # ---------------------------------------------------------
    else:
        raise KeyError(
            f"No usable well column found. Columns present: {df.columns.tolist()}"
        )

    # ---------------------------------------------------------
    # Rename PTOD standard columns
    # ---------------------------------------------------------
    rename_map = {
        "DDD": "cpd_id",
        "PTODCONCVALUE": "conc",
        "Plate ID": "plate_id",
    }

    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)


    # ---------------------------------------------------------
    # NORMALISE plate_id (remove suffixes like '_15082024')
    # ---------------------------------------------------------
    if "plate_id" in df.columns:
        if df["plate_id"].astype(str).str.contains("_").any():
            before_vals = df["plate_id"].unique()[:10]

            df["plate_id"] = df["plate_id"].astype(str).str.split("_").str[0]

            after_vals = df["plate_id"].unique()[:10]

            LOGGER.warning(
                "Plate IDs in library metadata contained suffixes. They were normalised.\n"
                "Examples before: %s\n"
                "Examples after : %s",
                before_vals,
                after_vals,
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
) -> Dict[str, Path | None]:
    """
    Discover CellProfiler output tables in a directory.

    Finds both .csv and .csv.gz variants for:
        - *_Image
        - *_Acrosome
        - *_SpermCells
        - *_FilteredNuclei

    Only the Image table is required; others are optional.

    Returns
    -------
    dict
        Keys:
            image
            acrosome
            spermcells
            filterednuclei
    """

    def _find(pattern: str) -> Path | None:
        files = sorted(
            list(cp_dir.glob(pattern + ".csv")) +
            list(cp_dir.glob(pattern + ".csv.gz"))
        )
        return files[0] if files else None

    image_path = _find("*_Image")
    acrosome_path = _find("*_Acrosome")
    spermcells_path = _find("*_SpermCells")
    nuclei_path = _find("*_FilteredNuclei")

    if image_path is None:
        raise FileNotFoundError(
            f"No '*_Image.csv' or '*_Image.csv.gz' found in {cp_dir}"
        )

    LOGGER.info(f"Using Image table: {image_path}")

    if acrosome_path:
        LOGGER.info(f"Found Acrosome table: {acrosome_path}")
    if spermcells_path:
        LOGGER.info(f"Found SpermCells table: {spermcells_path}")
    if nuclei_path:
        LOGGER.info(f"Found FilteredNuclei table: {nuclei_path}")

    return {
        "image": image_path,
        "acrosome": acrosome_path,
        "spermcells": spermcells_path,
        "filterednuclei": nuclei_path,
    }



def load_acrosome_table(*, acrosome_path: Path) -> pd.DataFrame:
    """
    Load the CellProfiler object-level Acrosome table (.csv or .csv.gz).

    This does not yet integrate with the pipeline but is kept ready
    for future use (QC, morphology, AR object-level validation).

    Parameters
    ----------
    acrosome_path : Path
        Path to the Acrosome table.

    Returns
    -------
    pandas.DataFrame
        Raw object-level acrosome table.
    """
    LOGGER.info("Loading Acrosome table from '%s'", acrosome_path)

    compression = "gzip" if str(acrosome_path).endswith(".gz") else None

    df = pd.read_csv(
        acrosome_path,
        sep=None,
        engine="python",
        compression=compression,
    )

    LOGGER.info("Acrosome table loaded with %d rows and %d columns", *df.shape)
    return df


def load_acrosome_minimal(
    *,
    path: Path,
    cols: List[str] = ["ImageNumber"],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Efficiently load only required columns from a large Acrosome table.

    Uses chunked reading to avoid ever loading the whole file at once.

    Parameters
    ----------
    path : Path
        Path to the Acrosome CSV/CSV.GZ.
    cols : list of str
        Columns to load (default = just ImageNumber).
    chunksize : int
        Number of rows per chunk.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the required columns.
    """
    LOGGER.info("Loading minimal Acrosome columns from %s", path)

    compression = "gzip" if str(path).endswith(".gz") else None

    dfs = []

    for chunk in pd.read_csv(
        path,
        sep=None,
        engine="python",
        compression=compression,
        usecols=lambda c: c in cols,
        chunksize=chunksize,
    ):
        dfs.append(chunk)

    df_out = pd.concat(dfs, ignore_index=True)
    LOGGER.info("Minimal Acrosome load complete: %d rows", len(df_out))

    return df_out


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


def get_top_compounds(
    *,
    fisher_df: pd.DataFrame,
    n: int = 20,
    score_col: str = "delta_AR_pct",
) -> List[str]:
    """
    Return the top N compounds ranked by a chosen metric (default: delta_AR_pct).
    """
    ranked = (
        fisher_df
        .dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
    )
    return ranked["cpd_id"].head(n).tolist()



def merge_image_with_metadata(
    *,
    df_image: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_ctrl: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge per-image counts with harmonised library metadata and optionally
    with a controls table. If df_ctrl is not provided, control types are
    inferred from well positions (columns 23 = DMSO, 24 = POS).

    The function:
        • normalises metadata column names ('OLPTID' → 'plate_id',
          'PTODWELLREFERENCE' → 'well_id', 'OBJDID' → 'cpd_id')
        • normalises well identifiers (A001 → A01)
        • merges CellProfiler per-image data with metadata on
          ('plate_id', 'well_id')
        • assigns cpd_type values: 'DMSO', 'POS', or 'TEST'
        • if df_ctrl is provided, overrides inferred control types

    Parameters
    ----------
    df_image : pandas.DataFrame
        Per-image CellProfiler table. Must contain:
        plate_id, well_id, Cell_count, AR_count.
    df_meta : pandas.DataFrame
        Stamped metadata including plate barcode, well ID and cpd_id.
    df_ctrl : pandas.DataFrame or None
        Optional control table with plate_id, well_id, control_type.

    Returns
    -------
    pandas.DataFrame
        Per-image table with metadata merged and control types assigned.
    """

    df = df_image.copy()
    meta = df_meta.copy()

    # ------------------------------------------------------------------
    # 1. Harmonise metadata colnames to match df_image conventions
    # ------------------------------------------------------------------
    meta = meta.rename(
        columns={
            "OLPTID": "plate_id",
            "PTODWELLREFERENCE": "well_id",
            "OBJDID": "cpd_id",
            # fallback variants:
            "Plate_Metadata": "plate_id",
            "Well_Metadata": "well_id",
        }
    )

    # concentration naming
    for col in ["conc", "Concentration", "PTODCONCVALUE"]:
        if col in meta.columns:
            meta = meta.rename(columns={col: "conc"})
            break

    # ------------------------------------------------------------------
    # 2. Normalise well strings
    # ------------------------------------------------------------------
    meta["well_id"] = meta["well_id"].map(lambda w: standardise_well_string(well=w))

    # ------------------------------------------------------------------
    # 3. Merge metadata into df_image
    # ------------------------------------------------------------------
    LOGGER.info("Merging per-image data with stamped metadata.")

    df = df.merge(
        meta,
        how="left",
        on=["plate_id", "well_id"],   #"plate_id", 
        suffixes=("", "_meta"),
    )

    # warn for missing metadata
    if df["cpd_id"].isna().any():
        missing = df["cpd_id"].isna().sum()
        LOGGER.warning(
            "%d image rows have missing cpd_id after metadata merge. "
            "Check plate_id and well_id formatting.",
            missing,
        )

    # Default all wells to TEST unless we reassign them
    df["cpd_type"] = "TEST"

    # ------------------------------------------------------------------
    # 4. Optional control file
    # ------------------------------------------------------------------
    if df_ctrl is not None and not df_ctrl.empty:
        LOGGER.info("Merging control annotations from provided control file.")
        ctrl = df_ctrl.copy()

        ctrl = ctrl.rename(
            columns={
                "source_plate_barcode": "plate_id",
                "source_well": "well_id",
            }
        )
        ctrl["well_id"] = ctrl["well_id"].map(lambda w: standardise_well_string(well=w))

        df = df.merge(
            ctrl[["plate_id", "well_id", "control_type"]],
            how="left",
            on=["plate_id", "well_id"],
            suffixes=("", "_ctrl"),
        )
        df["cpd_type"] = df["control_type"].fillna(df["cpd_type"])
        df.drop(columns=["control_type"], inplace=True, errors="ignore")
    # ------------------------------------------------------------------
    # 5. Control inference (DMSO and POS)
    # ------------------------------------------------------------------

    # Extract numeric well column: A01 → 1, B12 → 12
    df["_colnum"] = df["well_id"].str[1:].astype(int)

    # If is_dmso not present or all missing → try to infer
    if "is_dmso" not in df.columns or df["is_dmso"].isna().all():
        LOGGER.warning(
            "DMSO inference: no 'is_dmso' column present or all NaN. "
            "Attempting to infer controls from cpd_id."
        )

        # Try metadata-based inference first
        if "cpd_id" in df.columns and df["cpd_id"].notna().any():
            inferred_mask = df["cpd_id"].astype(str).str.upper().eq("DMSO")

            if inferred_mask.any():
                df["cpd_type"] = np.where(inferred_mask, "DMSO", "TEST")
                df["is_dmso"] = inferred_mask
                LOGGER.info("DMSO wells successfully identified from cpd_id column.")

            else:
                # Metadata exists but has *no* DMSO entries → fallback
                LOGGER.critical(
                    "No DMSO wells detected in metadata (cpd_id). "
                    "Falling back to vendor assumption: column 23 = DMSO, column 24 = POS. "
                    "THIS MAY BE INCORRECT FOR THIS SCREEN."
                )

                # Fallback assumption
                mask_low = df["_colnum"] == 23
                mask_high = df["_colnum"] == 24

                df["cpd_type"] = np.where(mask_low, "DMSO", df["cpd_type"])
                df["cpd_type"] = np.where(mask_high, "POS", df["cpd_type"])

                df["is_dmso"] = mask_low

        else:
            # No cpd_id column available → fallback only
            LOGGER.critical(
                "cpd_id column absent. Cannot identify DMSO from metadata. "
                "Falling back to vendor assumption: col 23 = DMSO, col 24 = POS. "
                "THIS FALLBACK IS ONLY SAFE IF THE SCREEN USES THESE POSITIONS."
            )

            mask_low = df["_colnum"] == 23
            mask_high = df["_colnum"] == 24

            df["cpd_type"] = np.where(mask_low, "DMSO", "TEST")
            df["cpd_type"] = np.where(mask_high, "POS", df["cpd_type"])
            df["is_dmso"] = mask_low

    # Clean up
    df.drop(columns=["_colnum"], inplace=True, errors="ignore")

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

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    Compute Benjamini–Hochberg q-values for an array of p-values.

    Parameters
    ----------
    p_values : numpy.ndarray
        Array of p-values.

    Returns
    -------
    numpy.ndarray
        Array of q-values in the same order.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.shape[0]

    if n == 0:
        return np.array([])

    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = p[order] * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_corrected = np.empty_like(q)
    q_corrected[order] = q

    return np.clip(q_corrected, 0, 1)



def run_fisher_per_compound_dose(*, df_well_qc: pd.DataFrame) -> pd.DataFrame:
    """
    Run Fisher’s exact tests per compound–dose versus pooled DMSO controls.

    DMSO wells must have conc = NaN. Non-DMSO wells must have real numeric conc.

    Returns a per-compound/dose result table.
        Run Fisher’s exact test per compound–dose versus pooled DMSO controls.

    Parameters
    ----------
    df_well_qc : pandas.DataFrame
        Per-well QC table with required columns:
        - plate_id
        - cpd_id
        - cpd_type
        - conc
        - total_cells
        - total_AR_cells
        - qc_keep

    Returns
    -------
    pandas.DataFrame
        Per compound–dose Fisher results with:
        plate_id, cpd_id, conc, cpd_type, n_wells,
        total_cells_compound, total_AR_compound, AR_pct_compound,
        total_cells_dmso, total_AR_dmso, AR_pct_dmso,
        delta_AR_pct, odds_ratio, p_value, q_value
    """
    LOGGER.info("DEBUG: Entering Fisher test function")

    df = df_well_qc[df_well_qc["qc_keep"]].copy()
    LOGGER.info("DEBUG: Initial QC-pass rows = %d", df.shape[0])

    # ------------------------------------------------------------
    # STEP 1 — Clean conc
    # ------------------------------------------------------------
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")

    # Identify DMSO
    dmso_mask = df["cpd_type"].eq("DMSO")
    df.loc[dmso_mask, "conc"] = np.nan

    LOGGER.info("DEBUG: Unique conc BEFORE NaN-drop = %s",
                df["conc"].unique())

    # ------------------------------------------------------------
    # Separate test wells (non-DMSO)
    # ------------------------------------------------------------
    test_df = df.loc[~dmso_mask].copy()

    LOGGER.info("DEBUG: Unique conc AFTER NaN-drop/float = %s",
                test_df["conc"].unique())

    if test_df.empty:
        raise ValueError("No non-DMSO wells remain; cannot run Fisher tests.")

    # ------------------------------------------------------------
    # Compute pooled DMSO baseline
    # ------------------------------------------------------------
    dmso_df = df.loc[dmso_mask].copy()
    if dmso_df.empty:
        raise ValueError("No DMSO wells found after QC; cannot compute baseline.")

    total_cells_dmso = int(dmso_df["total_cells"].sum())
    total_AR_dmso = int(dmso_df["total_AR_cells"].sum())
    AR_pct_dmso = (total_AR_dmso / total_cells_dmso) * 100.0

    LOGGER.info("Pooled DMSO counts: AR=%d, non-AR=%d, AR%%=%.3f",
                total_AR_dmso,
                total_cells_dmso - total_AR_dmso,
                AR_pct_dmso)

    # ------------------------------------------------------------
    # STEP 2 — Safe grouping by unique concentration per compound
    # ------------------------------------------------------------
    combos = (
        test_df[["cpd_id", "conc", "cpd_type"]]
        .dropna(subset=["conc"])
        .drop_duplicates()
        .sort_values(["cpd_id", "conc"])
    )

    results = []
    p_vals = []

    # ------------------------------------------------------------
    # STEP 3 — Fisher test per (cpd_id, conc)
    # ------------------------------------------------------------
    for _, row in combos.iterrows():
        cpd_id = row["cpd_id"]
        conc = float(row["conc"])
        cpd_type = row["cpd_type"]

        subset = test_df.loc[(test_df["cpd_id"] == cpd_id) &
                             (test_df["conc"] == conc)]

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
            ],
            dtype=float,
        )

        # Clip negative (safety)
        table = np.clip(table, 0, None)

        if table.sum() == 0:
            continue

        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        p_vals.append(p_value)

        results.append(
            {
                "plate_id": subset["plate_id"].iloc[0],
                "cpd_id": cpd_id,
                "conc": conc,
                "cpd_type": cpd_type,
                "n_wells": subset.shape[0],
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
        LOGGER.warning("No compound–dose combinations to test.")
        return results_df

    # ------------------------------------------------------------
    # Add q-values
    # ------------------------------------------------------------
    results_df["q_value"] = benjamini_hochberg(np.array(p_vals))

    LOGGER.info("Fisher testing completed for %d compound–dose combinations.",
                results_df.shape[0])
    return results_df



# ---------------------------------------------------------------------------
# Plotting and HTML report
# ---------------------------------------------------------------------------

def df_to_html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    """
    Convert a DataFrame into a safe HTML <table> string with limited rows.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to render.
    max_rows : int
        Maximum number of rows to display in the HTML summary.

    Returns
    -------
    str
        HTML <table> element as a string.
    """
    if df is None or df.empty:
        return "<p>No data available.</p>"

    df_show = df.head(max_rows).copy()
    html = df_show.to_html(
        index=False,
        border=0,
        classes="qc-table",
        escape=False,
    )
    if len(df) > max_rows:
        html += f"<p>Showing first {max_rows} of {len(df)} rows…</p>"
    return html


def qc_section_html(
    *,
    qc_results: Dict[str, pd.DataFrame],
    intensity_qc: pd.DataFrame | None,
    pdf_path: Path,
) -> str:
    """
    Create the HTML block for object-level QC.

    Parameters
    ----------
    qc_results : dict
        Dictionary from object_level_qc() with keys:
            image_qc
            well_qc
            flagged_wells
    intensity_qc : pandas.DataFrame or None
        Intensity QC table.
    pdf_path : Path
        Path to the QC PDF report.

    Returns
    -------
    str
        HTML string representing the QC section.
    """

    parts = []
    parts.append("<h2>Object-Level QC Summary</h2>")

    # Link to PDF
    if pdf_path.exists():
        parts.append(
            f"<p><a href='{pdf_path.name}' target='_blank'>Download full QC PDF report</a></p>"
        )

    # Image QC table
    parts.append("<h3>Image-Level QC (first rows)</h3>")
    parts.append(df_to_html_table(qc_results.get("image_qc")))

    # Well QC table
    parts.append("<h3>Well-Level QC Summary</h3>")
    parts.append(df_to_html_table(qc_results.get("well_qc")))

    # Flagged wells
    flagged = qc_results.get("flagged_wells")
    if flagged is not None and not flagged.empty:
        parts.append("<h3>Flagged Wells (Potential Issues)</h3>")
        parts.append(df_to_html_table(flagged))
    else:
        parts.append("<h3>No flagged wells detected.</h3>")

    # Intensity QC
    if intensity_qc is not None:
        parts.append("<h3>Intensity QC Summary</h3>")
        parts.append(df_to_html_table(intensity_qc))
    else:
        parts.append("<h3>No intensity QC available.</h3>")

    return "\n".join(parts)


def plot_inducer_boxplot_per_compound(
    *,
    df: pd.DataFrame,
    output_path: Path,
    ar_col: str = "AR_percent",
    cpd_type_col: str = "cpd_type",
    cpd_col: str = "cpd_id",
):
    """
    Create a box plot comparing AR% for each compound vs DMSO without
    producing gigantic figures. Automatically caps the figure width and
    switches to rotated, compact labels.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    df_dmso = df[df[cpd_type_col] == "DMSO"].copy()
    df_ind = df[df[cpd_type_col] == "TEST"].copy()

    if df_dmso.empty or df_ind.empty:
        LOGGER.warning(
            "Cannot create compound-wise inducer boxplot: missing DMSO or TEST wells."
        )
        return

    # Unique compounds
    unique_cpds = (
        df_ind[cpd_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    unique_cpds = [c for c in unique_cpds if c.lower() not in {"nan", "none", ""}]
    unique_cpds = sorted(unique_cpds)

    groups = ["DMSO"] + unique_cpds
    data = [df_dmso[ar_col].dropna().values]
    for cpd in unique_cpds:
        data.append(df_ind[df_ind[cpd_col] == cpd][ar_col].dropna().values)

    num_groups = len(groups)

    # -----------------------------------------------------------------
    # SAFETY FIX: cap width to max 40 inches (≈4800 px at 120 dpi)
    # -----------------------------------------------------------------
    max_width_inches = 12.0
    width_per_group = 0.4  # smaller spacing to fit more
    fig_width = min(max_width_inches, max(8.0, num_groups * width_per_group))

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bp = ax.boxplot(
        data,
        labels=groups,
        patch_artist=True,
        showfliers=False,
        widths=0.3,
    )

    colors = ["lightgrey"] + ["skyblue"] * (len(groups) - 1)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jitter points
    rng = np.random.default_rng(seed=42)
    for i, values in enumerate(data, start=1):
        x_jitter = rng.normal(loc=i, scale=0.04, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            alpha=0.5,
            s=18,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_ylabel("AR percentage")
    ax.set_title("Compound-wise AR Inducers Compared with DMSO")

    # SAFETY FIX: rotate labels to avoid huge horizontal size
    plt.xticks(rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    LOGGER.info(
        "Saved safe-size compound-wise inducer boxplot to %s (groups=%d)",
        output_path,
        num_groups,
    )


def plot_inducer_boxplot_per_compound_OLD(
    *,
    df: pd.DataFrame,
    output_path: Path,
    ar_col: str = "AR_percent",
    cpd_type_col: str = "cpd_type",
    cpd_col: str = "cpd_id",
):
    """
    Create a box plot comparing AR% for each inducing compound against DMSO.

    One box is drawn per compound. All points are plotted with jitter.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-well summary containing AR measurements.
    output_path : Path
        Path to save the resulting PNG.
    ar_col : str
        Column name containing AR percentage values.
    cpd_type_col : str
        Column specifying compound type (e.g. 'TEST', 'DMSO', etc.).
    cpd_col : str
        Column specifying compound identifier.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract groups
    df_dmso = df[df[cpd_type_col] == "DMSO"].copy()
    df_ind = df[df[cpd_type_col] == "TEST"].copy()

    if df_dmso.empty or df_ind.empty:
        LOGGER.warning("Cannot create compound-wise inducer boxplot: missing DMSO or TEST wells.")
        return

    # Unique inducing compounds
    # Clean compound IDs: drop NaNs, convert to string
    unique_cpds = (
        df_ind[cpd_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Optional: remove any weird leftovers like 'nan', '', 'None'
    unique_cpds = [c for c in unique_cpds if c.lower() not in {"nan", "none", ""}]

    # Sort safely as strings
    unique_cpds = sorted(unique_cpds)

    # Data list: first DMSO, then each compound
    groups = ["DMSO"] + unique_cpds

    data = [df_dmso[ar_col].dropna().values]
    for cpd in unique_cpds:
        data.append(df_ind[df_ind[cpd_col] == cpd][ar_col].dropna().values)

    # Build plot
    num_groups = len(groups)
    fig, ax = plt.subplots(figsize=(max(8, num_groups * 0.8), 6))

    bp = ax.boxplot(
        data,
        labels=groups,
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )

    # colors
    colors = ["lightgrey"] + ["skyblue"] * (len(groups) - 1)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay points with jitter
    rng = np.random.default_rng(seed=42)
    for i, values in enumerate(data, start=1):
        x_jitter = rng.normal(loc=i, scale=0.06, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            alpha=0.6,
            s=22,
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_ylabel("AR percentage")
    ax.set_title("Compound-wise AR Inducers Compared with DMSO")
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved compound-wise inducer boxplot to %s", output_path)



def plot_inducer_boxplot(
    *,
    df: pd.DataFrame,
    output_path: Path,
    ar_col: str = "AR_percent",
    cpd_type_col: str = "cpd_type",
):
    """
    Produce a box plot comparing AR% between compounds that induce the
    acrosome reaction and DMSO controls. All points are plotted with
    jitter for transparency.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-well summary containing AR measurements.
    output_path : Path
        Full path for saving the resulting PNG file.
    ar_col : str, default 'AR_percent'
        Column representing AR percentage per well.
    cpd_type_col : str, default 'cpd_type'
        Column indicating compound class ('TEST', 'DMSO', etc.).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter groups
    df_comp = df[df[cpd_type_col] == "TEST"].copy()
    df_dmso = df[df[cpd_type_col] == "DMSO"].copy()

    if df_comp.empty or df_dmso.empty:
        LOGGER.warning(
            "Cannot generate inducer boxplot: missing DMSO or TEST wells."
        )
        return

    groups = ["DMSO", "Inducers"]
    data = [
        df_dmso[ar_col].dropna().values,
        df_comp[ar_col].dropna().values,
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Box plot
    bp = ax.boxplot(
        data,
        labels=groups,
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )

    # color boxes
    for patch, color in zip(bp["boxes"], ["lightgrey", "skyblue"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jittered scatter points
    rng = np.random.default_rng(seed=42)
    for i, values in enumerate(data, start=1):
        x_jitter = rng.normal(loc=i, scale=0.04, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            alpha=0.6,
            s=20,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_ylabel("AR percentage")
    ax.set_title("Compounds that Induce AR: Comparison with DMSO")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved inducer boxplot to %s", output_path)



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


def summarise_compound_effects(
    *,
    fisher_df: pd.DataFrame,
    q_threshold: float = 0.05,
    min_increase: float = 0.0,
    max_decrease: float = 0.0,
) -> pd.DataFrame:
    """
    Summarise the overall AR effect of each compound across doses.

    For each cpd_id, doses are classified as:
      - significant increases: q_value <= q_threshold and delta_AR_pct > min_increase
      - significant decreases: q_value <= q_threshold and delta_AR_pct < -max_decrease

    The compound is then assigned an overall effect_class:
      - 'only_inducer': at least one significant increase, no significant decreases
      - 'only_suppressor': at least one significant decrease, no significant increases
      - 'mixed_effect': at least one significant increase and one significant decrease
      - 'no_significant_effect': no significant increases or decreases

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound–dose Fisher results.
    q_threshold : float, optional
        Maximum FDR q-value for a dose to be considered significant.
    min_increase : float, optional
        Minimum delta_AR_pct to classify a dose as a significant increase.
    max_decrease : float, optional
        Minimum absolute delta_AR_pct to classify a dose as a significant
        decrease (delta_AR_pct < -max_decrease).

    Returns
    -------
    pandas.DataFrame
        One row per cpd_id with columns including:
        - cpd_id
        - effect_class
        - n_doses
        - n_sig_increase
        - n_sig_decrease
        - max_delta_AR_pct
        - min_delta_AR_pct
        - max_sig_increase
        - min_sig_decrease
        - min_q_increase
        - min_q_decrease
    """
    if fisher_df.empty:
        LOGGER.warning(
            "Fisher results table is empty; per-compound summary will be empty."
        )
        return pd.DataFrame(
            columns=[
                "cpd_id",
                "effect_class",
                "n_doses",
                "n_sig_increase",
                "n_sig_decrease",
                "max_delta_AR_pct",
                "min_delta_AR_pct",
                "max_sig_increase",
                "min_sig_decrease",
                "min_q_increase",
                "min_q_decrease",
            ]
        )

    records: List[Dict[str, object]] = []

    for cpd_id, group in fisher_df.groupby("cpd_id"):
        g = group.copy()

        # All doses for this compound
        n_doses = int(g.shape[0])

        # Significant increases and decreases
        sig_inc = g[
            (g["q_value"] <= q_threshold)
            & (g["delta_AR_pct"] > min_increase)
        ]
        sig_dec = g[
            (g["q_value"] <= q_threshold)
            & (g["delta_AR_pct"] < -max_decrease)
        ]

        n_sig_inc = int(sig_inc.shape[0])
        n_sig_dec = int(sig_dec.shape[0])

        # Overall effect class
        if n_sig_inc == 0 and n_sig_dec == 0:
            effect_class = "no_significant_effect"
        elif n_sig_inc > 0 and n_sig_dec == 0:
            effect_class = "only_inducer"
        elif n_sig_inc == 0 and n_sig_dec > 0:
            effect_class = "only_suppressor"
        else:
            effect_class = "mixed_effect"

        max_delta = float(g["delta_AR_pct"].max())
        min_delta = float(g["delta_AR_pct"].min())

        max_sig_inc = float(sig_inc["delta_AR_pct"].max()) if n_sig_inc > 0 else np.nan
        min_sig_dec = float(sig_dec["delta_AR_pct"].min()) if n_sig_dec > 0 else np.nan

        min_q_inc = float(sig_inc["q_value"].min()) if n_sig_inc > 0 else np.nan
        min_q_dec = float(sig_dec["q_value"].min()) if n_sig_dec > 0 else np.nan

        records.append(
            {
                "cpd_id": cpd_id,
                "effect_class": effect_class,
                "n_doses": n_doses,
                "n_sig_increase": n_sig_inc,
                "n_sig_decrease": n_sig_dec,
                "max_delta_AR_pct": max_delta,
                "min_delta_AR_pct": min_delta,
                "max_sig_increase": max_sig_inc,
                "min_sig_decrease": min_sig_dec,
                "min_q_increase": min_q_inc,
                "min_q_decrease": min_q_dec,
            }
        )

    summary_df = pd.DataFrame(records)
    LOGGER.info(
        "Per-compound summary created for %d compounds.", summary_df.shape[0]
    )
    return summary_df


def rank_compounds(
    *,
    fisher_df: pd.DataFrame,
    score_col: str = "delta_AR_pct",
    n: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rank compounds by a chosen statistic and return:
      - the full ranked table
      - the top N compound IDs

    Ranking defaults to descending delta_AR_pct.
    """
    ranked = (
        fisher_df
        .dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )

    # Get top N compound IDs
    topN_list = ranked["cpd_id"].head(n).tolist()

    return ranked, topN_list


def plot_compound_effect_classes_barplot(
    *,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot a bar chart of the number of compounds in each effect class.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Per-compound summary table from summarise_compound_effects().
        Must contain an 'effect_class' column.
    output_path : Path
        Path to save the PNG figure.
    """
    if summary_df.empty or "effect_class" not in summary_df.columns:
        LOGGER.warning(
            "Per-compound summary is empty or missing 'effect_class'; "
            "skipping effect-class barplot."
        )
        return

    counts = summary_df["effect_class"].value_counts()
    # Optional ordering for a more interpretable plot
    order = [
        "only_inducer",
        "only_suppressor",
        "mixed_effect",
        "no_significant_effect",
    ]
    labels = [cls for cls in order if cls in counts.index]
    values = [int(counts[cls]) for cls in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)

    ax.set_xlabel("Effect class")
    ax.set_ylabel("Number of compounds")
    ax.set_title("Per-compound AR effect classes")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    LOGGER.info(
        "Saved per-compound effect-class barplot to '%s'", output_path
    )


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


def resolve_ar_column(df: pd.DataFrame) -> str:
    """
    Return the best AR% column available in a DataFrame.

    Tries, in order:
        1. 'AR_pct_compound'  (from Fisher)
        2. 'AR_pct_well'      (from per-well QC)
        3. 'AR_percent'       (older scripts)
        4. 'AR_pct'           (legacy)

    Raises KeyError if none exist.
    """
    candidates = [
        "AR_pct_compound",
        "AR_pct_well",
        "AR_percent",
        "AR_pct",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "No AR% column present. Tried: " + ", ".join(candidates)
    )


# --------------------------------------------------------------------------------------
# Helper: 4-parameter logistic
# --------------------------------------------------------------------------------------
def four_param_logistic(x: np.ndarray, top: float, bottom: float,
                        ec50: float, hill: float) -> np.ndarray:
    """
    Four-parameter logistic (4PL) model.

    Parameters
    ----------
    x : numpy.ndarray
        Concentrations (linear scale).
    top : float
        Maximum response value.
    bottom : float
        Minimum response value.
    ec50 : float
        EC50 value.
    hill : float
        Hill slope.

    Returns
    -------
    numpy.ndarray
        Model-predicted response.
    """
    return bottom + (top - bottom) / (1.0 + (x / ec50) ** hill)


# --------------------------------------------------------------------------------------
# FITTER
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# PLOTTER: INCLUDES DMSO BAND, MEDIAN, GREY MARKERS, SIGNIFICANCE MASK
# --------------------------------------------------------------------------------------


def fit_dose_response_per_compound(
    *,
    fisher_df: pd.DataFrame,
    min_drc_points: int = 3,
) -> pd.DataFrame:
    """
    Fit a four-parameter logistic (4PL) dose–response model to each compound.

    This function:
      • Excludes all DMSO wells from fitting.
      • Ensures concentrations are numeric.
      • Requires a minimum number of unique doses.
      • Supports both activators and suppressors by allowing
        positive or negative Hill slopes.
      • Provides robust initial parameter guesses.
      • Uses bounded optimisation to avoid unstable fits.
      • Computes r² safely and never returns NaN.

    Parameters
    ----------
    fisher_df : pandas.DataFrame
        Per-compound / per-dose Fisher results. Must contain:
        ['cpd_id', 'conc', 'AR_pct_compound', 'cpd_type'].
    min_drc_points : int, default 3
        Minimum number of distinct non-control concentrations required
        to attempt a fit.

    Returns
    -------
    pandas.DataFrame
        One row per compound with:
        cpd_id, fit_success, EC50, top, bottom, hill,
        max_delta_AR_pct, r_squared, effect_class.
    """

    # ---------------------------
    # 4PL MODEL
    # ---------------------------
    def logistic_4pl(x, bottom, ec50, hill, top):
        """
        Standard four-parameter logistic model.

        The model supports both increasing (hill > 0)
        and decreasing (hill < 0) curves.
        """
        return bottom + (top - bottom) / (1.0 + (x / ec50) ** hill)

    # Output records
    results: List[Dict[str, object]] = []

    # Group Fisher results by compound
    for cpd_id, sub in fisher_df.groupby("cpd_id"):
        rec: Dict[str, object] = {
            "cpd_id": cpd_id,
            "fit_success": False,
            "EC50": float("nan"),
            "top": float("nan"),
            "bottom": float("nan"),
            "hill": float("nan"),
            "max_delta_AR_pct": float("nan"),
            "r_squared": float("nan"),
            "effect_class": "none",
        }

        # ----------------------------------------------
        # 1. Remove DMSO – must NEVER enter the fit
        # ----------------------------------------------
        sub = sub[sub["cpd_type"] != "DMSO"].copy()
        if sub.empty:
            results.append(rec)
            continue

        # ----------------------------------------------
        # 2. Ensure numeric concentration
        # ----------------------------------------------
        sub["conc"] = pd.to_numeric(sub["conc"], errors="coerce")
        sub = sub.dropna(subset=["conc"])

        if sub.empty:
            results.append(rec)
            continue

        # ----------------------------------------------
        # 3. Require ≥ min_drc_points unique doses
        # ----------------------------------------------
        unique_doses = np.sort(sub["conc"].unique())
        if len(unique_doses) < min_drc_points:
            results.append(rec)
            continue

        # ----------------------------------------------
        # 4. Extract x and y
        # ----------------------------------------------
        x = unique_doses
        # Aggregate AR% across wells for each concentration
        y = (
            sub.groupby("conc")["AR_pct_compound"]
            .mean()
            .reindex(x)
            .values
        )

        # Ecological check — must not contain NaN
        if np.isnan(x).any() or np.isnan(y).any():
            results.append(rec)
            continue

        # ----------------------------------------------
        # 5. Determine whether increasing or decreasing
        # ----------------------------------------------
        is_increasing = y[-1] > y[0]

        # ----------------------------------------------
        # 6. Parameter initialisation
        # ----------------------------------------------
        bottom_init = float(np.min(y))
        top_init = float(np.max(y))
        ec50_init = float(np.median(x))
        hill_init = 1.0 if is_increasing else -1.0

        p0 = [bottom_init, ec50_init, hill_init, top_init]

        # ----------------------------------------------
        # 7. Parameter bounds
        # ----------------------------------------------
        # bottom between 0–100
        # top between 0–100
        # EC50 positive
        # hill between -5 and +5
        bounds = (
            [0.0, 1e-12, -5.0, 0.0],     # lower
            [100.0, 1e12, 5.0, 100.0],   # upper
        )

        # ----------------------------------------------
        # 8. Attempt fitting
        # ----------------------------------------------
        try:
            popt, _ = curve_fit(
                logistic_4pl,
                x,
                y,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )
            bottom, ec50, hill, top = popt

        except Exception:
            # Fit failed
            results.append(rec)
            continue

        # ----------------------------------------------
        # 9. Compute predictions and r²
        # ----------------------------------------------
        y_pred = logistic_4pl(x, bottom, ec50, hill, top)

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0.0:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        # ----------------------------------------------
        # 10. Determine effect class
        # ----------------------------------------------
        delta = y - np.mean(y)
        max_delta = float(np.max(delta))

        if is_increasing:
            effect_class = "activator"
        elif not is_increasing:
            effect_class = "suppressor"
        else:
            effect_class = "none"

        # ----------------------------------------------
        # 11. Populate record
        # ----------------------------------------------
        # r-squared clean formatting here
        r2_clean = round(float(r2), 3)
        rec.update(
            {
                "fit_success": True,
                "EC50": float(ec50),
                "top": float(top),
                "bottom": float(bottom),
                "hill": float(hill),
                "max_delta_AR_pct": max_delta,
                "r_squared": r2_clean,
                "effect_class": effect_class,
            }
        )

        results.append(rec)

    return pd.DataFrame(results)




def plot_dose_response_examples(
    *,
    df_well_qc: pd.DataFrame,
    fisher_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    output_dir: Path,
    max_plots: int = 2000,
) -> List[Path]:
    """
    Generate per-compound dose–response plots including:
    - DMSO points (grey)
    - DMSO median line and IQR band
    - Observed compound AR%
    - 4PL fit curve
    - Significance (red = q < 0.05 AND outside IQR)

    Parameters
    ----------
    df_well_qc : pandas.DataFrame
        Per-well QC-filtered table.
    fisher_df : pandas.DataFrame
        Fisher per-dose aggregated table.
    fit_df : pandas.DataFrame
        Output of fit_dose_response_per_compound().
    output_dir : pathlib.Path
        Directory for saving plots.
    max_plots : int, optional
        Maximum number of compounds to plot.

    Returns
    -------
    list[pathlib.Path]
        Paths to saved PNG plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # Extract DMSO baseline (OPTION A)
    dmso = df_well_qc[df_well_qc["cpd_type"] == "DMSO"]
    if dmso.empty:
        raise ValueError("No DMSO wells found for plotting baseline.")

    dmso_vals = dmso["AR_pct_well"].astype(float)
    dmso_median = dmso_vals.median()
    dmso_q1 = dmso_vals.quantile(0.25)
    dmso_q3 = dmso_vals.quantile(0.75)

    # Merge fits
    merged = fisher_df.merge(fit_df, on="cpd_id", how="left")

    # Unique compounds to plot
    compounds = merged["cpd_id"].unique()[:max_plots]

    for cpd_id in compounds:
        sub = merged[merged["cpd_id"] == cpd_id].sort_values("conc").copy()
        if sub.empty:
            continue

        conc = sub["conc"].values.astype(float)
        ar = sub["AR_pct_compound"].values.astype(float)
        qvals = sub["q_value"].fillna(1.0).values

        # Determine significance (correct rule)
        valid = ~np.isnan(ar)
        sig_mask = valid & (qvals < 0.05) & ((ar < dmso_q1) | (ar > dmso_q3))

        # Prepare figure
        fig, ax = plt.subplots(figsize=(6, 4))

        # DMSO band
        ax.axhspan(dmso_q1, dmso_q3, color="lightgrey", alpha=0.4)
        ax.axhline(dmso_median, color="grey", ls="--", lw=1)

        # DMSO points  -- uncomment if we want them back. 
        #ax.scatter(
        #    np.full(dmso_vals.size, conc.min()*0.8),
        #    dmso_vals,
        #    s=20,
        #    color="grey",
        #    alpha=0.7,
        #    label="DMSO wells",
        # )

        # Observed data
        for x, y, is_sig in zip(conc, ar, sig_mask):
            color = "red" if is_sig else "black"
            ax.scatter(x, y, s=50, color=color)

        # 4PL curve
        fit_row = fit_df[fit_df["cpd_id"] == cpd_id]
        if not fit_row.empty and fit_row.iloc[0]["fit_success"]:
            top = fit_row.iloc[0]["top"]
            bottom = fit_row.iloc[0]["bottom"]
            ec50 = fit_row.iloc[0]["EC50"]
            hill = fit_row.iloc[0]["hill"]

            xfit = np.logspace(np.log10(conc.min()),
                               np.log10(conc.max()), 200)
            yfit = four_param_logistic(xfit, top, bottom, ec50, hill)
            ax.plot(xfit, yfit, color="blue", lw=2)

            title = f"{cpd_id}  (EC50 = {ec50:.2e})"
        else:
            title = f"{cpd_id}  (fit failed)"

        ax.set_xscale("log")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("AR% (aggregated)")
        ax.set_title(title)

        out = output_dir / f"{cpd_id}_drc.png"
        # ADD THIS TO AUTOSCALE
        # ax.autoscale(enable=True, axis="y", tight=False)
        # ax.set_ylim(0, 50)

        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)

        paths.append(out)

    return paths



def qc_summary_for_metadata(*, df_meta: pd.DataFrame) -> None:
    """
    Print QC summaries for harmonised metadata.

    This includes:
        • unique plate barcodes
        • wells per barcode
        • compounds per plate
        • missing or malformed well identifiers

    Parameters
    ----------
    df_meta : pandas.DataFrame
        Harmonised metadata table containing at least:
        plate_id, well_id, cpd_id.
    """
    LOGGER.info("---- Metadata QC summary ----")

    # --- Unique barcodes ---
    plates = sorted(df_meta["plate_id"].unique())
    LOGGER.info("Unique plate barcodes (%d): %s", len(plates), ", ".join(plates))

    # --- Wells per barcode ---
    LOGGER.info("Wells per plate barcode:")
    wells_per = (
        df_meta.groupby("plate_id")["well_id"]
        .nunique()
        .sort_index()
        .to_dict()
    )
    for plate, n in wells_per.items():
        LOGGER.info("    %s: %d wells", plate, n)

    # --- Compounds per plate ---
    LOGGER.info("Compounds per plate:")
    compounds_per = (
        df_meta.groupby("plate_id")["cpd_id"]
        .nunique()
        .sort_index()
        .to_dict()
    )
    for plate, n in compounds_per.items():
        LOGGER.info("    %s: %d compounds", plate, n)

    # --- Missing or malformed well IDs ---
    malformed_mask = (
        df_meta["well_id"].isna()
        | ~df_meta["well_id"].astype(str).str.match(r"^[A-H][0-9]{2}$")
    )

    malformed_rows = df_meta[malformed_mask]

    LOGGER.info(
        "Malformed or missing wells: %d rows",
        malformed_rows.shape[0],
    )

    if not malformed_rows.empty:
        LOGGER.info(
            "Examples of malformed wells:\n%s",
            malformed_rows.head(10).to_string(index=False),
        )

    LOGGER.info("---- End metadata QC summary ----")



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


def load_motility_csv(path: Path) -> pd.DataFrame:
    """
    Load motility data (HA, PM, TM, VCL metrics) from a CSV/TSV file.

    The file must contain:
        - cpd_id
        - HA_value, HA_pc
        - PM_value, PM_pc
        - TM_value, TM_pc
        - VCL_median_value, VCL_median_pc

    Any missing columns will raise an error.

    Parameters
    ----------
    path : Path
        Path to CSV or TSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with motility metrics, filtered to required columns.
    """

    # Auto-detect delimiter (tab or comma)
    df = pd.read_csv(path, sep=None, engine="python")

    required_cols = [
        "cpd_id",
        "HA_value", "HA_pc",
        "PM_value", "PM_pc",
        "TM_value", "TM_pc",
        "VCL_median_value", "VCL_median_pc",
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Motility file missing required columns: {missing}")

    # Only keep required columns
    df = df[required_cols].copy()

    # Drop rows with missing cpd_id (e.g. DMSO wells)
    df = df.dropna(subset=["cpd_id"])

    return df


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
    inducer_boxplot_png: Path | None = None,
    inducer_per_cpd_boxplot_png: Path | None = None,
    compound_summary_tsv: Path | None = None,
    class_barplot_png: Path | None = None,
    qc_results: Dict[str, pd.DataFrame] | None = None,
    intensity_qc: pd.DataFrame | None = None,
    qc_pdf_path: Path | None = None,
) -> Path:
    """
    Write a simple HTML report linking TSVs and embedding plots and previews.

    The report includes:
      - Download links for all main TSV outputs.
      - Embedded, scrollable HTML previews of the TSV tables
        (first N rows only).
      - Static PNG plots and optional interactive volcano link.
      - Optional dose–response curve PNGs.
      - Optional barplot of compounds that only induce AR.
      - Optional per-compound effect-class barplot.

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
    inducing_barplot_png : Path or None, optional
        Path to “only inducers” barplot PNG, if generated.
    compound_summary_tsv : Path or None, optional
        Path to per-compound summary TSV, if generated.
    class_barplot_png : Path or None, optional
        Path to per-compound effect-class barplot PNG, if generated.

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
    if compound_summary_tsv is not None and compound_summary_tsv.exists():
        parts.append(
            f"<li>Per-compound effect summary: "
            f"<a href='{compound_summary_tsv.name}'>{compound_summary_tsv.name}</a></li>"
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
    if compound_summary_tsv is not None and compound_summary_tsv.exists():
        parts.append(
            tsv_preview_html(
                path=compound_summary_tsv,
                max_rows=100,
                title="Per-compound effect summary",
            )
        )

    # Plate heatmap
    if plate_heatmap_png is not None and plate_heatmap_png.exists():
        parts.append("<h2>Plate layout – AR% per well (QC-kept)</h2>")
        parts.append(f"<img src='{plate_heatmap_png.name}' alt='Plate heatmap'>")

    # ---------------------------------------------------------
    # OBJECT-LEVEL QC SECTION
    # ---------------------------------------------------------
    if qc_results is not None and qc_results:
        parts.append("<h2>Object-Level QC Summary</h2>")

        if qc_pdf_path is not None and qc_pdf_path.exists():
            parts.append(
                f"<p><a href='{qc_pdf_path.name}' target='_blank'>"
                "Download full QC PDF report</a></p>"
            )

        # Image QC
        parts.append("<h3>Image-Level QC</h3>")
        parts.append(df_to_html_table(qc_results.get("image_qc")))

        # Well QC
        parts.append("<h3>Well-Level QC</h3>")
        parts.append(df_to_html_table(qc_results.get("well_qc")))

        # Flagged wells
        flagged = qc_results.get("flagged_wells")
        if flagged is not None and not flagged.empty:
            parts.append("<h3>Flagged Wells</h3>")
            parts.append(df_to_html_table(flagged))
        else:
            parts.append("<p>No flagged wells detected.</p>")

        # Intensity QC
        if intensity_qc is not None:
            parts.append("<h3>Intensity QC Summary</h3>")
            parts.append(df_to_html_table(intensity_qc))



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

    # “Only inducers” barplot
    if inducer_per_cpd_boxplot_png is not None and inducer_per_cpd_boxplot_png.exists():
        parts.append("<h2>Compound-wise AR Inducers</h2>")
        parts.append(f"<img src='{inducer_per_cpd_boxplot_png.name}' "
                    f"alt='Compound-wise inducer boxplot'>")

    if inducing_barplot_png is not None and inducing_barplot_png.exists():
        parts.append("<h2>Compounds that only induce AR</h2>")
        parts.append(
            "<p>Compounds shown here have no doses that reduce AR below the "
            "DMSO level, and at least one significant dose (FDR ≤ 0.05) that "
            "increases AR. Bars reflect the dose with the largest ΔAR%.</p>"
        )
        parts.append(
            f"<img src='{inducing_barplot_png.name}' "
            f"alt='Compounds that only induce AR'>"
        )

    # “Only inducers” boxplot
    if inducer_boxplot_png is not None and inducer_boxplot_png.exists():
        parts.append("<h2>Compounds That Only Induce AR</h2>")
        parts.append(f"<img src='{inducer_boxplot_png.name}' alt='Inducer boxplot'>")



    # Per-compound effect-class barplot
    if class_barplot_png is not None and class_barplot_png.exists():
        parts.append("<h2>Summary of per-compound effect classes</h2>")
        parts.append(
            "<p>Counts of compounds classified as only inducers, only "
            "suppressors, mixed effect, or having no significant effect.</p>"
        )
        parts.append(
            f"<img src='{class_barplot_png.name}' "
            f"alt='Per-compound effect class summary'>"
        )

    # DRC curves
    if drc_pngs:
        parts.append("<h2>Example dose–response curves</h2>")
        for png in drc_pngs:
            parts.append(f"<h3>{png.stem}</h3>")
            rel_path = png.relative_to(output_dir)
            parts.append(f"<img src='{rel_path.as_posix()}' alt='{png.stem}'>")


    # ---------------------------------------------------------
    # Optional motility metrics section (HA, PM, TM, VCL)
    # ---------------------------------------------------------
    if compound_summary_tsv is not None and compound_summary_tsv.exists():
        try:
            df_summary = pd.read_csv(compound_summary_tsv, sep="\t")

            motility_cols = [
                "cpd_id",
                "HA_value", "HA_pc",
                "PM_value", "PM_pc",
                "TM_value", "TM_pc",
                "VCL_median_value", "VCL_median_pc",
            ]

            # Identify columns actually present in the TSV
            available = [c for c in motility_cols if c in df_summary.columns]

            # Only show section if at least one motility metric exists
            if len(available) > 1:  # must include cpd_id + ≥1 metric
                parts.append("<hr>")
                parts.append("<h2>Additional Motility Readouts (HA, PM, TM, VCL)</h2>")
                parts.append(
                    "<p>These values were supplied from a separate motility "
                    "assay and merged by compound ID. Missing values indicate "
                    "no corresponding measurement for that compound.</p>"
                )

                motility_html = (
                    df_summary[available]
                    .sort_values("cpd_id")
                    .to_html(index=False, float_format="%.3f")
                )
                parts.append(motility_html)
                parts.append("<hr>")

        except Exception as err:
            LOGGER.error(
                "Failed to render motility section in HTML report: %s", err
            )


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
    configure_logging(verbosity=args.verbosity, output_dir=args.output_dir)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    LOGGER.info("Starting acrosome dose–response analysis")

    df_acrosome = None
    df_spermcells = None
    df_nuclei = None
    fit_df = None
    mot_df = None


    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)



    cp_dir = Path(args.cp_dir)
    lib_path = Path(args.library_metadata)

    # Controls may be omitted (multi-plate runs)
    if args.controls is None:
        df_ctrl = None
    else:
        ctrl_path = Path(args.controls)
        df_ctrl = load_controls(controls_path=ctrl_path)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover and load CellProfiler Image table
    cp_files = discover_cp_files(cp_dir=cp_dir)

    # Required
    image_path = cp_files["image"]
    df_image = load_image_counts(
        image_path=image_path,
        plate_col=args.image_plate_col,
        well_col=args.image_well_col,
        cell_count_col=args.image_cell_count_col,
        ar_count_col=args.image_ar_count_col,
    )

    # Normalise plate_id suffixes (e.g. 24AP3815_15082024 → 24AP3815)
    if df_image["plate_id"].astype(str).str.contains("_").any():
        before = df_image["plate_id"].unique()[:10]
        df_image["plate_id"] = df_image["plate_id"].astype(str).str.split("_").str[0]
        after = df_image["plate_id"].unique()[:10]
        LOGGER.warning(
            "Plate IDs in the Image table contained suffixes and were normalised.\n"
            "Example before: %s\n"
            "Example after : %s",
            before,
            after,
        )


    # Optional: load object-level acrosome table
    acrosome_path = cp_files["acrosome"]

    if acrosome_path is not None:
        df_acrosome = load_acrosome_minimal(
            path=acrosome_path,
            cols=["ImageNumber"],   # add more columns later if needed
        )
        LOGGER.info("Loaded object-level acrosome table with %d rows", len(df_acrosome))

    if cp_files["spermcells"] is not None:
        df_spermcells = load_acrosome_minimal(
            path=cp_files["spermcells"],
            cols=["ImageNumber"]
        )
        LOGGER.info("Loaded object-level sperm cells table with %d rows", len(df_spermcells))
    else:
        df_spermcells = None

    if cp_files["filterednuclei"] is not None:
        df_nuclei = load_acrosome_minimal(
            path=cp_files["filterednuclei"],
            cols=["ImageNumber"]
        )
        LOGGER.info("Loaded object-level filtered nuclei table with %d rows", len(df_nuclei))
    else:
        df_nuclei = None



    # 2. Load metadata and controls
    df_meta = load_compound_metadata(metadata_path=lib_path)
    qc_summary_for_metadata(df_meta=df_meta)

    # ---------------------------------------------------------
    # Optional: load motility metrics (one row per cpd_id)
    # ---------------------------------------------------------
    mot_df = None
    if args.motility_csv is not None:
        LOGGER.info("Loading motility data from '%s'", args.motility_csv)
        try:
            mot_df = load_motility_csv(path=args.motility_csv)
            LOGGER.info(
                "Loaded motility metrics (%d rows, %d columns)",
                mot_df.shape[0], mot_df.shape[1],
            )
        except Exception as err:
            LOGGER.error("Failed to load motility file: %s", err)
            mot_df = None



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

    inducer_per_cpd_boxplot_png = Path(args.output_dir) / "inducer_boxplot_per_compound.png"

 
    plot_inducer_boxplot_per_compound(
        df=df_well_qc[df_well_qc["qc_keep"]].rename(columns={"AR_pct_well": "AR_percent"}),
        output_path=inducer_per_cpd_boxplot_png,
        ar_col="AR_percent",
        cpd_type_col="cpd_type",
        cpd_col="cpd_id",
    )


    # ---------------------------------------------------------
    # OBJECT-LEVEL QC
    # ---------------------------------------------------------
    qc_results = object_level_qc(
        df_image=df_image,
        df_acrosome=df_acrosome,
        df_sperm=df_spermcells if "df_spermcells" in locals() else None,
        df_nuclei=df_nuclei if "df_nuclei" in locals() else None,
    )
    intensity_qc_table = intensity_qc(
            df_acrosome=df_acrosome,
            df_image=df_image,
        )


    if qc_results:
        # Save QC tables inside output_dir
        qc_out = Path(args.output_dir) / "object_qc"
        qc_out.mkdir(exist_ok=True, parents=True)

        for name, df in qc_results.items():
            out_path = qc_out / f"{name}.tsv"
            df.to_csv(out_path, sep="\t", index=False)
            LOGGER.info("Wrote object-level QC: %s", out_path)


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


    write_qc_pdf(
        output_path=Path(args.output_dir)/"object_qc"/"QC_report.pdf",
        image_qc=qc_results["image_qc"],
        well_qc=qc_results["well_qc"],
        intensity_qc=intensity_qc_table,
    )


    LOGGER.info("DEBUG: Checking conc column BEFORE Fisher test:")
    LOGGER.info(
        "Unique conc values sample: %s",
        df_well_qc["conc"].dropna().unique()[:20]
    )

    # 6. Fisher tests per compound–dose
    fisher_df = run_fisher_per_compound_dose(df_well_qc=df_well_qc)

    logging.info("DEBUG Fisher columns: %s", fisher_df.columns.tolist())

    # ==========================================
    # DEBUG: Inspect rows per compound in fisher_df
    # ==========================================
    LOGGER.info("DEBUG: Inspecting rows per compound in fisher_df...")

    # ---------------------------------------------------------
    # Safe handling for single-replicate datasets (no conc/empty Fisher)
    # ---------------------------------------------------------
    if fisher_df is None or fisher_df.empty or "cpd_id" not in fisher_df.columns:
        LOGGER.warning("Fisher results empty or missing cpd_id column. "
                    "This looks like a single-replicate dataset. "
                    "Skipping dose-response and Fisher-dependent plots.")


        # Determine which AR% column exists

        # Automatically detect the correct AR% column for single-rep datasets
        possible_ar_cols = [
            "AR_pct_compound", "AR_pct", "AR_pct_well", "AR_percent",
            "AR_pct_x", "AR_pct_y", "AR_QC_pct"
        ]

        ar_col = None
        for col in possible_ar_cols:
            if col in df_well_qc.columns:
                ar_col = col
                break

        if ar_col is None:
            raise KeyError(
                f"No AR% column found in df_well_qc. "
                f"Available columns: {df_well_qc.columns.tolist()}"
            )

        single_rep_summary = (
            df_well_qc.groupby("cpd_id", as_index=False)
            .agg(
                AR_pct_mean=(ar_col, "mean"),
                AR_pct_median=(ar_col, "median"),
                n_wells=("cpd_id", "count"),
            )
        )

        out_path = output_dir / "acrosome_single_rep_summary.tsv"
        single_rep_summary.to_csv(out_path, sep="\t", index=False)
        LOGGER.info("Wrote single-replicate summary to '%s'", out_path)

        # Do NOT return – allow HTML creation to continue
        # but skip all Fisher/DRC-dependent sections
        skip_fisher_drc = True

        # Skip remaining Fisher/dose-response logic
        return


    # ---------------------------------------------------------
    # Optional debug: inspect fisher_df per compound
    # ---------------------------------------------------------
    if fisher_df is not None and not fisher_df.empty and "cpd_id" in fisher_df.columns:
        unique_cpds = fisher_df["cpd_id"].unique()
        LOGGER.info("DEBUG: First 10 compounds in fisher_df: %s", unique_cpds[:10].tolist())
    else:
        LOGGER.info("DEBUG: Skipping fisher_df inspection (empty or missing cpd_id).")

    for c in unique_cpds[:10]:  # first 10 compounds only
        sub = fisher_df[fisher_df["cpd_id"] == c]
        LOGGER.info(
            "CPD %s: n_rows=%d; conc_list=%s; AR_pct_compound_list=%s",
            c,
            sub.shape[0],
            list(sub["conc"].round(3)),
            list(sub["AR_pct_compound"].round(2)),
        )
    LOGGER.info("DEBUG: Finished inspecting compound rows.\n")


    # Debug: inspect AR-related columns in df_well_qc
    try:
        ar_cols = [c for c in df_well_qc.columns
                if ("AR" in c.upper()) or ("acrosome" in c.lower())]

        if not ar_cols:
            logging.warning("DEBUG: No AR-related columns found in df_well_qc!")
        else:
            logging.info("DEBUG: AR-related columns in df_well_qc: %s", ar_cols)
            logging.info("DEBUG AR column head:\n%s", df_well_qc[ar_cols].head())
    except Exception as err:
        logging.error("DEBUG: Failed to inspect AR columns: %s", err)



    # ---------------------------------------------------------
    # Rank compounds and write top-N table
    # ---------------------------------------------------------

    # Top 20 compound IDs (sorted by default: delta_AR_pct)
    top20_cpds = get_top_compounds(fisher_df=fisher_df, n=20)

    # Full ranked table + top N list
    ranked_df, top20_cpds = rank_compounds(
        fisher_df=fisher_df,
        score_col="delta_AR_pct",
        n=20,
    )

    ranked_tsv = output_dir / "compound_ranking.tsv"
    ranked_df.to_csv(ranked_tsv, sep="\t", index=False)
    LOGGER.info("Wrote ranked compound table: %s", ranked_tsv)

    # Subset well-level data for top20 compounds
    subset_df = df_well_qc[df_well_qc["cpd_id"].isin(top20_cpds)]

    # Boxplot for only top20 compounds
    plot_inducer_boxplot_per_compound(
        df=subset_df.rename(columns={"AR_pct_well": "AR_percent"}),
        output_path=output_dir / "boxplot_top20.png",
        ar_col="AR_percent",
        cpd_type_col="cpd_type",
        cpd_col="cpd_id",
    )

    # ---------------------------------------------------------
    # Per-compound boxplots (one PNG per compound)
    # ---------------------------------------------------------

    single_cpd_dir = output_dir / "boxplots_per_compound"
    single_cpd_dir.mkdir(exist_ok=True)

    df_plot = df_well_qc[df_well_qc["qc_keep"]].rename(
        columns={"AR_pct_well": "AR_percent"}
    )

    for cpd in top20_cpds:
        out_path = single_cpd_dir / f"boxplot_{cpd}.png"

        # Subset for this single compound + DMSO
        local_df = df_plot[
            (df_plot["cpd_type"] == "DMSO") |
            ((df_plot["cpd_type"] == "TEST") & (df_plot["cpd_id"] == cpd))
        ]

        LOGGER.info("Plotting DMSO vs %s (n=%d)", cpd, local_df.shape[0])

        plot_inducer_boxplot_per_compound(
            df=local_df,
            output_path=out_path,
            ar_col="AR_percent",
            cpd_type_col="cpd_type",
            cpd_col="cpd_id",
        )


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

    # Per-compound summary
    compound_summary_tsv: Path | None = None
    compound_summary_df: pd.DataFrame | None = None
    compound_summary_df = summarise_compound_effects(
        fisher_df=fisher_df,
        q_threshold=0.05,
        min_increase=0.0,
        max_decrease=0.0,
    )
    if compound_summary_df is not None and not compound_summary_df.empty:
        compound_summary_tsv = output_dir / "acrosome_per_compound_summary.tsv"
        compound_summary_df.to_csv(
            compound_summary_tsv,
            sep="\t",
            index=False,
        )
        LOGGER.info(
            "Wrote per-compound summary to '%s'", compound_summary_tsv
        )
    else:
        LOGGER.warning(
            "Per-compound summary is empty; no summary TSV will be written."
        )

    # ---------------------------------------------------------
    # Optional: merge motility metrics into per-compound summary
    # ---------------------------------------------------------
    if mot_df is not None and compound_summary_df is not None:
        LOGGER.info("Merging motility metrics into compound summary (left merge on cpd_id).")
        before = compound_summary_df.shape[0]

        LOGGER.info("Example cpd_id values from compound summary: %s",
            compound_summary_df['cpd_id'].head(10).tolist())

        LOGGER.info("Example cpd_id values from motility data: %s",
                    mot_df['cpd_id'].head(10).tolist())

        # Normalise cpd_id in mot_df
        mot_df["cpd_id"] = (
            mot_df["cpd_id"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
        )

        compound_summary_df = compound_summary_df.merge(
            mot_df,
            on="cpd_id",
            how="left",
        )

        LOGGER.info(
            "Motility merge complete. Summary rows before=%d, after=%d.",
            before,
            compound_summary_df.shape[0],
        )



    # 7. Optional dose–response fitting
    drc_tsv: Path | None = None
    drc_df: pd.DataFrame | None = None

    # 8. Plots
    plate_heatmap_png = output_dir / "plate_AR_pct_heatmap.png"
    plot_plate_heatmap(
        df_well_qc=df_well_qc[df_well_qc["qc_keep"]],
        output_path=plate_heatmap_png,
    )

    volcano_png = output_dir / "fisher_volcano.png"
    plot_volcano(fisher_df=fisher_df, output_path=volcano_png)

    volcano_interactive_html: Path | None = plot_volcano_interactive(
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
    plot_top_delta_barplot(
        fisher_df=fisher_df,
        output_path=barplot_png,
        top_n=30,
    )

    class_barplot_png: Path | None = None
    if compound_summary_df is not None and not compound_summary_df.empty:
        class_barplot_png = output_dir / "compound_effect_classes_barplot.png"
        plot_compound_effect_classes_barplot(
            summary_df=compound_summary_df,
            output_path=class_barplot_png,
        )


    inducing_barplot_png = output_dir / "inducing_compounds_barplot.png"
    plot_inducing_compounds_barplot(
        fisher_df=fisher_df,
        output_path=inducing_barplot_png,
        q_threshold=0.05,
        min_delta=0.0,
        top_n=30,
    )

    drc_pngs: List[Path] = []


    # Run dose–response plotting only if user requested it
    drc_pngs = []
    fit_df = None


    if args.fit_dose_response:
        fit_df = fit_dose_response_per_compound(fisher_df=fisher_df)


        # Merge effect class into fit_df so HTML shows the interpretation
        #if compound_summary_df is not None and not compound_summary_df.empty:
        #    fit_df = fit_df.merge(
        #        compound_summary_df[["cpd_id", "effect_class"]],
        #        on="cpd_id",
        #        how="left",
         #   )

        if compound_summary_df is not None and not compound_summary_df.empty:
            # Remove the effect_class column created during fitting
            if "effect_class" in fit_df.columns:
                fit_df = fit_df.drop(columns=["effect_class"])

            fit_df = fit_df.merge(
                compound_summary_df[["cpd_id", "effect_class"]],
                on="cpd_id",
                how="left",
            )
    
        drc_tsv = output_dir / "acrosome_drc_fits.tsv"
        fit_df.to_csv(drc_tsv, sep="\t", index=False)
        LOGGER.info("Wrote DRC fit summary to '%s'", drc_tsv)

        drc_pngs = plot_dose_response_examples(
            df_well_qc=df_well_qc,
            fisher_df=fisher_df,
            fit_df=fit_df,
            output_dir=results_dir / "dose_response_plots",
            max_plots=2000,
        )



    # ---------------------------------------------------------
    # Optional: merge motility metrics into dose–response fits
    # ---------------------------------------------------------
    if mot_df is not None and fit_df is not None:
        LOGGER.info("Merging motility metrics into dose–response fit table.")
        before_rows = fit_df.shape[0]

        fit_df = fit_df.merge(
            mot_df,
            on="cpd_id",
            how="left",
        )

        LOGGER.info(
            "Motility merge into DRC fits complete. Rows before=%d, after=%d.",
            before_rows,
            fit_df.shape[0],
        )
    else:
        LOGGER.info("Skipping motility merge into DRC fits.")


    # Inducer boxplot with jittered points and DMSO for reference
    inducer_boxplot_png = Path(args.output_dir) / "inducer_boxplot.png"

    plot_inducer_boxplot(
        df=df_well_qc[df_well_qc["qc_keep"]].rename(
            columns={"AR_pct_well": "AR_percent"}
        ),
        output_path=inducer_boxplot_png,
        ar_col="AR_percent",
        cpd_type_col="cpd_type",
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
        inducer_per_cpd_boxplot_png=inducer_per_cpd_boxplot_png,
        compound_summary_tsv=compound_summary_tsv,
        class_barplot_png=class_barplot_png,
        qc_results=qc_results,
        intensity_qc=intensity_qc_table,
        qc_pdf_path=Path(args.output_dir) / "object_qc" / "QC_report.pdf",
    )



    LOGGER.info("HTML report written to '%s'", html_report)

    LOGGER.info("Analysis complete.")


if __name__ == "__main__":
    main()