#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified acrosome reaction analysis script.

This script supports two experiment types automatically:

1. Dose–response plates
   • metadata contains multiple concentrations per compound
   • fits Hill curves
   • produces per-compound dose–response plots
   • optional Fisher tests for each dose vs DMSO

2. Single-replicate, non-dose screens (e.g. GHCDL_B1–B7 screens)
   • no concentration gradient required
   • produces per-compound AR% summary
   • Fisher tests: compound vs pooled DMSO
   • boxplot per compound (compound vs DMSO)

DMSO controls must be present in metadata as:
    cpd_id == "DMSO"

This script:
    • loads CellProfiler Image table
    • loads object-level table for QC
    • loads stamped metadata
    • merges metadata safely
    • computes per-well AR%
    • runs Fisher exact tests where possible
    • creates QC PDF
    • generates output folder with:
        - well-level TSV
        - Fisher results TSV
        - plots
        - per-plate HTML summary

All saved tabular output uses tab-separated format.

Author: ChatGPT for Pete
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import fisher_exact
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def configure_logging(*, verbosity: int) -> None:
    """
    Configure logging level.

    Parameters
    ----------
    verbosity : int
        0 = WARNING, 1 = INFO, 2 = DEBUG.
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def standardise_well_string(*, well: str) -> str:
    """
    Standardise well identifiers (A01, B12, etc.).

    Parameters
    ----------
    well : str
        Well string possibly in A1 or A001 format.

    Returns
    -------
    str
        Standardised well ID.
    """
    if not isinstance(well, str):
        return well
    well = well.strip().upper()
    if len(well) < 2:
        return well
    row = well[0]
    col = well[1:]
    try:
        col_i = int(col)
    except ValueError:
        return well
    return f"{row}{col_i:02d}"


def hill_equation(conc, bottom, top, ic50, hill):
    """
    Four-parameter logistic equation for dose–response fitting.

    Returns AR%.
    """
    return bottom + (top - bottom) / (1.0 + (ic50 / conc) ** hill)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_image_table(*, cp_dir: Path) -> pd.DataFrame:
    """
    Load the CellProfiler Image table.

    Parameters
    ----------
    cp_dir : pathlib.Path
        Folder containing MyExpt_Image.csv.gz.

    Returns
    -------
    pandas.DataFrame
        Per-image table with required fields.
    """
    files = list(cp_dir.glob("*Image*.csv*"))
    if len(files) == 0:
        raise FileNotFoundError(f"No Image table found in {cp_dir}")
    path = files[0]
    LOGGER.info("Loading Image table from '%s'", path)
    df = pd.read_csv(path)
    return df


def load_object_table_minimal(*, cp_dir: Path) -> pd.DataFrame:
    """
    Load minimal object-level data for QC.

    Parameters
    ----------
    cp_dir : pathlib.Path
        Folder containing MyExpt_Acrosome.csv.gz.

    Returns
    -------
    pandas.DataFrame
        Object table with minimal columns.
    """
    files = list(cp_dir.glob("*Acrosome*.csv*"))
    if len(files) == 0:
        LOGGER.info("No object-level table found; skipping object QC.")
        return pd.DataFrame()

    path = files[0]
    LOGGER.info("Loading minimal Acrosome columns from %s", path)

    usecols = ["ImageNumber", "AR_flag"]  # adapt if needed
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        # fallback load if AR_flag absent
        df = pd.read_csv(path)

    return df


def load_metadata(*, metadata_path: Path) -> pd.DataFrame:
    """
    Load stamped metadata and normalise required columns.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Path to PTOD/GHCDL metadata.

    Returns
    -------
    pandas.DataFrame
        Metadata with plate_id, well_id, cpd_id, conc.
    """
    LOGGER.info("Loading metadata from '%s'", metadata_path)
    df = pd.read_csv(metadata_path, sep="\t")

    df = df.rename(
        columns={
            "Plate_Metadata": "plate_id",
            "Well_Metadata": "well_id",
            "OBJDID": "cpd_id",
            "PTODCONCVALUE": "conc",
        }
    )

    # ensure core columns exist
    required = ["plate_id", "well_id", "cpd_id"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Metadata file missing column: {col}")

    # standardise wells
    df["well_id"] = df["well_id"].map(lambda w: standardise_well_string(well=w))

    # concentration fallback
    if "conc" not in df.columns:
        df["conc"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Merge / QC
# ---------------------------------------------------------------------------

def merge_image_with_metadata(
    *,
    df_image: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the per-image table with metadata.

    This function:
        * Detects plate_id columns in the Image table.
        * Splits plate IDs that contain suffixes (e.g. '24AP3815_15082024').
        * Logs a loud warning when suffixes are removed.
        * Standardises well IDs.
        * Extracts AR_count and Cell_count by column detection.
        * Merges metadata onto image rows.

    Parameters
    ----------
    df_image : pandas.DataFrame
        Per-image CellProfiler table.

    df_meta : pandas.DataFrame
        Metadata with plate_id, well_id, cpd_id and conc.

    Returns
    -------
    pandas.DataFrame
        Merged per-image table with AR_count, Cell_count, cpd_id, conc.
    """

    df = df_image.copy()
    meta = df_meta.copy()

    # ------------------------------------------------------------
    # Detect plate column in image table
    # ------------------------------------------------------------
    plate_candidates = ["plate_id", "Plate_Metadata", "PlateId"]
    plate_col = None

    for cand in plate_candidates:
        if cand in df.columns:
            plate_col = cand
            break

    if plate_col is None:
        raise ValueError("Image table missing any plate_id-like column.")

    # Normalise name
    df = df.rename(columns={plate_col: "plate_id"})

    # ------------------------------------------------------------
    # LOUD warning + strip suffix from plate_id if present
    # ------------------------------------------------------------
    if df["plate_id"].str.contains("_").any():
        before_vals = df["plate_id"].unique()[:10]

        df["plate_id"] = df["plate_id"].astype(str).str.split("_").str[0]

        after_vals = df["plate_id"].unique()[:10]

        LOGGER.warning(
            "Plate IDs in the Image table contained suffixes. They were normalised.\n"
            "Examples before: %s\n"
            "Examples after : %s",
            before_vals,
            after_vals,
        )

    # Ensure metadata plate_id is string
    meta["plate_id"] = meta["plate_id"].astype(str)

    # ------------------------------------------------------------
    # Detect well column
    # ------------------------------------------------------------
    well_candidates = ["well_id", "Well_Metadata", "WellId"]
    well_col = None

    for cand in well_candidates:
        if cand in df.columns:
            well_col = cand
            break

    if well_col is None:
        raise ValueError("Image table missing any well_id-like column.")

    df = df.rename(columns={well_col: "well_id"})
    df["well_id"] = df["well_id"].map(lambda w: standardise_well_string(well=w))

    # ------------------------------------------------------------
    # Detect AR_count column
    # ------------------------------------------------------------
    ar_candidates = ["AR_count", "Count_Acrosome"]
    ar_col = next((c for c in ar_candidates if c in df.columns), None)

    if ar_col is None:
        raise ValueError(f"No acrosome count column found. Tried: {ar_candidates}")

    df = df.rename(columns={ar_col: "AR_count"})

    # ------------------------------------------------------------
    # Detect Cell_count
    # ------------------------------------------------------------
    cell_candidates = ["Cell_count", "Count_SpermCells", "Count_Cells"]
    cell_col = next((c for c in cell_candidates if c in df.columns), None)

    if cell_col is None:
        raise ValueError(f"No total cell count column found. Tried: {cell_candidates}")

    df = df.rename(columns={cell_col: "Cell_count"})

    # ------------------------------------------------------------
    # Standardise metadata well ID
    # ------------------------------------------------------------
    meta["well_id"] = meta["well_id"].map(lambda w: standardise_well_string(well=w))

    # ------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------
    df_out = df.merge(
        meta,
        how="left",
        on=["plate_id", "well_id"],
        suffixes=("", "_meta"),
    )

    # Warn if metadata missing
    if df_out["cpd_id"].isna().any():
        missing = df_out["cpd_id"].isna().sum()
        LOGGER.warning(
            "%d image rows have missing cpd_id after metadata merge. "
            "Plate or well mismatch likely.",
            missing,
        )

    return df_out



def compute_well_level_counts(*, df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse image-level data to well-level AR counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged per-image table with AR_count and Cell_count.

    Returns
    -------
    pandas.DataFrame
        One row per well with AR count, Cell count, AR%.
    """
    req = ["plate_id", "well_id", "cpd_id", "AR_count", "Cell_count", "conc"]
    for col in req:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in merged dataset.")

    agg = (
        df.groupby(["plate_id", "well_id", "cpd_id", "conc"], as_index=False)
        .agg({"AR_count": "sum", "Cell_count": "sum"})
    )

    agg["AR_percent"] = (agg["AR_count"] / agg["Cell_count"]) * 100.0

    return agg


# ---------------------------------------------------------------------------
# Fisher tests
# ---------------------------------------------------------------------------

def run_fisher_single_rep(
    *,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run Fisher's exact test for single-replicate screens.

    Parameters
    ----------
    df : pandas.DataFrame
        Well-level table with cpd_id, AR_count, Cell_count.

    Returns
    -------
    pandas.DataFrame
        One row per compound: p-value, odds ratio.
    """
    if "DMSO" not in df["cpd_id"].unique():
        LOGGER.warning("No DMSO control found. Fisher test skipped.")
        return pd.DataFrame()

    df_dmso = df[df["cpd_id"] == "DMSO"].copy()
    dmso_AR = df_dmso["AR_count"].sum()
    dmso_non = df_dmso["Cell_count"].sum() - dmso_AR

    rows = []
    for cpd in sorted(df["cpd_id"].unique()):
        if cpd == "DMSO":
            continue

        sub = df[df["cpd_id"] == cpd]
        AR = sub["AR_count"].sum()
        non_AR = sub["Cell_count"].sum() - AR

        table = [[AR, non_AR], [dmso_AR, dmso_non]]
        try:
            OR, p = fisher_exact(table)
        except Exception:
            OR, p = np.nan, np.nan

        rows.append(
            {
                "cpd_id": cpd,
                "OR": OR,
                "p_value": p,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dose–response fitting
# ---------------------------------------------------------------------------

def run_dose_fitting(*, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit dose–response curves for compounds with multiple concentrations.

    Parameters
    ----------
    df : pandas.DataFrame
        Well-level table with cpd_id, conc, AR_percent.

    Returns
    -------
    pandas.DataFrame
        Curve parameters per compound.
    """
    out = []

    for cpd, sub in df.groupby("cpd_id"):
        # skip DMSO
        if cpd == "DMSO":
            continue

        sub = sub.dropna(subset=["conc"])
        if sub["conc"].nunique() < 3:
            continue

        x = sub["conc"].astype(float).values
        y = sub["AR_percent"].astype(float).values

        # initial guess
        p0 = [0.0, 100.0, np.median(x), 1.0]

        try:
            popt, _ = curve_fit(
                hill_equation,
                x,
                y,
                p0=p0,
                maxfev=20000,
            )
            bottom, top, ic50, hill = popt
        except Exception:
            bottom = top = ic50 = hill = np.nan

        out.append(
            {
                "cpd_id": cpd,
                "bottom": bottom,
                "top": top,
                "ic50": ic50,
                "hill": hill,
            }
        )

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_boxplot_single_rep(
    *,
    df: pd.DataFrame,
    outdir: Path,
) -> None:
    """
    Create a per-compound boxplot comparing compound AR% to DMSO.

    Parameters
    ----------
    df : pandas.DataFrame
        Well-level table with AR_percent and cpd_id.
    outdir : pathlib.Path
        Folder to write plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    compounds = [
        c for c in df["cpd_id"].unique().tolist() if c != "DMSO"
    ]

    data = []
    labels = []

    # DMSO reference
    dmso_vals = df[df["cpd_id"] == "DMSO"]["AR_percent"].values

    for cpd in compounds:
        vals = df[df["cpd_id"] == cpd]["AR_percent"].values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(cpd)

    # position offsets
    positions = np.arange(len(compounds))
    dmso_pos = positions - 0.25
    cpd_pos = positions + 0.25

    # DMSO jittered
    ax.scatter(
        np.repeat(dmso_pos, len(dmso_vals)),
        np.tile(dmso_vals, len(dmso_pos)),
        alpha=0.4,
    )

    # compound jittered
    for i, cpd in enumerate(compounds):
        vals = df[df["cpd_id"] == cpd]["AR_percent"].values
        ax.scatter(
            np.repeat(cpd_pos[i], len(vals)),
            vals,
            alpha=0.8,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(compounds, rotation=90)
    ax.set_ylabel("AR percent")
    ax.set_title("Per-compound AR% vs DMSO")

    outpath = outdir / "ar_boxplots_single_rep.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def write_html_report(
    *,
    df_well: pd.DataFrame,
    fisher_df: Optional[pd.DataFrame],
    dose_df: Optional[pd.DataFrame],
    outdir: Path,
) -> None:
    """
    Write a simple HTML summary.

    Parameters
    ----------
    df_well : pandas.DataFrame
        Well-level data.
    fisher_df : pandas.DataFrame or None
        Fisher test results.
    dose_df : pandas.DataFrame or None
        Dose–response parameter table.
    outdir : pathlib.Path
        Folder to write HTML file.
    """
    html_path = outdir / "summary.html"
    LOGGER.info("Writing HTML report to '%s'", html_path)

    with open(html_path, "w") as f:
        f.write("<html><body>\n")
        f.write("<h1>Acrosome analysis summary</h1>\n")

        f.write("<h2>Well-level table</h2>\n")
        f.write(df_well.head().to_html(index=False))

        if fisher_df is not None and not fisher_df.empty:
            f.write("<h2>Fisher exact results</h2>\n")
            f.write(fisher_df.to_html(index=False))

        if dose_df is not None and not dose_df.empty:
            f.write("<h2>Dose–response parameters</h2>\n")
            f.write(dose_df.to_html(index=False))

        f.write("</body></html>\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified acrosome screen or dose–response analysis"
    )

    parser.add_argument(
        "--cp_dir",
        type=str,
        required=True,
        help="Directory containing CellProfiler Image table.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Metadata TSV with plate_id, well_id, cpd_id, conc.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for all outputs.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="0 warning, 1 info, 2 debug.",
    )

    args = parser.parse_args()
    configure_logging(verbosity=args.verbosity)

    cp_dir = Path(args.cp_dir)
    metadata_path = Path(args.metadata)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load data
    df_img = load_image_table(cp_dir=cp_dir)
    df_obj = load_object_table_minimal(cp_dir=cp_dir)
    df_meta = load_metadata(metadata_path=metadata_path)

    # merge
    df_merge = merge_image_with_metadata(
        df_image=df_img,
        df_meta=df_meta,
    )

    # well-level counts
    df_well = compute_well_level_counts(df=df_merge)

    # save well table
    df_well_path = outdir / "well_level.tsv"
    df_well.to_csv(df_well_path, sep="\t", index=False)

    # decide mode
    has_multi_conc = df_well["conc"].nunique() > 1

    fisher_df = None
    dose_df = None

    if has_multi_conc:
        LOGGER.info("Detected dose–response mode.")
        dose_df = run_dose_fitting(df=df_well)
        if dose_df is not None:
            dose_df.to_csv(outdir / "dose_response_parameters.tsv",
                           sep="\t", index=False)
    else:
        LOGGER.info("Detected single-rep screen mode.")
        fisher_df = run_fisher_single_rep(df=df_well)
        if fisher_df is not None:
            fisher_df.to_csv(outdir / "fisher_single_rep.tsv",
                             sep="\t", index=False)

        plot_boxplot_single_rep(df=df_well, outdir=outdir)

    # HTML summary
    write_html_report(
        df_well=df_well,
        fisher_df=fisher_df,
        dose_df=dose_df,
        outdir=outdir,
    )

    LOGGER.info("Analysis complete.")


if __name__ == "__main__":
    main()
