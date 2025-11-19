#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse acrosome reaction (AR) rates versus DMSO controls on a single plate.

Overview
--------
This script expects a per-object/per-cell table from CellProfiler (or a similar
pipeline) with at least:

    - plate identifier column (e.g. "Plate_Metadata")
    - well identifier column (e.g. "Well_Metadata")
    - compound identifier column (e.g. "cpd_id")
    - compound type column (e.g. "cpd_type" with "DMSO" controls)
    - a binary AR flag column (e.g. "is_acrosome_reacted")

The analysis proceeds in several stages:

1. Load the per-cell table and coerce the AR flag to 0/1.
2. Aggregate per well to obtain:
       total_cells, total_AR_cells, AR_pct_well.
3. Perform basic QC:
       - drop wells with total_cells < min_cells_per_well
       - among DMSO wells, flag AR% outliers using an IQR rule.
4. For each non-DMSO compound on the plate:
       - pool AR/non-AR counts across its (QC-passed) wells
       - pool AR/non-AR counts for DMSO wells on the same plate
       - run a two-sided Fisher's exact test
       - compute odds ratio, 95% CI (approximate), and delta AR%
5. Adjust p-values across all tested compounds using Benjamini–Hochberg FDR.
6. Write a tab-separated summary table.
7. Generate basic visualisations:
       - plate heatmap of AR% per well (if plate layout is parsable)
       - volcano-style plot (log2(odds_ratio) vs -log10(p_adj))
       - simple barplot of delta AR% for top compounds
8. Build a minimal HTML report that links to the TSV and embeds the PNG figures.

Notes
-----
- This script is written for a single-plate analysis where DMSO controls are
  present on that plate.
- All comparisons are against DMSO on the same plate.
- Fisher's exact test is two-sided, so both increased and decreased AR
  probabilities are detectable and reported.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


LOGGER = logging.getLogger(__name__)


def configure_logging(*, verbosity: int) -> None:
    """
    Configure the root logger.

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
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    LOGGER.debug("Logging configured with level %s", logging.getLevelName(level))


def load_per_cell_table(
    *,
    input_path: Path,
    sep: str | None,
) -> pd.DataFrame:
    """
    Load the per-cell/per-object table.

    Parameters
    ----------
    input_path : Path
        Path to the input table (TSV or CSV).
    sep : str or None
        Delimiter to use. If None, a simple heuristic is applied based on
        the file suffix.

    Returns
    -------
    pandas.DataFrame
        Loaded data frame.
    """
    if sep is None:
        if input_path.suffix.lower() == ".csv":
            sep = ","
        else:
            sep = "\t"

    LOGGER.info("Loading per-cell table from '%s' with sep='%s'", input_path, sep)
    df = pd.read_csv(input_path, sep=sep)
    LOGGER.info("Loaded table with %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def coerce_ar_flag(
    *,
    df: pd.DataFrame,
    ar_flag_col: str,
) -> pd.DataFrame:
    """
    Coerce the AR flag column to binary 0/1.

    Any non-zero / True / 'True' / 'yes' / 'AR' value is treated as 1.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    ar_flag_col : str
        Name of the AR flag column.

    Returns
    -------
    pandas.DataFrame
        Data frame with a new 'AR_flag_bin' column.
    """
    if ar_flag_col not in df.columns:
        raise KeyError(f"AR flag column '{ar_flag_col}' not found in table.")

    LOGGER.info("Coercing AR flag column '%s' to binary", ar_flag_col)

    col = df[ar_flag_col]

    def _to_bin(x: object) -> int:
        if pd.isna(x):
            return 0
        if isinstance(x, (int, float, np.integer, np.floating)):
            return int(x != 0)
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "y", "ar", "reacted"}:
            return 1
        return 0

    df = df.copy()
    df["AR_flag_bin"] = col.map(_to_bin).astype(int)
    LOGGER.debug(
        "AR_flag_bin value counts:\n%s",
        df["AR_flag_bin"].value_counts(dropna=False),
    )
    return df


def aggregate_per_well(
    *,
    df: pd.DataFrame,
    plate_col: str,
    well_col: str,
    cpd_id_col: str,
    cpd_type_col: str,
) -> pd.DataFrame:
    """
    Aggregate per-object table into per-well AR counts and percentages.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-cell data frame containing 'AR_flag_bin'.
    plate_col : str
        Name of the plate identifier column.
    well_col : str
        Name of the well identifier column.
    cpd_id_col : str
        Column containing the compound identifier.
    cpd_type_col : str
        Column containing the compound type (e.g. DMSO, positive control, test).

    Returns
    -------
    pandas.DataFrame
        Per-well summary with:
        - Plate
        - Well
        - cpd_id
        - cpd_type
        - total_cells
        - total_AR_cells
        - AR_pct_well
    """
    required_cols = {plate_col, well_col, cpd_id_col, cpd_type_col, "AR_flag_bin"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for aggregation: {sorted(missing)}")

    LOGGER.info("Aggregating per well using plate=%s, well=%s", plate_col, well_col)

    group_cols = [plate_col, well_col, cpd_id_col, cpd_type_col]
    agg = (
        df.groupby(group_cols, dropna=False)["AR_flag_bin"]
        .agg(
            total_cells="count",
            total_AR_cells="sum",
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
    cpd_type_col: str,
    control_type: str,
    min_cells_per_well: int,
    control_iqr_multiplier: float,
) -> pd.DataFrame:
    """
    Flag wells that fail basic QC criteria.

    QC rules:
    ---------
    1. Drop any well with total_cells < min_cells_per_well.
    2. Among control (e.g. DMSO) wells, flag AR_pct_well outliers using
       an IQR rule:
           outlier if AR_pct < Q1 - k*IQR or AR_pct > Q3 + k*IQR,
       where k = control_iqr_multiplier.

    Parameters
    ----------
    df_well : pandas.DataFrame
        Per-well summary table.
    cpd_type_col : str
        Column containing compound type (e.g. DMSO, positive control, test).
    control_type : str
        Value in cpd_type_col that denotes DMSO controls.
    min_cells_per_well : int
        Minimum number of cells required for a well to be considered reliable.
    control_iqr_multiplier : float
        Multiplier for IQR to define AR% outliers among control wells.

    Returns
    -------
    pandas.DataFrame
        Copy of df_well with additional columns:
        - is_control : bool
        - qc_reason : comma-separated reasons (empty string if none)
        - qc_keep : bool (True if well passes QC)
    """
    df = df_well.copy()
    df["is_control"] = df[cpd_type_col].astype(str) == str(control_type)
    df["qc_reason"] = ""

    LOGGER.info(
        "Flagging wells for QC with min_cells_per_well=%d, control_iqr_multiplier=%.2f",
        min_cells_per_well,
        control_iqr_multiplier,
    )

    # Rule 1: low cell count
    low_mask = df["total_cells"] < min_cells_per_well
    df.loc[low_mask, "qc_reason"] = df.loc[low_mask, "qc_reason"].astype(str) + "LOW_CELL_COUNT"
    LOGGER.info("Wells failing low cell count: %d", low_mask.sum())

    # Rule 2: AR% outliers among controls only
    control_df = df[df["is_control"] & ~low_mask]
    if control_df.empty:
        LOGGER.warning("No control wells available for AR outlier QC.")
    else:
        q1 = control_df["AR_pct_well"].quantile(0.25)
        q3 = control_df["AR_pct_well"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - control_iqr_multiplier * iqr
        upper = q3 + control_iqr_multiplier * iqr
        LOGGER.info(
            "Control AR%% IQR-based thresholds: Q1=%.3f, Q3=%.3f, IQR=%.3f, "
            "lower=%.3f, upper=%.3f",
            q1,
            q3,
            iqr,
            lower,
            upper,
        )

        outlier_mask = df["is_control"] & ~low_mask & (
            (df["AR_pct_well"] < lower) | (df["AR_pct_well"] > upper)
        )
        df.loc[outlier_mask, "qc_reason"] = (
            df.loc[outlier_mask, "qc_reason"].replace("", "CONTROL_AR_OUTLIER") + 
            df.loc[outlier_mask, "qc_reason"].where(
                df.loc[outlier_mask, "qc_reason"] == "",
                "," + "CONTROL_AR_OUTLIER",
            )
        )
        LOGGER.info(
            "Control wells flagged as AR%% outliers: %d",
            outlier_mask.sum(),
        )

    # Define qc_keep: keep wells with no qc_reason
    df["qc_keep"] = df["qc_reason"].astype(str) == ""
    LOGGER.info(
        "QC summary: %d wells kept, %d wells dropped",
        df["qc_keep"].sum(),
        (~df["qc_keep"]).sum(),
    )
    return df


def benjamini_hochberg(
    *,
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Apply Benjamini–Hochberg FDR correction to a set of p-values.

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


def run_fisher_per_compound(
    *,
    df_well_qc: pd.DataFrame,
    plate_col: str,
    cpd_id_col: str,
    cpd_type_col: str,
    control_type: str,
) -> pd.DataFrame:
    """
    Run Fisher's exact test for each compound versus DMSO on a single plate.

    Parameters
    ----------
    df_well_qc : pandas.DataFrame
        Per-well table with QC flags applied and 'qc_keep' column.
    plate_col : str
        Plate identifier column.
    cpd_id_col : str
        Compound identifier column.
    cpd_type_col : str
        Compound type column (e.g. DMSO, positive control, test).
    control_type : str
        Value in cpd_type_col that denotes DMSO controls.

    Returns
    -------
    pandas.DataFrame
        Per-compound table with:
        - plate
        - cpd_id
        - cpd_type
        - n_wells
        - total_cells_compound
        - total_AR_compound
        - AR_pct_compound
        - total_cells_control
        - total_AR_control
        - AR_pct_control
        - delta_AR_pct
        - odds_ratio
        - p_value
        - q_value (FDR-adjusted)
    """
    df = df_well_qc[df_well_qc["qc_keep"]].copy()
    if df.empty:
        raise ValueError("No wells remain after QC; cannot run Fisher tests.")

    plates = df[plate_col].unique()
    if len(plates) != 1:
        LOGGER.warning(
            "Multiple plates detected (%s); proceeding but treating all as one pool.",
            plates,
        )

    control_mask = df[cpd_type_col].astype(str) == str(control_type)
    control_df = df[control_mask]
    if control_df.empty:
        raise ValueError(
            f"No control wells found for control_type='{control_type}'. "
            "Cannot perform comparisons.",
        )

    LOGGER.info(
        "Fisher tests will compare each compound to %d control wells.",
        control_df.shape[0],
    )

    # Pooled control counts
    total_cells_control = int(control_df["total_cells"].sum())
    total_AR_control = int(control_df["total_AR_cells"].sum())
    AR_pct_control = (total_AR_control / total_cells_control) * 100.0

    LOGGER.info(
        "Pooled DMSO control counts: AR=%d, non-AR=%d, AR%%=%.3f",
        total_AR_control,
        total_cells_control - total_AR_control,
        AR_pct_control,
    )

    # Group by compound (excluding control_type)
    non_control_df = df[~control_mask].copy()
    compounds = (
        non_control_df[[cpd_id_col, cpd_type_col]]
        .drop_duplicates()
        .sort_values([cpd_type_col, cpd_id_col])
    )

    results: List[Dict[str, object]] = []
    p_values: List[float] = []

    for _, row in compounds.iterrows():
        cpd_id = row[cpd_id_col]
        cpd_type = row[cpd_type_col]

        c_df = non_control_df[non_control_df[cpd_id_col] == cpd_id]
        if c_df.empty:
            continue

        total_cells_compound = int(c_df["total_cells"].sum())
        total_AR_compound = int(c_df["total_AR_cells"].sum())
        AR_pct_compound = (total_AR_compound / total_cells_compound) * 100.0
        delta_AR_pct = AR_pct_compound - AR_pct_control

        table = np.array(
            [
                [total_AR_compound, total_cells_compound - total_AR_compound],
                [total_AR_control, total_cells_control - total_AR_control],
            ]
        )

        # Two-sided Fisher's exact test
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        p_values.append(p_value)

        results.append(
            {
                "plate": str(df[plate_col].iloc[0]),
                "cpd_id": cpd_id,
                "cpd_type": cpd_type,
                "n_wells": int(c_df.shape[0]),
                "total_cells_compound": total_cells_compound,
                "total_AR_compound": total_AR_compound,
                "AR_pct_compound": AR_pct_compound,
                "total_cells_control": total_cells_control,
                "total_AR_control": total_AR_control,
                "AR_pct_control": AR_pct_control,
                "delta_AR_pct": delta_AR_pct,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No non-control compounds found for Fisher testing.")

    q_values = benjamini_hochberg(p_values=np.asarray(p_values, dtype=float))
    results_df["q_value"] = q_values

    LOGGER.info(
        "Fisher testing completed for %d compounds.", results_df.shape[0]
    )
    return results_df


def plot_plate_heatmap(
    *,
    df_well_qc: pd.DataFrame,
    plate_col: str,
    well_col: str,
    value_col: str,
    output_path: Path,
) -> None:
    """
    Plot a plate-layout heatmap of AR% (or another value) per well.

    This function assumes well identifiers follow a standard format like 'A01',
    'B12', etc. If parsing fails, a warning is logged and the plot is skipped.

    Parameters
    ----------
    df_well_qc : pandas.DataFrame
        Per-well table (QC-passed wells preferred).
    plate_col : str
        Plate identifier column (used only for logging).
    well_col : str
        Well identifier column (e.g. 'Well_Metadata').
    value_col : str
        Column with the value to plot (e.g. 'AR_pct_well').
    output_path : Path
        Where to save the PNG file.
    """
    df = df_well_qc.copy()
    plate_id = str(df[plate_col].iloc[0]) if not df.empty else "NA"

    LOGGER.info(
        "Attempting plate heatmap for plate '%s' using column '%s'",
        plate_id,
        value_col,
    )

    # Parse row (letter) and column (integer) from well IDs like 'A01'
    def _parse_well(w: str) -> Tuple[str, int] | Tuple[None, None]:
        if not isinstance(w, str) or len(w) < 2:
            return None, None
        row = w[0].upper()
        try:
            col = int(w[1:])
        except ValueError:
            return None, None
        return row, col

    df["well_row"], df["well_col"] = zip(*df[well_col].astype(str).map(_parse_well))

    if df["well_row"].isna().any() or df["well_col"].isna().any():
        LOGGER.warning(
            "Could not parse all well IDs into rows/columns; skipping plate heatmap."
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
        heat[i, j] = r[value_col]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heat, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"Plate {plate_id} – {value_col}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved plate heatmap to '%s'", output_path)


def plot_volcano(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot a simple volcano-style plot for Fisher results.

    x-axis: log2(odds_ratio)
    y-axis: -log10(q_value)

    Parameters
    ----------
    results_df : pandas.DataFrame
        Per-compound Fisher results.
    output_path : Path
        Path to save the PNG figure.
    """
    df = results_df.copy()

    # Handle odds_ratio <= 0 (rare, but may happen if counts are zero)
    df["log2_or"] = df["odds_ratio"].replace(0, np.nan)
    df["log2_or"] = np.log2(df["log2_or"])
    df["neg_log10_q"] = -np.log10(df["q_value"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["log2_or"], df["neg_log10_q"], s=20, alpha=0.7)

    ax.axvline(0.0, colour="grey", linestyle="--", linewidth=1.0)
    ax.set_xlabel("log2(odds ratio, compound vs DMSO)")
    ax.set_ylabel("-log10(FDR q-value)")
    ax.set_title("Acrosome reaction – Fisher test per compound")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved volcano plot to '%s'", output_path)


def plot_top_delta_barplot(
    *,
    results_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """
    Plot a barplot of delta AR% for the top-N compounds by smallest q-value.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Per-compound Fisher results.
    output_path : Path
        Path to save the PNG figure.
    top_n : int, optional
        Number of top compounds to display, by smallest q-value.
    """
    df = results_df.sort_values("q_value", ascending=True).head(top_n).copy()
    df = df.sort_values("delta_AR_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(df))))
    ax.barh(df["cpd_id"].astype(str), df["delta_AR_pct"])

    ax.set_xlabel("Delta AR% (compound − DMSO)")
    ax.set_ylabel("Compound ID")
    ax.set_title(f"Top {len(df)} compounds by FDR (delta AR%)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Saved top delta AR%% barplot to '%s'", output_path)


def write_html_report(
    *,
    output_dir: Path,
    summary_tsv: Path,
    plate_heatmap_png: Path | None,
    volcano_png: Path | None,
    barplot_png: Path | None,
) -> Path:
    """
    Write a simple HTML report linking to the TSV and embedding PNG figures.

    Parameters
    ----------
    output_dir : Path
        Output directory where the HTML file will be written.
    summary_tsv : Path
        Path to the main summary TSV file.
    plate_heatmap_png : Path or None
        Path to the plate heatmap PNG, if generated.
    volcano_png : Path or None
        Path to the volcano PNG, if generated.
    barplot_png : Path or None
        Path to the barplot PNG, if generated.

    Returns
    -------
    Path
        Path to the HTML file.
    """
    html_path = output_dir / "acrosome_vs_dmso_report.html"
    LOGGER.info("Writing HTML report to '%s'", html_path)

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<title>Acrosome reaction vs DMSO – report</title>")
    parts.append(
        "<style>"
        "body { font-family: Arial, sans-serif; margin: 1.5em; }"
        "h1, h2 { colour: #333333; }"
        "img { max-width: 100%; height: auto; margin-bottom: 1em; }"
        "code { background-colour: #f0f0f0; padding: 0.1em 0.3em; }"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<h1>Acrosome reaction vs DMSO – single-plate analysis</h1>")

    parts.append("<p>Main summary table (tab-separated): "
                 f"<a href='{summary_tsv.name}'>{summary_tsv.name}</a></p>")

    if plate_heatmap_png is not None and plate_heatmap_png.exists():
        parts.append("<h2>Plate layout – AR% per well (QC-kept wells)</h2>")
        parts.append(f"<img src='{plate_heatmap_png.name}' alt='Plate heatmap'>")

    if volcano_png is not None and volcano_png.exists():
        parts.append("<h2>Fisher test results – volcano plot</h2>")
        parts.append(f"<img src='{volcano_png.name}' alt='Volcano plot'>")

    if barplot_png is not None and barplot_png.exists():
        parts.append("<h2>Top compounds by FDR (delta AR%)</h2>")
        parts.append(f"<img src='{barplot_png.name}' alt='Delta AR% barplot'>")

    parts.append("<p>All results are relative to pooled DMSO controls on this plate. "
                 "P-values are two-sided Fisher's exact tests, with FDR control "
                 "using the Benjamini–Hochberg procedure.</p>")

    parts.append("</body></html>")

    html_content = "\n".join(parts)
    html_path.write_text(html_content, encoding="utf-8")

    return html_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyse acrosome reaction vs DMSO on a single plate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="Path to per-cell/per-object table (TSV or CSV).",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory to write outputs (TSV, PNG, HTML).",
    )
    parser.add_argument(
        "--sep",
        dest="sep",
        default=None,
        help="Explicit input delimiter. If not set, inferred from suffix.",
    )
    parser.add_argument(
        "--plate_col",
        dest="plate_col",
        default="Plate_Metadata",
        help="Plate identifier column.",
    )
    parser.add_argument(
        "--well_col",
        dest="well_col",
        default="Well_Metadata",
        help="Well identifier column.",
    )
    parser.add_argument(
        "--cpd_id_col",
        dest="cpd_id_col",
        default="cpd_id",
        help="Compound identifier column.",
    )
    parser.add_argument(
        "--cpd_type_col",
        dest="cpd_type_col",
        default="cpd_type",
        help="Compound type column (e.g. DMSO, positive, test).",
    )
    parser.add_argument(
        "--control_type",
        dest="control_type",
        default="DMSO",
        help="Value in cpd_type_col that denotes control wells.",
    )
    parser.add_argument(
        "--ar_flag_col",
        dest="ar_flag_col",
        default="is_acrosome_reacted",
        help="AR flag column to coerce to binary (0/1).",
    )
    parser.add_argument(
        "--min_cells_per_well",
        dest="min_cells_per_well",
        type=int,
        default=50,
        help="Minimum cell count per well for inclusion.",
    )
    parser.add_argument(
        "--control_iqr_multiplier",
        dest="control_iqr_multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier for defining AR%% outliers among controls.",
    )
    parser.add_argument(
        "--verbosity",
        dest="verbosity",
        type=int,
        default=1,
        help="Verbosity level: 0=warnings, 1=info, 2=debug.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for command-line execution.
    """
    args = parse_args()
    configure_logging(verbosity=args.verbosity)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_cells = load_per_cell_table(
        input_path=input_path,
        sep=args.sep,
    )
    df_cells = coerce_ar_flag(
        df=df_cells,
        ar_flag_col=args.ar_flag_col,
    )

    df_well = aggregate_per_well(
        df=df_cells,
        plate_col=args.plate_col,
        well_col=args.well_col,
        cpd_id_col=args.cpd_id_col,
        cpd_type_col=args.cpd_type_col,
    )

    df_well_qc = flag_wells_for_qc(
        df_well=df_well,
        cpd_type_col=args.cpd_type_col,
        control_type=args.control_type,
        min_cells_per_well=args.min_cells_per_well,
        control_iqr_multiplier=args.control_iqr_multiplier,
    )

    fisher_df = run_fisher_per_compound(
        df_well_qc=df_well_qc,
        plate_col=args.plate_col,
        cpd_id_col=args.cpd_id_col,
        cpd_type_col=args.cpd_type_col,
        control_type=args.control_type,
    )

    # Write main summary TSV
    summary_tsv = output_dir / "acrosome_vs_dmso_fisher_results.tsv"
    fisher_df.to_csv(summary_tsv, sep="\t", index=False)
    LOGGER.info("Wrote Fisher results to '%s'", summary_tsv)

    # Plots
    plate_heatmap_png = output_dir / "plate_AR_pct_heatmap.png"
    plot_plate_heatmap(
        df_well_qc=df_well_qc[df_well_qc["qc_keep"]],
        plate_col=args.plate_col,
        well_col=args.well_col,
        value_col="AR_pct_well",
        output_path=plate_heatmap_png,
    )

    volcano_png = output_dir / "fisher_volcano.png"
    plot_volcano(
        results_df=fisher_df,
        output_path=volcano_png,
    )

    barplot_png = output_dir / "top_delta_AR_pct_barplot.png"
    plot_top_delta_barplot(
        results_df=fisher_df,
        output_path=barplot_png,
        top_n=30,
    )

    # HTML report
    html_report = write_html_report(
        output_dir=output_dir,
        summary_tsv=summary_tsv,
        plate_heatmap_png=plate_heatmap_png,
        volcano_png=volcano_png,
        barplot_png=barplot_png,
    )
    LOGGER.info("HTML report written to '%s'", html_report)


if __name__ == "__main__":
    main()
