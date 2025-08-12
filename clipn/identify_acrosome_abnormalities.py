#!/usr/bin/env python3
# coding: utf-8
"""
Batch comparison: All compounds vs. DMSO — Acrosome Features
===========================================================

For each compound, compare its wells to DMSO wells using a non-parametric test
(Mann–Whitney U by default, or Kolmogorov–Smirnov) plus Earth Mover’s Distance.
Outputs per-compound:
- Top N features (TSV + XLSX)
- All significant features by FDR (TSV + XLSX)
- Top acrosome-only features (TSV + XLSX)
- Significant acrosome-only features (TSV + XLSX)
- Acrosome group summary (TSV + XLSX), plus significant subgroup (by min pvalue)

Input
-----
--ungrouped_list : CSV/TSV containing a column 'path' listing well-level tables.

The well-level tables should contain numeric feature columns and at least:
- a compound id column (default: 'cpd_id')
- a compound type/label column (default: 'cpd_type') or specify --dmso_col

DMSO detection:
- Prefer an explicit column with --dmso_col (defaults to cpd_type).
- If that column is missing, fallback to scanning text-like columns for a
  case-insensitive substring match to --dmso_label across each row.

Usage
-----
python batch_acrosome_vs_dmso.py \
  --ungrouped_list dataset_paths.txt \
  --output_dir acrosome_vs_DMSO \
  --dmso_label DMSO \
  --dmso_col cpd_type \
  --test mw \
  --top_features 10 \
  --fdr_alpha 0.05
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from statsmodels.stats.multitest import multipletests


# -------------------
# Logging & utilities
# -------------------

def setup_logger(log_file: str) -> logging.Logger:
    """
    Configure a logger for both file and console output.

    Parameters
    ----------
    log_file : str
        Path to the log file.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger("acrosome_vs_dmso_logger")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    # Reset handlers if re-run in the same interpreter
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def detect_delimiter(path: str) -> str:
    """
    Detect delimiter in a small text file; prefer tab if both appear.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        Detected delimiter: '\\t' or ','.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(4096)
    has_tab = "\t" in sample
    has_comma = "," in sample
    if has_tab and has_comma:
        return "\t"
    if has_tab:
        return "\t"
    if has_comma:
        return ","
    return "\t"


def read_table_auto(path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Read CSV/TSV with auto delimiter detection.

    Parameters
    ----------
    path : str
        Path to table.
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    sep = detect_delimiter(path)
    logger.debug("Reading table %s (sep=%r)", path, sep)
    try:
        return pd.read_csv(path, sep=sep)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        raise


def ensure_columns(df: pd.DataFrame, required_cols: List[str], fill_value="missing") -> pd.DataFrame:
    """
    Ensure required columns exist; fill with a constant if missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    required_cols : list[str]
        Columns to ensure.
    fill_value : Any
        Value to fill for missing columns.

    Returns
    -------
    pd.DataFrame
        Updated table.
    """
    for col in required_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


def safe_write_tsv(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    """
    Robust TSV writer with clear logging.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : str
        Output path.
    logger : logging.Logger
        Logger.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(map(str, t)) for t in out.columns.to_list()]
    else:
        out.columns = out.columns.map(str)
    out.to_csv(path, sep="\t", index=False)
    logger.info("Wrote %s rows × %s cols -> %s", out.shape[0], out.shape[1], path)


def safe_write_xlsx(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    """
    Robust XLSX writer with clear logging.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : str
        Output path.
    logger : logging.Logger
        Logger.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(map(str, t)) for t in out.columns.to_list()]
    else:
        out.columns = out.columns.map(str)
    out.to_excel(path, index=False)
    logger.info("Wrote %s rows × %s cols -> %s", out.shape[0], out.shape[1], path)


def reorder_and_write(
    df: pd.DataFrame,
    filename: str,
    col_order: List[str],
    logger: logging.Logger,
    filetype: str = "tsv",
) -> None:
    """
    Ensure standard columns (in order), append any extras, and write.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    filename : str
        Output file path.
    col_order : list[str]
        Desired standard column order.
    logger : logging.Logger
        Logger.
    filetype : str
        'tsv' or 'excel'.
    """
    df_out = df.copy()
    for col in col_order:
        if col not in df_out.columns:
            df_out[col] = np.nan

    cols_in_order = [c for c in col_order if c in df_out.columns]
    remaining = [c for c in df_out.columns if c not in cols_in_order]
    df_out = df_out[cols_in_order + remaining]

    if filetype == "tsv":
        safe_write_tsv(df_out, filename, logger)
    elif filetype == "excel":
        safe_write_xlsx(df_out, filename, logger)
    else:
        raise ValueError("filetype must be 'tsv' or 'excel'")

    logger.debug("Columns in %s: %s", filename, df_out.columns.tolist())


# ----------------
# Data preparation
# ----------------

def load_ungrouped_files(
    list_file: str,
    logger: logging.Logger,
    required_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Load and vertically concatenate all well-level tables listed in a file.

    Parameters
    ----------
    list_file : str
        CSV/TSV with a column 'path' of file paths.
    logger : logging.Logger
        Logger.
    required_cols : list[str] | None
        Columns to ensure exist in each table.

    Returns
    -------
    pd.DataFrame
        Combined well-level table (harmonised by column intersection).
    """
    if required_cols is None:
        required_cols = ["cpd_id", "cpd_type"]

    logger.info("Reading file-of-files: %s", list_file)
    df_list = read_table_auto(list_file, logger)
    if "path" not in df_list.columns:
        logger.error("Input list must have a column named 'path'.")
        raise ValueError("Missing 'path' column in file-of-files.")

    dfs = []
    for path in df_list["path"]:
        logger.info("Reading well-level data: %s", path)
        tmp = read_table_auto(path, logger)
        tmp = ensure_columns(tmp, required_cols, fill_value="missing")
        dfs.append(tmp)

    if not dfs:
        raise ValueError("No input files were read.")

    # Harmonise by intersection of columns
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    if not common_cols:
        logger.error("No common columns across input files.")
        raise ValueError("No common columns in well-level files.")
    logger.info("%d columns are common to all files.", len(common_cols))

    dfs = [d[list(common_cols)] for d in dfs]
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Combined well-level data shape: %s", combined.shape)
    return combined


def select_feature_columns(
    df: pd.DataFrame,
    id_cols: List[str],
    logger: logging.Logger,
    drop_all_nan: bool = True,
    drop_constant: bool = True,
) -> List[str]:
    """
    Select numeric feature columns, excluding ID/label metadata, and optionally
    drop all-NaN or constant columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    id_cols : list[str]
        Non-feature columns to exclude.
    logger : logging.Logger
        Logger.
    drop_all_nan : bool
        Drop columns that are all NaN.
    drop_constant : bool
        Drop columns with zero variance.

    Returns
    -------
    list[str]
        Selected feature columns.
    """
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in id_cols
    ]

    if drop_all_nan:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if not df[c].isna().all()]
        logger.info("Dropped %d all-NaN feature columns.", before - len(feature_cols))

    if drop_constant and feature_cols:
        before = len(feature_cols)
        variances = df[feature_cols].var(numeric_only=True)
        feature_cols = [c for c in feature_cols if variances.loc[c] > 0.0 or pd.isna(variances.loc[c]) is False]
        logger.info("Dropped %d constant-variance feature columns.", before - len(feature_cols))

    logger.info("Selected %d feature columns.", len(feature_cols))
    return feature_cols


def infer_acrosome_features(feature_cols: List[str], logger: logging.Logger) -> List[str]:
    """
    Identify features containing 'acrosome' (case-insensitive).

    Parameters
    ----------
    feature_cols : list[str]
        Feature column names.
    logger : logging.Logger
        Logger.

    Returns
    -------
    list[str]
        Feature names containing 'acrosome'.
    """
    acrosome_feats = [f for f in feature_cols if "acrosome" in f.lower()]
    logger.info("Found %d acrosome features.", len(acrosome_feats))
    if acrosome_feats:
        logger.debug("Example acrosome features: %s", acrosome_feats[:10])
    return acrosome_feats


def infer_compartments(feature_cols: List[str], logger: logging.Logger) -> Dict[str, List[str]]:
    """
    Infer compartments from feature names using the last underscore token.

    Parameters
    ----------
    feature_cols : list[str]
        Feature names.
    logger : logging.Logger
        Logger.

    Returns
    -------
    dict[str, list[str]]
        Mapping compartment -> feature list.
    """
    group_map: Dict[str, List[str]] = defaultdict(list)
    for feat in feature_cols:
        tokens = feat.split("_")
        compartment = tokens[-1].lower() if len(tokens) > 1 else "unknown"
        group_map[compartment].append(feat)
    logger.info("Inferred %d compartments.", len(group_map))
    return dict(group_map)


def dmso_mask_from_column(
    df: pd.DataFrame,
    label: str,
    col: str,
    logger: logging.Logger,
) -> Tuple[pd.Series, str]:
    """
    Build a boolean mask for DMSO wells based on an explicit column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label : str
        DMSO label (case-insensitive).
    col : str
        Column name to check.
    logger : logging.Logger
        Logger.

    Returns
    -------
    (mask, reason) : (pd.Series, str)
        Boolean mask and reason string for logging.
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index), f"column '{col}' missing"

    col_vals = df[col].astype(str).str.upper().fillna("")
    mask = col_vals.eq(label.upper())
    return mask, f"matched {mask.sum()} rows in column '{col}'"


def dmso_mask_fallback_scan(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Fallback: mark a row as DMSO if ANY text-like column contains the label.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label : str
        DMSO label (case-insensitive).

    Returns
    -------
    pd.Series
        Boolean mask.
    """
    upper_label = label.upper()
    text_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(text_cols) == 0:
        return pd.Series(False, index=df.index)
    # Fast partial: build a per-column mask and OR-reduce
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        mask |= df[col].astype(str).str.upper().str.contains(upper_label, na=False)
    return mask


# ---------------
# Stats utilities
# ---------------

def compare_distributions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_cols: List[str],
    logger: logging.Logger,
    test: str = "mw",
) -> pd.DataFrame:
    """
    Compare distributions of each feature between two well sets using a
    non-parametric test and EMD (Wasserstein distance).

    Parameters
    ----------
    df1 : pd.DataFrame
        Query group (e.g., a compound's wells).
    df2 : pd.DataFrame
        Comparator group (e.g., DMSO wells).
    feature_cols : list[str]
        Feature columns to test.
    logger : logging.Logger
        Logger.
    test : str
        'mw' for Mann–Whitney U (default) or 'ks' for Kolmogorov–Smirnov.

    Returns
    -------
    pd.DataFrame
        Columns: feature, stat, raw_pvalue, abs_median_diff, emd, med_query, med_comp
    """
    rows = []
    use_mw = (test.lower() == "mw")

    for feat in feature_cols:
        x1 = df1[feat].dropna()
        x2 = df2[feat].dropna()

        if len(x1) < 2 or len(x2) < 2:
            stat, p, emd = np.nan, np.nan, np.nan
        else:
            try:
                if use_mw:
                    stat, p = mannwhitneyu(x1, x2, alternative="two-sided")
                else:
                    stat, p = ks_2samp(x1, x2)
            except Exception as exc:
                logger.debug("Test failed for %s: %s", feat, exc)
                stat, p = np.nan, np.nan

            try:
                emd = wasserstein_distance(x1, x2)
            except Exception:
                emd = np.nan

        med1 = np.median(x1) if len(x1) else np.nan
        med2 = np.median(x2) if len(x2) else np.nan
        rows.append({
            "feature": feat,
            "stat": stat,
            "raw_pvalue": p,
            "abs_median_diff": np.abs(med1 - med2),
            "emd": emd,
            "med_query": med1,
            "med_comp": med2,
        })

    return pd.DataFrame(rows)


def add_fdr(
    df_stats: pd.DataFrame,
    alpha: float,
    logger: logging.Logger,
    p_col: str = "raw_pvalue",
) -> pd.DataFrame:
    """
    Add BH-FDR to a stats table safely (handles NaNs).

    Parameters
    ----------
    df_stats : pd.DataFrame
        Stats table with a raw p-value column.
    alpha : float
        FDR threshold.
    logger : logging.Logger
        Logger.
    p_col : str
        Name of raw p-value column.

    Returns
    -------
    pd.DataFrame
        With new columns: pvalue_bh, reject_bh (bool).
    """
    out = df_stats.copy()
    pvals = out[p_col].to_numpy()
    valid = ~np.isnan(pvals)

    p_bh = np.full_like(pvals, fill_value=np.nan, dtype=float)
    reject = np.zeros_like(valid, dtype=bool)

    if valid.sum() > 0:
        rej, p_corr, _, _ = multipletests(pvals[valid], method="fdr_bh", alpha=alpha)
        p_bh[valid] = p_corr
        reject[valid] = rej
    else:
        logger.warning("No valid p-values for FDR correction.")

    out["pvalue_bh"] = p_bh
    out["reject_bh"] = reject
    return out


def group_feature_stats(
    feat_stats: pd.DataFrame,
    group_map: Dict[str, List[str]],
    fdr_alpha: float = 0.05,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Summarise per-feature stats by group, including size and FDR counts.

    Parameters
    ----------
    feat_stats : pd.DataFrame
        Per-feature stats table (should include 'pvalue_bh').
    group_map : dict[str, list[str]]
        group -> feature list mapping.
    fdr_alpha : float
        FDR threshold.
    logger : logging.Logger | None
        Logger.

    Returns
    -------
    pd.DataFrame
        Columns: group, mean_abs_median_diff, min_raw_pvalue, mean_emd,
                 min_pvalue_bh, n_features_in_group, n_features_sig_fdr
    """
    grouped = []
    for group, feats in group_map.items():
        group_df = feat_stats[feat_stats["feature"].isin(feats)]
        if group_df.empty:
            continue
        grouped.append({
            "group": group,
            "mean_abs_median_diff": group_df["abs_median_diff"].mean(),
            "min_raw_pvalue": group_df["raw_pvalue"].min(),
            "mean_emd": group_df["emd"].mean() if "emd" in group_df.columns else np.nan,
            "min_pvalue_bh": group_df["pvalue_bh"].min() if "pvalue_bh" in group_df.columns else np.nan,
            "n_features_in_group": len(group_df),
            "n_features_sig_fdr": int((group_df.get("pvalue_bh", pd.Series(dtype=float)) <= fdr_alpha).sum()),
        })

    out = pd.DataFrame(grouped)
    if logger:
        logger.debug("Summarised %d groups.", out.shape[0])
    return out


# ----
# Main
# ----

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Batch compare all compounds to DMSO for acrosome features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ungrouped_list", required=True, help="CSV/TSV with column 'path' of well-level files.")
    parser.add_argument("--cpd_id_col", default="cpd_id", help="Compound ID column.")
    parser.add_argument("--cpd_type_col", default="cpd_type", help="Compound type column (metadata).")
    parser.add_argument("--dmso_label", default="DMSO", help="Label used for DMSO rows.")
    parser.add_argument("--dmso_col", default="cpd_type", help="Column to check for DMSO (fallback: scan all text columns).")
    parser.add_argument("--output_dir", default="acrosome_vs_DMSO", help="Output folder.")
    parser.add_argument("--top_features", type=int, default=10, help="Number of top features to report.")
    parser.add_argument("--fdr_alpha", type=float, default=0.05, help="FDR threshold.")
    parser.add_argument("--test", choices=["mw", "ks"], default="mw", help="Non-parametric test to use.")
    parser.add_argument("--log_file", default="acrosome_vs_dmso.log", help="Log file name.")
    args = parser.parse_args()

    # Output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    sig_dir = os.path.join(args.output_dir, "significant")
    os.makedirs(sig_dir, exist_ok=True)

    logger = setup_logger(os.path.join(args.output_dir, args.log_file))
    logger.info("Starting batch analysis: all compounds vs. DMSO.")

    # Load & select features
    df = load_ungrouped_files(args.ungrouped_list, logger)
    id_cols = [args.cpd_id_col, args.cpd_type_col]
    feature_cols = select_feature_columns(df, id_cols=id_cols, logger=logger)

    # Acrosome subsets / groups
    acrosome_feats = infer_acrosome_features(feature_cols, logger)
    group_map = infer_compartments(feature_cols, logger)

    # DMSO mask
    dmso_mask, reason = dmso_mask_from_column(df, args.dmso_label, args.dmso_col, logger)
    if not dmso_mask.any():
        logger.warning(
            "No DMSO rows found using column=%r (%s). Falling back to scanning text columns.",
            args.dmso_col, reason,
        )
        dmso_mask = dmso_mask_fallback_scan(df, args.dmso_label)

    dmso_df = df[dmso_mask]
    logger.info("Found %d DMSO wells.", dmso_df.shape[0])

    # Non-DMSO compound ids
    compounds = df.loc[~dmso_mask, args.cpd_id_col].dropna().astype(str).unique().tolist()
    logger.info("Found %d non-DMSO compounds for analysis.", len(compounds))

    # Standard column order for outputs
    standard_cols = [
        "feature", "stat", "raw_pvalue", "abs_median_diff", "emd",
        "med_query", "med_comp", "pvalue_bh", "reject_bh",
        "n_query_wells", "n_dmso_wells",
    ]
    group_cols = [
        "group", "mean_abs_median_diff", "min_raw_pvalue", "mean_emd",
        "min_pvalue_bh", "n_features_in_group", "n_features_sig_fdr",
    ]

    # Per-compound analysis
    for cpd_id in compounds:
        logger.info("==== Analysing compound: %s ====", cpd_id)
        query_df = df[(df[args.cpd_id_col].astype(str) == str(cpd_id)) & (~dmso_mask)]
        if query_df.empty:
            logger.warning("No wells found for %s, skipping.", cpd_id)
            continue

        # All features
        logger.info("Comparing %s to DMSO across all features.", cpd_id)
        all_stats = compare_distributions(query_df, dmso_df, feature_cols, logger, test=args.test)
        all_stats["n_query_wells"] = query_df.shape[0]
        all_stats["n_dmso_wells"] = dmso_df.shape[0]
        all_stats = add_fdr(all_stats, alpha=args.fdr_alpha, logger=logger)
        all_stats = all_stats.sort_values(["reject_bh", "abs_median_diff"], ascending=[False, False])

        # Top features
        # if we want all features out, then uncomment
        # top_all = all_stats.head(args.top_features).copy()
        # all_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_top_features.tsv")
        # all_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_top_features.xlsx")
        # reorder_and_write(top_all, all_tsv, standard_cols, logger, "tsv")
        # reorder_and_write(top_all, all_xlsx, standard_cols, logger, "excel")

        # Significant features
        sig_all = all_stats[all_stats["reject_bh"]].copy()
        if not sig_all.empty:
            sig_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_significant_features.tsv")
            sig_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_significant_features.xlsx")
            reorder_and_write(sig_all, sig_tsv, standard_cols, logger, "tsv")
            reorder_and_write(sig_all, sig_xlsx, standard_cols, logger, "excel")

        # Acrosome-only features
        if acrosome_feats:
            logger.info("Comparing %s to DMSO across acrosome features.", cpd_id)
            acro_stats = compare_distributions(query_df, dmso_df, acrosome_feats, logger, test=args.test)
            acro_stats["n_query_wells"] = query_df.shape[0]
            acro_stats["n_dmso_wells"] = dmso_df.shape[0]
            acro_stats = add_fdr(acro_stats, alpha=args.fdr_alpha, logger=logger)
            acro_stats = acro_stats.sort_values(["reject_bh", "abs_median_diff"], ascending=[False, False])

            # top_acro = acro_stats.head(args.top_features).copy()
            # acro_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_top_features.tsv")
            # acro_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_top_features.xlsx")
            # reorder_and_write(top_acro, acro_tsv, standard_cols, logger, "tsv")
            # reorder_and_write(top_acro, acro_xlsx, standard_cols, logger, "excel")

            sig_acro = acro_stats[acro_stats["reject_bh"]].copy()
            if not sig_acro.empty:
                sig_acro_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_features.tsv")
                sig_acro_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_features.xlsx")
                reorder_and_write(sig_acro, sig_acro_tsv, standard_cols, logger, "tsv")
                reorder_and_write(sig_acro, sig_acro_xlsx, standard_cols, logger, "excel")

        # Acrosome group summary (based on *all* features’ stats)
        if "acrosome" in group_map:
            logger.info("Summarising acrosome group for %s.", cpd_id)
            acro_group = {"acrosome": group_map["acrosome"]}
            # Use the all-features stats table so FDR columns exist
            acro_group_stats = group_feature_stats(all_stats, acro_group, fdr_alpha=args.fdr_alpha, logger=logger)

            acro_grp_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_group.tsv")
            acro_grp_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_group.xlsx")
            reorder_and_write(acro_group_stats, acro_grp_tsv, group_cols, logger, "tsv")
            reorder_and_write(acro_group_stats, acro_grp_xlsx, group_cols, logger, "excel")

            # “Significant” group by min_raw_pvalue threshold
            sig_grp = acro_group_stats[acro_group_stats["min_raw_pvalue"] <= args.fdr_alpha].copy()
            if not sig_grp.empty:
                sig_grp_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_group.tsv")
                sig_grp_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_group.xlsx")
                reorder_and_write(sig_grp, sig_grp_tsv, group_cols, logger, "tsv")
                reorder_and_write(sig_grp, sig_grp_xlsx, group_cols, logger, "excel")

    logger.info("Batch comparison complete.")


if __name__ == "__main__":
    main()
