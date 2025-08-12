#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Attribution & Statistical Comparison for CLIPn Neighbourhoods
---------------------------------------------------------------------

For each query compound, this script compares well-level **numeric** features
against background sets and nearest neighbours:

1) Query vs DMSO/background (per-feature tests: Mann–Whitney U or KS)
   - Effect sizes: median difference, Wasserstein distance
   - p-values and BH–FDR (across ALL tested features per comparison)
   - Top-N enriched and depleted features

2) Query vs each of its nearest neighbours (optional)
   - Same statistics per neighbour pair

3) Optional feature grouping (e.g., compartment/channel) for compact summaries

Input files are TSV or CSV; all outputs are **TSV** (no comma-separated outputs).

Typical usage
-------------
python explain_feature_driven_results.py \
    --ungrouped_list features_manifest.tsv \
    --query_ids queries.txt \
    --nn_file nearest_neighbours.tsv \
    --output_dir out/explain \
    --test mw \
    --top_features 15 \
    --nn_per_query 10 \
    --feature_group_file feature_groups.tsv

Required columns
----------------
- Well-level feature tables must contain:
  cpd_id (and numeric features; metadata columns are ignored)
- Nearest-neighbour table must contain:
  cpd_id (or query_id), neighbour_id, distance

Outputs (per query)
-------------------
- <query>/query_vs_background_stats.tsv
- <query>/top_enriched_features.tsv
- <query>/top_depleted_features.tsv
- <query>/nn_compare/<neighbour>_stats.tsv    (if --nn_file provided)
- <query>/group_summary.tsv                    (if --feature_group_file provided)

Notes
-----
- Harmonises feature files by intersecting shared columns.
- Fills NaNs in features with column medians (logged).
- Uses two-sided tests by default.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp, wasserstein_distance


# -------------------------- Logging ------------------------------------------


def setup_logger(*, output_dir: str, log_name: str = "explain_features.log") -> logging.Logger:
    """
    Configure file and console logging.

    Parameters
    ----------
    output_dir : str
        Output directory to write log file.
    log_name : str
        Log filename placed in output_dir.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / log_name

    logger = logging.getLogger("explain_features")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fileh = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(stream)
    logger.addHandler(fileh)
    logger.info("Logging to %s", log_path)
    return logger


# -------------------------- I/O & utils --------------------------------------


def read_table(*, path: str) -> pd.DataFrame:
    """
    Read TSV/CSV with automatic delimiter detection.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    return pd.read_csv(path, sep=None, engine="python")


def load_feature_manifest(*, list_or_single: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load one or more well-level feature files and harmonise columns.

    Parameters
    ----------
    list_or_single : str
        A TSV/CSV with a 'path' column pointing to multiple TSVs,
        or a single TSV of features.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Harmonised concatenated table.
    """
    try:
        meta = read_table(path=list_or_single)
        if "path" in meta.columns:
            logger.info("Detected manifest with %d feature files.", len(meta))
            dfs = []
            for p in meta["path"]:
                df = read_table(path=str(p))
                dfs.append(df)
            common = set(dfs[0].columns)
            for d in dfs[1:]:
                common &= set(d.columns)
            if not common:
                raise ValueError("No common columns across feature files.")
            keep = sorted(common)
            dfs = [d[keep] for d in dfs]
            out = pd.concat(dfs, ignore_index=True)
            logger.info("Harmonised features shape: %s", out.shape)
            return out
    except Exception as e:
        logger.info("Not a manifest or failed to parse as such (%s). Loading as single TSV.", e)

    out = read_table(path=list_or_single)
    logger.info("Loaded single feature table: shape=%s", out.shape)
    return out


def ensure_numeric_features(*, df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keep numeric feature columns and impute NaNs with column medians.

    Parameters
    ----------
    df : pd.DataFrame
        Input well-level table.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    (df_num, cols) : Tuple[pd.DataFrame, List[str]]
        Clean numeric-only feature matrix and column names.
    """
    if "cpd_id" not in df.columns:
        raise ValueError("Feature table must contain 'cpd_id' column.")
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("No numeric feature columns found.")
    df_num = df[["cpd_id"] + numeric].copy()
    nans = df_num[numeric].isna().sum().sum()
    if int(nans) > 0:
        logger.warning("Found %d NaN values in features; imputing with column medians.", int(nans))
        med = df_num[numeric].median(axis=0)
        df_num[numeric] = df_num[numeric].fillna(value=med)
    return df_num, numeric


def parse_query_ids(*, arg: str) -> List[str]:
    """
    Parse query IDs from comma-separated string or file.

    Parameters
    ----------
    arg : str
        Comma list or path to file with one ID per line.

    Returns
    -------
    List[str]
        Query identifiers.
    """
    if os.path.isfile(arg):
        with open(arg) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]


def load_neighbours(*, nn_file: str, query_id: str, n_top: int, logger: logging.Logger) -> List[str]:
    """
    Return top-N neighbour IDs for a given query.

    Parameters
    ----------
    nn_file : str
        Nearest-neighbour TSV/CSV.
    query_id : str
        Query compound identifier.
    n_top : int
        Top-N neighbours to select.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    List[str]
        Neighbour compound IDs.
    """
    nn = read_table(path=nn_file)
    qcol = "cpd_id" if "cpd_id" in nn.columns else ("query_id" if "query_id" in nn.columns else None)
    if qcol is None or "neighbour_id" not in nn.columns:
        raise ValueError("NN table must contain 'cpd_id' (or 'query_id') and 'neighbour_id'.")
    hits = nn[nn[qcol].astype(str).str.upper() == str(query_id).upper()].copy()
    if "distance" in hits.columns:
        hits = hits.sort_values("distance", ascending=True)
    out = hits["neighbour_id"].astype(str).unique().tolist()[:n_top]
    logger.info("Selected %d neighbours for %s: %s", len(out), query_id, out)
    return out


def benjamini_hochberg(*, pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR.

    Parameters
    ----------
    pvals : np.ndarray
        Array of p-values.

    Returns
    -------
    np.ndarray
        Adjusted q-values (same shape).
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty_like(p)
    q[order] = p[order] * n / ranks
    q = np.minimum.accumulate(q[order[::-1]])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def run_two_sample_test(*, a: np.ndarray, b: np.ndarray, test: str) -> float:
    """
    Two-sample test p-value.

    Parameters
    ----------
    a : np.ndarray
        Sample A values.
    b : np.ndarray
        Sample B values.
    test : str
        'mw' (Mann–Whitney U) or 'ks' (Kolmogorov–Smirnov).

    Returns
    -------
    float
        Two-sided p-value.
    """
    if test == "mw":
        res = mannwhitneyu(a, b, alternative="two-sided")
        return float(res.pvalue)
    res = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return float(res.pvalue)


def summarise_groups(
    *,
    stats_df: pd.DataFrame,
    feature_group_map: Dict[str, str],
    top_n: int
) -> pd.DataFrame:
    """
    Summarise per-feature statistics into groups by mean absolute effect.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Per-feature stats with columns ['feature','median_diff','emd','p','q'].
    feature_group_map : Dict[str, str]
        Mapping feature -> group.
    top_n : int
        Top-N groups to retain.

    Returns
    -------
    pd.DataFrame
        Group summary.
    """
    tmp = stats_df.copy()
    tmp["group"] = tmp["feature"].map(feature_group_map).fillna("ungrouped")
    tmp["abs_effect"] = tmp["median_diff"].abs()
    grp = (
        tmp.groupby("group", as_index=False)
        .agg(
            n_features=("feature", "count"),
            mean_abs_effect=("abs_effect", "mean"),
            median_abs_effect=("abs_effect", "median"),
            min_q=("q", "min"),
        )
        .sort_values(["mean_abs_effect", "n_features"], ascending=[False, False])
        .head(top_n)
    )
    return grp


def write_tsv(*, df: pd.DataFrame, path: str | Path, logger: logging.Logger) -> None:
    """
    Write a DataFrame to TSV.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : str | Path
        Output path.
    logger : logging.Logger
        Logger instance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=path, sep="\t", index=False)
    logger.info("Wrote %d rows -> %s", len(df), path)


# -------------------------- Core comparisons ---------------------------------


def per_feature_stats(
    *,
    df: pd.DataFrame,
    feature_cols: List[str],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    test: str
) -> pd.DataFrame:
    """
    Compute per-feature statistics between two groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing features.
    feature_cols : List[str]
        Numeric feature columns.
    mask_a : np.ndarray
        Boolean mask for group A rows.
    mask_b : np.ndarray
        Boolean mask for group B rows.
    test : str
        'mw' or 'ks'.

    Returns
    -------
    pd.DataFrame
        Stats per feature: feature, median_A, median_B, median_diff, emd, p, q.
    """
    rows = []
    a_idx = np.where(mask_a)[0]
    b_idx = np.where(mask_b)[0]
    for feat in feature_cols:
        a = df[feat].values[a_idx]
        b = df[feat].values[b_idx]
        med_a = float(np.median(a))
        med_b = float(np.median(b))
        med_diff = med_a - med_b
        emd = float(wasserstein_distance(a, b))
        p = run_two_sample_test(a=a, b=b, test=test)
        rows.append(
            {
                "feature": feat,
                "median_A": med_a,
                "median_B": med_b,
                "median_diff": med_diff,
                "emd": emd,
                "p": p,
            }
        )
    out = pd.DataFrame(rows)
    out["q"] = benjamini_hochberg(pvals=out["p"].values)
    out = out.sort_values("q", ascending=True)
    return out


def run_query_background(
    *,
    df_num: pd.DataFrame,
    feature_cols: List[str],
    query_id: str,
    test: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compare query vs background (non-query, or optional DMSO subset if present).

    Parameters
    ----------
    df_num : pd.DataFrame
        Numeric-only well-level table (must include 'cpd_id').
    feature_cols : List[str]
        Numeric feature columns.
    query_id : str
        Query compound identifier.
    test : str
        'mw' or 'ks'.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Stats per feature.
    """
    is_query = df_num["cpd_id"].astype(str).str.upper() == str(query_id).upper()

    # Optional: if a DMSO background column exists, prefer that as background
    # Otherwise, use all non-query wells
    if "cpd_type" in df_num.columns:
        is_dmso = df_num["cpd_type"].astype(str).str.upper().str.contains("DMSO")
        if is_dmso.any():
            mask_b = is_dmso.values
            logger.info("Using DMSO wells as background (n=%d).", int(mask_b.sum()))
        else:
            mask_b = ~is_query.values
            logger.info("Using all non-query wells as background (n=%d).", int(mask_b.sum()))
    else:
        mask_b = ~is_query.values
        logger.info("Using all non-query wells as background (n=%d).", int(mask_b.sum()))

    mask_a = is_query.values
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    logger.info("Query wells: %d; background wells: %d", n_a, n_b)

    if n_a < 3 or n_b < 10:
        logger.warning("Too few wells for robust stats (query<3 or background<10). Proceeding anyway.")

    stats = per_feature_stats(df=df_num, feature_cols=feature_cols, mask_a=mask_a, mask_b=mask_b, test=test)
    return stats


def run_query_vs_neighbours(
    *,
    df_num: pd.DataFrame,
    feature_cols: List[str],
    query_id: str,
    neighbour_ids: List[str],
    test: str,
    out_dir: str,
    logger: logging.Logger
) -> None:
    """
    Compare query vs each neighbour compound individually.

    Parameters
    ----------
    df_num : pd.DataFrame
        Numeric feature table with 'cpd_id'.
    feature_cols : List[str]
        Numeric feature columns.
    query_id : str
        Query compound identifier.
    neighbour_ids : List[str]
        Neighbour compound IDs.
    test : str
        'mw' or 'ks'.
    out_dir : str
        Output directory for per-neighbour TSVs.
    logger : logging.Logger
        Logger instance.
    """
    qmask = df_num["cpd_id"].astype(str).str.upper() == str(query_id).upper()
    nn_dir = Path(out_dir) / "nn_compare"
    nn_dir.mkdir(parents=True, exist_ok=True)

    for nid in neighbour_ids:
        nmask = df_num["cpd_id"].astype(str).str.upper() == str(nid).upper()
        if qmask.sum() < 3 or nmask.sum() < 3:
            logger.info("Skipping neighbour %s due to low sample counts (query=%d, neighbour=%d).",
                        nid, int(qmask.sum()), int(nmask.sum()))
            continue
        stats = per_feature_stats(df=df_num, feature_cols=feature_cols, mask_a=qmask.values, mask_b=nmask.values, test=test)
        out = nn_dir / f"{query_id}_vs_{nid}_stats.tsv"
        write_tsv(df=stats, path=out, logger=logger)


# -------------------------- Grouping -----------------------------------------


def load_feature_groups(*, feature_group_file: str, logger: logging.Logger) -> Dict[str, str]:
    """
    Load feature->group mapping from CSV/TSV.

    Parameters
    ----------
    feature_group_file : str
        Path to mapping file with columns: feature, group.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Dict[str, str]
        Mapping from feature name to group label.
    """
    df = read_table(path=feature_group_file)
    need = {"feature",
