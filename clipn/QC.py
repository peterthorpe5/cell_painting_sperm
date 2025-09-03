#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature QC: Categorical‑Like Audit, Variance Diagnostics, and Summary Table

This standalone script audits feature columns in a wide table and produces:

1) A **per‑feature audit** table with: ``n_unique``, uniqueness ratio,
   integer‑rate, variance, missingness, flags for *categorical‑like* and
   *low‑variance*, and a consolidated **decision**
   (``keep``/``drop_categorical_like``/``drop_low_var``).
2) Shortlists of ``categorical_like_features.tsv`` and
   ``low_variance_features.tsv``.
3) A **multi‑page PDF** with diagnostics (now with more descriptive titles):
      • Variance histogram (log‑scaled x‑axis) with threshold annotation.
      • Uniqueness‑ratio histogram with threshold annotation.
      • Integer‑rate histogram.
      • Missingness histogram.
      • Scatter: log10(variance) vs uniqueness ratio (points coloured by flags).
      • Optional PCA scree plot (explained variance & cumulative) on a sample.
      • Value histograms for up to *N* flagged categorical‑like features.
4) A **feature summary table** ``feature_summary.tsv`` with common descriptive
   statistics per feature: number of entries (non‑NA), number of NAs, total
   rows, missing fraction, number of distinct values, min, max, mean, median,
   standard deviation, and simple **type inference**
   (``binary``, ``low_cardinality_numeric``, ``likely_ordinal``,
   ``numeric_continuous``).

The input may be TSV/CSV (``.gz`` transparently supported). The script attempts
to select feature columns by dtype (numeric), with optional explicit metadata
columns to exclude and a technical‑name blocklist (e.g., ``ImageNumber``).

Examples
--------
Use defaults and write outputs into ``qc_out``::

    python QC.py \
        --input /path/to/table.tsv.gz \
        --out qc_out \
        --variance_threshold 0.05 \
        --metadata_cols cpd_id cpd_type Plate_Metadata Well_Metadata Library \
        --drop_technical

Customise categorical‑like heuristics and add PCA::

    python QC.py \
        --input table.csv \
        --out qc_out \
        --low_card_unique_max 12 \
        --low_card_ratio_max 0.03 \
        --pca --pca_components 30 --pca_sample_rows 25000
"""
from __future__ import annotations

import argparse
import gzip
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional for PCA
try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except Exception:  # pragma: no cover - optional
    _SKLEARN_OK = False


# -------------------------
# Constants / name heuristics
# -------------------------

TECHNICAL_FEATURE_BLOCKLIST = {
    "ImageNumber",
    "Number_Object_Number",
    "ObjectNumber",
    "TableNumber",
}

METADATA_REGEX = re.compile(
    r"""(?ix)
        ( ^metadata($|_)         # Metadata*, *_Metadata
        | _metadata$
        | ^filename_             # FileName_*
        | ^pathname_             # PathName_*
        | ^url_                  # URL_*
        | ^parent_               # Parent_*
        | ^children_             # Children_*
        | (^|_)imagenumber$      # ImageNumber (optionally prefixed)
        | ^number_object_number$
        | ^objectnumber$
        | ^tablenumber$
        )
    """
)


# -------------------------
# Dataclasses
# -------------------------

@dataclass
class AuditConfig:
    """Configuration for the audit, diagnostics, and summaries.

    Parameters
    ----------
    variance_threshold : float
        Features with variance strictly below this are flagged as low‑variance.
    low_card_unique_max : int
        Flag *categorical‑like* if ``n_unique <= low_card_unique_max``.
    low_card_ratio_max : float
        Flag *categorical‑like* if ``n_unique / n_non_na <= low_card_ratio_max``.
    count_binary_as_categorical : bool
        If ``True``, features with exactly two unique non‑null values are
        flagged categorical‑like even if thresholds above would not.
    feature_include_regex : Optional[str]
        If provided, only columns whose names match this pattern are considered
        as features (after dtype filters). Useful to constrain to a feature
        prefix.
    feature_exclude_regex : Optional[str]
        If provided, exclude columns whose names match this pattern.
    drop_technical : bool
        Exclude known technical counters (e.g., ``ImageNumber``) from features.
    metadata_cols : Sequence[str]
        Column names to always treat as metadata (never as features).
    n_value_hists : int
        Number of example *categorical‑like* columns to plot as value
        histograms.
    seed : int
        Random seed for sampling (e.g., PCA sampling, example feature
        selection).
    do_pca : bool
        Whether to attempt an optional PCA scree plot.
    pca_components : int
        Maximum number of PCA components to compute/plot.
    pca_sample_rows : int
        Max rows to sample (without replacement) for PCA to keep memory bounded.
    title_prefix : Optional[str]
        Optional custom text to prepend to each plot title for context.
    """

    variance_threshold: float = 0.05
    low_card_unique_max: int = 10
    low_card_ratio_max: float = 0.02
    count_binary_as_categorical: bool = True
    feature_include_regex: Optional[str] = None
    feature_exclude_regex: Optional[str] = None
    drop_technical: bool = False
    metadata_cols: Sequence[str] = ()
    n_value_hists: int = 12
    seed: int = 0
    do_pca: bool = False
    pca_components: int = 20
    pca_sample_rows: int = 50000
    title_prefix: Optional[str] = None


# -------------------------
# I/O helpers
# -------------------------

def detect_delimiter(path: str | Path) -> str:
    """Detect the delimiter for a small text file.

    Prefers tab if both comma and tab are present; supports gzip ``.gz``.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    str
        ``"\t"`` or ``,"``.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(4096)
    if "\t" in sample and "," in sample:
        return "\t"
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return "\t"


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV with automatic delimiter and gzip handling.

    Parameters
    ----------
    path : str or Path
        Path to CSV/TSV (optionally ``.gz``).

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """
    sep = detect_delimiter(path)
    try:
        return pd.read_csv(path, sep=sep, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, sep=sep, engine="python")


# -------------------------
# Feature selection & metrics
# -------------------------

def _looks_like_metadata(name: str) -> bool:
    """Return ``True`` if a column name is metadata/housekeeping‑like.

    Parameters
    ----------
    name : str
        Column name.

    Returns
    -------
    bool
        Whether it matches metadata/housekeeping heuristics.
    """
    if name in TECHNICAL_FEATURE_BLOCKLIST:
        return True
    return bool(METADATA_REGEX.search(str(name).lower()))


def select_feature_columns(df: pd.DataFrame, cfg: AuditConfig) -> List[str]:
    """Infer numeric feature columns from a table.

    Parameters
    ----------
    df : pandas.DataFrame
        Input wide table.
    cfg : AuditConfig
        Audit configuration.

    Returns
    -------
    list of str
        Ordered list of feature column names.
    """
    candidate = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    # Drop explicit metadata cols
    meta_set = set(map(str, cfg.metadata_cols))
    candidate = [c for c in candidate if c not in meta_set]

    # Drop technical/metadata‑like if requested
    if cfg.drop_technical:
        candidate = [c for c in candidate if not _looks_like_metadata(c)]

    # Regex include/exclude
    if cfg.feature_include_regex:
        inc = re.compile(cfg.feature_include_regex)
        candidate = [c for c in candidate if inc.search(str(c))]
    if cfg.feature_exclude_regex:
        exc = re.compile(cfg.feature_exclude_regex)
        candidate = [c for c in candidate if not exc.search(str(c))]

    return candidate


def feature_metrics(x: pd.Series) -> Tuple[int, float, float, float, int]:
    """Compute basic discreteness/quality metrics for one feature.

    Parameters
    ----------
    x : pandas.Series
        Numeric series.

    Returns
    -------
    tuple
        ``(n_unique, uniqueness_ratio, integer_rate, variance, n_missing)``.
    """
    s = x.dropna()
    n_non_na = int(s.shape[0])
    n_unique = int(s.nunique(dropna=True))
    uniqueness_ratio = (n_unique / n_non_na) if n_non_na > 0 else 0.0

    # integer‑rate: fraction of values that are close to an integer
    # (tolerance 1e‑9 to be robust to float storage of ints)
    if n_non_na > 0:
        frac_part = np.abs(s.values - np.round(s.values))
        integer_rate = float(np.mean(frac_part < 1e-9))
    else:
        integer_rate = 0.0

    variance = float(np.nanvar(s.values, ddof=1)) if n_non_na > 1 else 0.0
    n_missing = int(x.isna().sum())
    return n_unique, uniqueness_ratio, integer_rate, variance, n_missing


def audit_features(df: pd.DataFrame, feature_cols: Sequence[str], cfg: AuditConfig) -> pd.DataFrame:
    """Audit a set of feature columns and flag potential categorical‑like ones.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_cols : sequence of str
        Columns to audit.
    cfg : AuditConfig
        Audit configuration.

    Returns
    -------
    pandas.DataFrame
        Per‑feature audit with metrics and ``decision`` column.
    """
    rows = []
    for col in feature_cols:
        n_u, u_ratio, int_rate, var, n_miss = feature_metrics(df[col])
        n_non_na = int(df[col].shape[0] - n_miss)

        # Flags
        is_constant = (n_u <= 1)
        is_binary = (n_u == 2)
        low_card = (n_u <= cfg.low_card_unique_max) or (u_ratio <= cfg.low_card_ratio_max)
        cat_like = is_constant or (cfg.count_binary_as_categorical and is_binary) or low_card
        low_var = (var < cfg.variance_threshold)

        decision = "keep"
        if cat_like:
            decision = "drop_categorical_like"
        if low_var:
            decision = "drop_low_var" if decision == "keep" else decision + "+low_var"

        rows.append({
            "feature": col,
            "n_unique": n_u,
            "uniqueness_ratio": u_ratio,
            "integer_rate": int_rate,
            "variance": var,
            "n_missing": n_miss,
            "n_non_missing": n_non_na,
            "flag_constant": is_constant,
            "flag_binary": is_binary,
            "flag_low_card": low_card,
            "flag_categorical_like": cat_like,
            "flag_low_variance": low_var,
            "decision": decision,
        })

    audit = pd.DataFrame(rows).sort_values([
        "flag_categorical_like", "variance", "n_unique"
    ], ascending=[False, True, True])
    return audit


# -------------------------
# Feature summary table
# -------------------------

def infer_simple_type(n_unique: int, integer_rate: float, uniqueness_ratio: float,
                      low_card_unique_max: int) -> str:
    """Infer a simple feature type from discreteness metrics.

    Parameters
    ----------
    n_unique : int
        Count of distinct non‑NA values.
    integer_rate : float
        Fraction of values that are integer‑like.
    uniqueness_ratio : float
        Distinct/entries ratio among non‑NA values.
    low_card_unique_max : int
        Threshold for low cardinality.

    Returns
    -------
    str
        One of ``binary``, ``low_cardinality_numeric``, ``likely_ordinal``,
        or ``numeric_continuous``.
    """
    if n_unique == 2:
        return "binary"
    if n_unique <= max(3, low_card_unique_max):
        return "low_cardinality_numeric"
    # Many distinct integer‑like values → ordinal scale likely
    if integer_rate > 0.98 and uniqueness_ratio < 0.25:
        return "likely_ordinal"
    return "numeric_continuous"


def build_feature_summary(df: pd.DataFrame, feature_cols: Sequence[str], cfg: AuditConfig) -> pd.DataFrame:
    """Construct a descriptive statistics table for features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing numeric features.
    feature_cols : sequence of str
        Feature column names to summarise.
    cfg : AuditConfig
        Audit configuration (used for type inference thresholds).

    Returns
    -------
    pandas.DataFrame
        Summary statistics per feature (TSV‑friendly).
    """
    rows: List[dict] = []
    for col in feature_cols:
        x = df[col]
        s = x.dropna()
        n_total = int(x.shape[0])
        n_na = int(x.isna().sum())
        n_non_na = n_total - n_na
        n_unique = int(s.nunique(dropna=True))
        uniq_ratio = (n_unique / n_non_na) if n_non_na else 0.0
        int_rate = 0.0
        if n_non_na:
            frac_part = np.abs(s.values - np.round(s.values))
            int_rate = float(np.mean(frac_part < 1e-9))
        summary_type = infer_simple_type(n_unique, int_rate, uniq_ratio, cfg.low_card_unique_max)
        # Numeric stats
        _min = float(np.nanmin(s)) if n_non_na else np.nan
        _max = float(np.nanmax(s)) if n_non_na else np.nan
        _mean = float(np.nanmean(s)) if n_non_na else np.nan
        _median = float(np.nanmedian(s)) if n_non_na else np.nan
        _std = float(np.nanstd(s, ddof=1)) if n_non_na > 1 else np.nan
        rows.append({
            "feature": col,
            "entries": n_non_na,
            "n_missing": n_na,
            "total_rows": n_total,
            "missing_fraction": (n_na / n_total) if n_total else 0.0,
            "n_unique": n_unique,
            "uniqueness_ratio": uniq_ratio,
            "integer_rate": int_rate,
            "min": _min,
            "max": _max,
            "mean": _mean,
            "median": _median,
            "std": _std,
            "inferred_type": summary_type,
        })
    out = pd.DataFrame(rows).sort_values(["inferred_type", "n_unique"]).reset_index(drop=True)
    return out


# -------------------------
# Plotting helpers
# -------------------------

def _safe_title(base: str, cfg: AuditConfig, suffix: Optional[str] = None) -> str:
    """Compose a descriptive plot title with optional global prefix.

    Parameters
    ----------
    base : str
        Core title text.
    cfg : AuditConfig
        Audit configuration (may include ``title_prefix``).
    suffix : str or None
        Optional trailing context.

    Returns
    -------
    str
        Generated title string.
    """
    parts: List[str] = []
    if cfg.title_prefix:
        parts.append(str(cfg.title_prefix))
    parts.append(base)
    if suffix:
        parts.append(str(suffix))
    return " — ".join(parts)


def _safe_log10(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute ``log10(max(x, eps))`` element‑wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.
    eps : float, optional
        Small positive number to clip at, by default ``1e-12``.

    Returns
    -------
    numpy.ndarray
        Log‑scaled values.
    """
    return np.log10(np.clip(x, eps, None))


def plot_variance_hist(ax: plt.Axes, var: np.ndarray, threshold: float, cfg: AuditConfig) -> None:
    """Plot a histogram of per‑feature variances with a log‑scaled x‑axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    var : numpy.ndarray
        Per‑feature variance values.
    threshold : float
        Vertical guide line location.
    cfg : AuditConfig
        Audit configuration.
    """
    ax.hist(var, bins=60)
    ax.set_xscale("log")
    ax.axvline(threshold, linestyle="--")
    ax.set_title(_safe_title(
        "Per‑feature variance (log x‑axis)", cfg,
        suffix=f"threshold={threshold:g}; features={len(var)}"
    ))
    ax.set_xlabel("variance")
    ax.set_ylabel("count")


def plot_uniqueness_hist(ax: plt.Axes, u_ratio: np.ndarray, low_card_ratio_max: float, cfg: AuditConfig) -> None:
    """Histogram of uniqueness ratio (n_unique / n_non_missing).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    u_ratio : numpy.ndarray
        Uniqueness ratios.
    low_card_ratio_max : float
        Threshold line to indicate low‑cardinality region.
    cfg : AuditConfig
        Audit configuration.
    """
    ax.hist(u_ratio, bins=60)
    ax.axvline(low_card_ratio_max, linestyle="--")
    ax.set_title(_safe_title(
        "Uniqueness ratio distribution", cfg,
        suffix=f"low‑cardinality if ≤ {low_card_ratio_max:g}"
    ))
    ax.set_xlabel("uniqueness ratio")
    ax.set_ylabel("count")


def plot_integer_rate_hist(ax: plt.Axes, int_rate: np.ndarray, cfg: AuditConfig) -> None:
    """Histogram of integer‑rate per feature.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    int_rate : numpy.ndarray
        Integer rates per feature.
    cfg : AuditConfig
        Audit configuration.
    """
    ax.hist(int_rate, bins=60)
    ax.set_title(_safe_title(
        "Integer‑like value fraction per feature", cfg,
        suffix=f"median={np.median(int_rate):.3f}"
    ))
    ax.set_xlabel("fraction of integer‑like values")
    ax.set_ylabel("count")


def plot_missingness_hist(ax: plt.Axes, miss_frac: np.ndarray, cfg: AuditConfig) -> None:
    """Histogram of feature missingness fraction.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    miss_frac : numpy.ndarray
        Fraction of missing values per feature.
    cfg : AuditConfig
        Audit configuration.
    """
    ax.hist(miss_frac, bins=60)
    ax.set_title(_safe_title(
        "Missingness per feature", cfg,
        suffix=f"median={np.median(miss_frac):.3f}"
    ))
    ax.set_xlabel("fraction missing")
    ax.set_ylabel("count")


def plot_scatter_var_vs_unique(ax: plt.Axes, var: np.ndarray, u_ratio: np.ndarray,
                               cat_mask: np.ndarray, thr: float, cfg: AuditConfig) -> None:
    """Scatter plot: log10(variance) vs uniqueness ratio.

    Points coloured by whether they are flagged categorical‑like.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    var : numpy.ndarray
        Per‑feature variances.
    u_ratio : numpy.ndarray
        Per‑feature uniqueness ratios.
    cat_mask : numpy.ndarray
        Boolean mask for categorical‑like features.
    thr : float
        Variance threshold (drawn as a vertical guide in *linear* space).
    cfg : AuditConfig
        Audit configuration.
    """
    x = _safe_log10(var)
    ax.scatter(x[~cat_mask], u_ratio[~cat_mask], alpha=0.6, s=10, label="kept")
    ax.scatter(x[cat_mask], u_ratio[cat_mask], alpha=0.8, s=12, label="categorical‑like")
    ax.axvline(math.log10(max(thr, 1e-12)), linestyle="--")
    ax.set_title(_safe_title(
        "log10(variance) vs uniqueness ratio", cfg,
        suffix=f"variance threshold={thr:g}; flagged={int(cat_mask.sum())}"
    ))
    ax.set_xlabel("log10(variance)")
    ax.set_ylabel("uniqueness ratio")
    ax.legend()


def plot_value_hist_grid(pdf: PdfPages, df: pd.DataFrame, features: Sequence[str],
                         cfg: AuditConfig, max_plots: int = 12) -> None:
    """Append a grid of value histograms for selected features to the PDF.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Open ``PdfPages`` handle to append figures to.
    df : pandas.DataFrame
        Source table for values.
    features : sequence of str
        Feature column names to plot.
    cfg : AuditConfig
        Audit configuration.
    max_plots : int
        Maximum number of histograms (features) to render.
    """
    feats = list(features)[:max_plots]
    if not feats:
        return

    n = len(feats)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, feats):
        try:
            ax.hist(df[col].dropna().values, bins=50)
            ax.set_title(str(col))
        except Exception:
            ax.text(0.5, 0.5, f"Error plotting {col}", ha="center")
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(_safe_title("Value distributions for categorical‑like features",
                             cfg, suffix=f"showing {len(feats)} of {len(features)}"))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# -------------------------
# PCA (optional)
# -------------------------

def try_pca_scree(pdf: PdfPages, df: pd.DataFrame, feature_cols: Sequence[str], cfg: AuditConfig) -> None:
    """Optionally compute and append a PCA scree plot to the PDF.

    This step is skipped if scikit‑learn is unavailable or if there are too
    few rows/columns. Rows are sampled (without replacement) up to
    ``cfg.pca_sample_rows``. Missing values are imputed with the column median
    and features are standardised to zero mean and unit variance.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Open ``PdfPages`` handle to append figures to.
    df : pandas.DataFrame
        Input table.
    feature_cols : sequence of str
        Features to include in PCA.
    cfg : AuditConfig
        Audit configuration.
    """
    if not _SKLEARN_OK:
        return

    X = df.loc[:, feature_cols]
    # Drop columns with all‑NaN to avoid degenerate behaviour
    X = X.dropna(axis=1, how="all")
    if X.shape[1] < 2 or X.shape[0] < 5:
        return

    # Sample rows for tractability
    if X.shape[0] > cfg.pca_sample_rows:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(X.index.values, size=cfg.pca_sample_rows, replace=False)
        X = X.loc[idx]

    # Impute medians then standardise
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_imp = imputer.fit_transform(X.values)
    X_std = scaler.fit_transform(X_imp)

    n_comp = int(min(cfg.pca_components, X_std.shape[1]))
    if n_comp < 2:
        return

    pca = PCA(n_components=n_comp, random_state=cfg.seed)
    pca.fit(X_std)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(evr) + 1), evr, marker="o", label="per‑component")
    ax.plot(range(1, len(cum) + 1), cum, marker="o", label="cumulative")
    ax.set_title(_safe_title("PCA scree plot (standardised features)", cfg))
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance ratio")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# -------------------------
# Main driver
# -------------------------

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as TSV, creating parent directories if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to save.
    path : pathlib.Path
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def run(input_path: str | Path, out_dir: str | Path, cfg: AuditConfig) -> None:
    """Execute the audit, render diagnostics, and save summaries.

    Parameters
    ----------
    input_path : str or Path
        Path to the input table (TSV/CSV, optionally ``.gz``).
    out_dir : str or Path
        Directory where outputs will be written.
    cfg : AuditConfig
        Audit configuration.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)

    # Optional context in plot titles: file stem and row/feature counts
    title_prefix = cfg.title_prefix
    if title_prefix is None:
        title_prefix = f"{Path(str(input_path)).name}"
    cfg = AuditConfig(**{**cfg.__dict__, "title_prefix": title_prefix})

    # Feature selection
    feature_cols = select_feature_columns(df, cfg)
    if not feature_cols:
        raise SystemExit("No numeric feature columns found after exclusions.")

    # Build audit and summary
    audit = audit_features(df, feature_cols, cfg)
    summary = build_feature_summary(df, feature_cols, cfg)

    # Persist audit, summary, and shortlists
    write_tsv(audit, out / "feature_discreteness_audit.tsv")
    write_tsv(summary, out / "feature_summary.tsv")
    write_tsv(
        audit.loc[audit["flag_categorical_like"], ["feature"]],
        out / "categorical_like_features.tsv",
    )
    write_tsv(
        audit.loc[audit["flag_low_variance"], ["feature"]],
        out / "low_variance_features.tsv",
    )

    # Multi‑page PDF
    pdf_path = out / "variance_diagnostics_log.pdf"
    with PdfPages(pdf_path) as pdf:
        # 1) Variance histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_variance_hist(ax, audit["variance"].to_numpy(), cfg.variance_threshold, cfg)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2) Uniqueness ratio hist
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_uniqueness_hist(ax, audit["uniqueness_ratio"].to_numpy(), cfg.low_card_ratio_max, cfg)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3) Integer‑rate hist
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_integer_rate_hist(ax, audit["integer_rate"].to_numpy(), cfg)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4) Missingness hist
        miss_frac = (audit["n_missing"].to_numpy() / np.maximum(1, audit["n_missing"].to_numpy() + audit["n_non_missing"].to_numpy()))
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_missingness_hist(ax, miss_frac, cfg)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 5) Scatter log10(var) vs uniqueness ratio
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        plot_scatter_var_vs_unique(
            ax,
            audit["variance"].to_numpy(),
            audit["uniqueness_ratio"].to_numpy(),
            audit["flag_categorical_like"].to_numpy(),
            cfg.variance_threshold,
            cfg,
        )
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 6) PCA scree (optional)
        if cfg.do_pca:
            try_pca_scree(pdf, df, feature_cols, cfg)

        # 7) Example value histograms for flagged categorical‑like features
        flagged = audit.loc[audit["flag_categorical_like"], "feature"].tolist()
        if flagged:
            plot_value_hist_grid(pdf, df, flagged, cfg, max_plots=cfg.n_value_hists)

    # Also save features‑used list for reproducibility
    pd.Series(feature_cols, name="feature").to_csv(out / "features_considered.tsv", sep="\t", index=False)

    print(f"Wrote audit to: {out}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command‑line arguments.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument list to parse (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Feature QC: categorical‑like audit, variance diagnostics, and summary table")
    p.add_argument("--input", required=True, help="Path to TSV/CSV (optionally .gz)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--variance_threshold", type=float, default=0.05, help="Low‑variance threshold (default: 0.05)")
    p.add_argument("--low_card_unique_max", type=int, default=10, help="Flag categorical‑like if n_unique <= this (default: 10)")
    p.add_argument("--low_card_ratio_max", type=float, default=0.02, help="Flag categorical‑like if n_unique/n_non_na <= this (default: 0.02)")
    p.add_argument("--no_binary", action="store_true", help="Do NOT force binary‑valued columns to categorical‑like")
    p.add_argument("--feature_include_regex", default=None, help="Only include columns whose names match this regex")
    p.add_argument("--feature_exclude_regex", default=None, help="Exclude columns whose names match this regex")
    p.add_argument("--drop_technical", action="store_true", help="Exclude known technical/housekeeping columns")
    p.add_argument("--metadata_cols", nargs="*", default=[], help="Columns to always treat as metadata (space‑separated)")
    p.add_argument("--n_value_hists", type=int, default=12, help="How many categorical‑like features to plot as value histograms (default: 12)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    p.add_argument("--pca", action="store_true", help="Also compute a PCA scree plot on a row sample")
    p.add_argument("--pca_components", type=int, default=20, help="Max PCA components to compute (default: 20)")
    p.add_argument("--pca_sample_rows", type=int, default=50000, help="Max rows sampled for PCA (default: 50k)")
    p.add_argument("--export_filtered_tsv",
                    type=str,
                    default=None,
                    help="If set, write a TSV with categorical‑like columns removed (no imputation/standardisation)."
                )
    p.add_argument("--export_metadata",
                    nargs="*",
                    default=["cpd_id","cpd_type","Library","Plate_Metadata","Well_Metadata"],
                    help="Metadata columns to always keep in the export."
                )
    p.add_argument("--title_prefix",
                   type=str,
                   default=None,
                   help="Optional string to prepend to plot titles (e.g., run label)")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for command‑line execution.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument list (defaults to ``sys.argv[1:]``).
    """
    args = parse_args(argv)
    cfg = AuditConfig(
        variance_threshold=args.variance_threshold,
        low_card_unique_max=args.low_card_unique_max,
        low_card_ratio_max=args.low_card_ratio_max,
        count_binary_as_categorical=not args.no_binary,
        feature_include_regex=args.feature_include_regex,
        feature_exclude_regex=args.feature_exclude_regex,
        drop_technical=args.drop_technical,
        metadata_cols=args.metadata_cols,
        n_value_hists=args.n_value_hists,
        seed=args.seed,
        do_pca=args.pca,
        pca_components=args.pca_components,
        pca_sample_rows=args.pca_sample_rows,
        title_prefix=args.title_prefix,
    )

    run(args.input, args.out, cfg)


if __name__ == "__main__":
    main()
